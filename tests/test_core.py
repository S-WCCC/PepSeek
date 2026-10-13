import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from pepseek.features import SequenceFeatures, FeatureStore
from pepseek.data import Dataset, collate_chain, collate
from pepseek.demo import create_demo
from pepseek.model import UnifiedPepSeek, ModelConfig, score_matrix
from pepseek.losses import interval_huber, binary_masked, region_loss, contrastive_loss, multitask_loss
from pepseek.metrics import binary_metrics
from pepseek.io import read_json, write_json, write_jsonl, jsonl

torch.set_num_threads(2)


@pytest.fixture
def example():
    builder = SequenceFeatures(toy_dim=16)
    p = [builder.build({"id": "p", "sequence": "ACDEFGHIKLMN"}, "protein")]
    q = [builder.build({"id": "q", "sequence": "ACDEF"}, "peptide")]
    inputs = {"protein": collate_chain(p, "protein"), "peptide": collate_chain(q, "peptide")}
    model = UnifiedPepSeek(ModelConfig(esm_dim=16, hidden=24, pair_dim=12, encoder_layers=1,
                        interaction_layers=2, protein_slots=3, peptide_slots=2, contact_rank=8, dropout=0))
    return builder, inputs, model


def test_fast_and_detailed_scores_identical(example):
    _, inputs, model = example
    model.eval()
    with torch.no_grad():
        full, fast = model(inputs, True, True), model(inputs, False)
    torch.testing.assert_close(full["score"], fast["score"], rtol=0, atol=0)
    torch.testing.assert_close(full["affinity"], fast["affinity"], rtol=0, atol=0)


@pytest.mark.parametrize("head", ["score", "affinity", "protein_interface_logits", "peptide_interface_logits", "contact_logits", "distance_log_probs", "region_membership_logits"])
def test_every_head_trains_interaction_backbone(example, head):
    _, inputs, model = example
    model.zero_grad(set_to_none=True)
    out = model(inputs, True, True)
    out[head].square().mean().backward()
    for block in model.interaction:
        grads = [p.grad for p in block.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0


def test_native_labels_rejected_from_forward(example):
    _, inputs, model = example
    with pytest.raises(ValueError):
        model({**inputs, "native_contact": torch.zeros(1)})


def test_cached_encoding_and_direct_forward_agree(example):
    _, inputs, model = example
    model.eval()
    with torch.no_grad():
        p, q = model.encode(inputs["protein"], "protein"), model.encode(inputs["peptide"], "peptide")
        torch.testing.assert_close(model(inputs, False)["score"], score_matrix(model, p, q).flatten())


def test_geometry_rotation_translation_invariance(example):
    _, inputs, model = example
    model.eval()
    q = inputs["peptide"]
    torch.manual_seed(3)
    q["conformer_coords"] = torch.randn_like(q["conformer_coords"])
    q["geometry_mask"][:] = True
    with torch.no_grad():
        first = model(inputs, False)["score"]
        rot, _ = torch.linalg.qr(torch.randn(3, 3))
        q["conformer_coords"] = q["conformer_coords"] @ rot + 15.
        second = model(inputs, False)["score"]
    torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-5)


def test_interval_exact_and_censored():
    pred = torch.tensor([2., 2., 2., 2.], requires_grad=True)
    loss = interval_huber(pred, torch.tensor([1., 1., 3., -torch.inf]), torch.tensor([1., 3., torch.inf, 1.]))
    torch.testing.assert_close(loss, torch.tensor([.5, 0., .5, .5]))
    loss.sum().backward()
    torch.testing.assert_close(pred.grad, torch.tensor([1., 0., -1., 1.]))


def test_unknown_structural_labels_have_no_gradient():
    x = torch.tensor([0., 0.], requires_grad=True)
    loss = binary_masked(x, torch.tensor([1., torch.nan])); loss.backward()
    assert x.grad[1] == 0 and x.grad[0] < 0


def test_region_permutation_invariance():
    x = torch.tensor([[3., 2., -3., -2.], [-3., -2., 3., 2.], [0., 0., 0., 0.]], requires_grad=True)
    target = torch.tensor([[1., 1., 0., 0.], [0., 0., 1., 1.]])
    torch.testing.assert_close(region_loss(x, target), region_loss(x, target.flip(0)))


def test_multi_positive_contrastive_and_empty_mask():
    scores = torch.tensor([[2., 2., -2.], [-2., -2., 2.]], requires_grad=True)
    positives = torch.tensor([[1, 1, 0], [0, 0, 1]], dtype=torch.bool)
    loss = contrastive_loss(scores, positives)
    assert torch.isfinite(loss)
    loss.backward(); assert torch.isfinite(scores.grad).all()
    assert contrastive_loss(scores, torch.zeros_like(positives)) == 0


def test_binary_metrics_ties():
    m = binary_metrics([0, 1, 0, 1], [1, 1, 1, 1])
    assert m["auroc"] == .5 and m["average_precision"] == .5
    assert binary_metrics([1, 1], [0, 1])["auroc"] is None


def test_data_seal_and_cross_partition_pair_rejected(tmp_path):
    path = create_demo(tmp_path / "dataset")
    ds = Dataset(path)
    assert ds.seal()["counts"] == {"train": 6, "valid": 3, "test": 3}
    records = jsonl(ds.root / "pairs.jsonl")
    duplicate = {**records[0], "id": "duplicate", "split": "test"}
    write_jsonl(ds.root / "pairs.jsonl", records + [duplicate])
    with pytest.raises(ValueError, match="crosses partitions"):
        Dataset(path)


def test_feature_recipe_mismatch_fails(tmp_path):
    store = FeatureStore(tmp_path / "a.sqlite", {"test": 1}); store.close()
    with pytest.raises(ValueError, match="mismatch"):
        FeatureStore(tmp_path / "a.sqlite", {"test": 2})


def test_missing_modalities_and_labels_stay_separate(tmp_path):
    ds = Dataset(create_demo(tmp_path / "dataset"))
    builder = SequenceFeatures(toy_dim=16)
    store = FeatureStore(tmp_path / "features.sqlite", builder.config)
    for side in ds.entities:
        for e in ds.entities[side].values():
            store.put(store.key(e, side), builder.build(e, side))
    batch = collate(ds, ds.split("train")[:3], store)
    assert set(batch) == {"protein", "peptide"}
    assert not batch["peptide"]["geometry_mask"].any()
    assert batch["peptide"]["conformer_mask"].any(1).all()
    store.close()


def test_batched_padding_does_not_change_score(example):
    builder, inputs, model = example
    model.eval()
    p = [builder.build({"id": "p", "sequence": "ACDEFGHIKLMN"}, "protein"), builder.build({"id": "p2", "sequence": "ACDEFGHIKLMNPQRSTVWY"}, "protein")]
    q = [builder.build({"id": "q", "sequence": "ACDEF"}, "peptide"), builder.build({"id": "q2", "sequence": "ACDEFGHIK"}, "peptide")]
    with torch.no_grad():
        first = model(inputs, False)["score"][0]
        batch = model({"protein": collate_chain(p, "protein"), "peptide": collate_chain(q, "peptide")}, False)["score"][0]
    torch.testing.assert_close(first, batch, atol=2e-6, rtol=2e-5)


def test_label_generation_masks_unresolved_residues(tmp_path):
    from pepseek.labels import structure_labels
    src, dst = tmp_path / "coords.npz", tmp_path / "labels.npz"
    np.savez(src, protein_heavy_atom_coords=np.array([[0., 0., 0.], [30., 0., 0.]]),
             peptide_heavy_atom_coords=np.array([[0., 0., 3.]]),
             protein_atom_residue=np.array([0, 2]), peptide_atom_residue=np.array([1]))
    structure_labels(src, dst, 3, 2)
    with np.load(dst) as z:
        assert z["contact"][0, 1] == 1 and z["contact"][2, 1] == 0
        assert np.isnan(z["contact"][1]).all() and np.isnan(z["contact"][:, 0]).all()
        assert z["regions"].shape == (1, 3)


def test_reliable_ranking_direction_and_filter(example):
    _, _, model = example
    prediction = torch.zeros(2, 5, requires_grad=True)
    rows = [{"protein_id": "p", "affinity": [{"endpoint": "pKd", "lower": y, "upper": y,
             "group_id": "assay", "reliable_ranking": True}]} for y in (7., 5.)]
    _, terms = multitask_loss({"score": torch.zeros(2, requires_grad=True), "affinity": prediction}, rows, [{}, {}], model.cfg.endpoints)
    terms["ranking"].backward()
    assert prediction.grad[0, 0] < 0 and prediction.grad[1, 0] > 0
    rows[1]["affinity"][0]["group_id"] = "another_assay"
    _, terms = multitask_loss({"score": torch.zeros(2), "affinity": prediction}, rows, [{}, {}], model.cfg.endpoints)
    assert "ranking" not in terms


def test_observed_region_metrics_are_permutation_invariant():
    from pepseek.metrics import observed_region_metrics
    pred = np.array([[.9, .9, .1, .1], [.1, .1, .9, .9]])
    target = np.array([[0., 0., 1., 1.], [1., 1., 0., 0.]])
    result = observed_region_metrics(pred, target)
    assert result["matched_dice"] == 1 and result["matched_iou"] == 1


def test_human_selenoprotein_and_extended_protein_tokens():
    builder = SequenceFeatures(toy_dim=16)
    f = builder.build({"id": "selenoprotein", "sequence": "ACDUOBZJX"}, "protein")
    assert f["esm"].shape == (9, 16)
    assert (f["aa"][3:] == 20).all()
    with pytest.raises(ValueError):
        builder.build({"id": "unmapped_modified_peptide", "sequence": "ACDU"}, "peptide")
