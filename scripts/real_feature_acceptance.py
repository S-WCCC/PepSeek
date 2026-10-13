"""Bounded CPU check of official ESM, chemical conformers and full-size backbone."""
import argparse
import gc
import time
from pathlib import Path
import numpy as np
import torch
from pepseek.features import SequenceFeatures
from pepseek.data import collate_chain
from pepseek.model import ModelConfig, UnifiedPepSeek
from pepseek.io import read_json, sha256, write_json, provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise ValueError("Choose a fresh receipt path")
    torch.set_num_threads(4)
    began = time.perf_counter()
    builder = SequenceFeatures(args.checkpoint, device="cpu", conformers=2, threads=2, window=12, overlap=4)
    p = builder.build({"id": "acceptance_protein", "sequence": "ACDEFGHIKLMNPQRSTVWY"}, "protein")
    q = builder.build({"id": "acceptance_peptide", "sequence": "AGAG"}, "peptide")
    assert p["esm"].shape == (len("ACDEFGHIKLMNPQRSTVWY"), 1280) and np.isfinite(p["esm"]).all()
    assert q["geometry_mask"].any(), "Generated geometry path was not exercised"
    recipe = builder.config
    del builder
    gc.collect()
    project = Path(__file__).resolve().parents[1]
    model = UnifiedPepSeek(ModelConfig(**read_json(project / "configs/model_m.json")))
    inputs = {"protein": collate_chain([p], "protein"), "peptide": collate_chain([q], "peptide")}
    outputs = model(inputs, detail=True, distance=True)
    loss = sum(outputs[k].square().mean() for k in ("score", "affinity", "contact_logits", "protein_interface_logits", "peptide_interface_logits", "region_membership_logits", "distance_log_probs"))
    loss.backward()
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
    assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in b.parameters()) for b in model.interaction)
    model.eval()
    with torch.no_grad():
        full, fast = model(inputs, True), model(inputs, False)
        torch.testing.assert_close(full["score"], fast["score"], atol=0, rtol=0)
    write_json(args.output, {"status": "passed", "scientific_training": False, "device": "cpu", "esm_asset_sha256": sha256(args.checkpoint),
                            "feature_recipe": recipe, "long_sequence_windows_exercised": True,
                            "retained_generated_conformers": int(q["geometry_mask"].any(1).sum()),
                            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                            "all_heads_backward_finite": True, "all_interaction_blocks_receive_gradient": True,
                            "score_fast_detail_identical": True, "seconds": time.perf_counter() - began, "provenance": provenance()})
    print(args.output)


if __name__ == "__main__":
    main()
