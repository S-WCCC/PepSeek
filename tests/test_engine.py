import os
import signal
from pathlib import Path
import pytest
import torch
from pepseek.demo import create_demo
from pepseek.data import Dataset
from pepseek.features import FeatureStore, SequenceFeatures, prepare_entities
from pepseek.io import read_json, write_json, sha256
from pepseek.engine import train, evaluate


def test_interrupt_resume_seal_and_test_boundary(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    manifest = create_demo(tmp_path / "fixture")
    ds = Dataset(manifest)
    builder = SequenceFeatures(toy_dim=16)
    features = tmp_path / "features.sqlite"
    store = FeatureStore(features, builder.config)
    for side in ds.entities:
        prepare_entities(ds.entities[side].values(), side, store, builder)
    store.close()
    train_config = tmp_path / "train.json"
    config = read_json(project / "configs/train_smoke.json")
    config["epochs"] = 3; config["max_steps"] = 6
    write_json(train_config, config)
    seen = []
    original_labels = Dataset.labels
    def labels(self, row):
        seen.append(row["split"])
        return original_labels(self, row)
    monkeypatch.setattr(Dataset, "labels", labels)
    import pepseek.engine as engine
    original_batch = engine.batch_loss
    requested = [False]
    def stop_after_first(*args, **kwargs):
        result = original_batch(*args, **kwargs)
        training = kwargs.get("training", args[8] if len(args) > 8 else True)
        # Engine passes validation's training=False as the eighth positional argument.
        if len(args) > 7:
            training = args[7]
        if training and not requested[0]:
            requested[0] = True
            os.kill(os.getpid(), signal.SIGTERM)
        return result
    monkeypatch.setattr(engine, "batch_loss", stop_after_first)
    run = tmp_path / "run"
    result = train(manifest, features, project / "configs/model_smoke.json", train_config, run)
    assert result["status"] == "interrupted"
    assert (run / "INTERRUPTED.json").exists() and not (run / "SELECTION_SEAL.json").exists()
    assert "test" not in seen
    monkeypatch.setattr(engine, "batch_loss", original_batch)
    train(manifest, features, project / "configs/model_smoke.json", train_config, run, resume=True)
    seal = read_json(run / "SELECTION_SEAL.json")
    assert seal["checkpoint_sha256"] == sha256(run / "best.pt") and not seal["test_labels_used"]
    assert "test" not in seen
    with pytest.raises(ValueError, match="finalized"):
        train(manifest, features, project / "configs/model_smoke.json", train_config, run, resume=True)
    result = evaluate(manifest, features, run / "best.pt", tmp_path / "evaluation")
    assert result["pairs"] == 3 and "test" in seen
    seal["checkpoint_sha256"] = "tampered"
    write_json(run / "SELECTION_SEAL.json", seal)
    with pytest.raises(ValueError, match="seal mismatch"):
        evaluate(manifest, features, run / "best.pt", tmp_path / "should_fail")
