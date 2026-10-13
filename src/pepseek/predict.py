"""Label-free pair prediction including residue-level outputs."""
from pathlib import Path
import numpy as np
import torch
from .data import collate_chain
from .engine import load_model
from .features import FeatureStore
from .io import digest, sha256, write_json, write_csv, provenance
from .screen import _check_checkpoint


@torch.inference_mode()
def predict(proteins, peptides, pairs, features, checkpoint, output, device="cpu", batch_size=4, detail=True):
    _check_checkpoint(checkpoint)
    entities = {"protein": {e["id"]: e for e in proteins}, "peptide": {e["id"]: e for e in peptides}}
    model, state = load_model(checkpoint, device)
    coverage = state["identity"].get("train_supervision_coverage", {})
    store = FeatureStore(features, readonly=True)
    if store.config != state["identity"]["feature_recipe"]:
        raise ValueError("Prediction feature recipe differs from training")
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Prediction output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    rows, batch = [], []
    seen = set()
    def emit(records):
        inputs = {s: collate_chain([store.get(store.key(entities[s][r[s + "_id"]], s)) for r in records], s, device)
                  for s in entities}
        pred = model(inputs, detail=detail)
        for i, r in enumerate(records):
            row = {"id": r["id"], "protein_id": r["protein_id"], "peptide_id": r["peptide_id"], "score": float(pred["score"][i])}
            row.update({ep: float(pred["affinity"][i, j]) if coverage.get("affinity", {}).get(ep, 0) else None
                        for j, ep in enumerate(model.cfg.endpoints)})
            if detail:
                lp = len(entities["protein"][r["protein_id"]]["sequence"])
                lq = len(entities["peptide"][r["peptide_id"]]["sequence"])
                filename = digest(r["id"]) + ".npz"
                np.savez_compressed(out / filename,
                    protein_interface_probability=pred["protein_interface_logits"][i, :lp].sigmoid().cpu().numpy(),
                    peptide_interface_probability=pred["peptide_interface_logits"][i, :lq].sigmoid().cpu().numpy(),
                    contact_probability=pred["contact_logits"][i, :lp, :lq].sigmoid().cpu().numpy(),
                    region_probability=pred["region_membership_logits"][i, :, :lp].sigmoid().cpu().numpy(),
                    conformer_weights=pred["conformer_weights"][i].cpu().numpy())
                row["detail_file"] = filename
            rows.append(row)
    for row in pairs:
        if set(row) != {"id", "protein_id", "peptide_id"}:
            raise ValueError("Prediction pairs accept identity columns only, never labels")
        if row["id"] in seen:
            raise ValueError("Duplicate pair ID")
        seen.add(row["id"]); batch.append(row)
        if len(batch) == batch_size:
            emit(batch); batch = []
    if batch:
        emit(batch)
    write_csv(out / "predictions.csv", rows)
    write_json(out / "RECEIPT.json", {"pairs": len(rows), "checkpoint_sha256": sha256(checkpoint),
                                    "input": "independently sequence-derived features", "residue_indexing": "zero-based sequence positions",
                                    "region_slots": "latent predicted memberships, not confirmed 3D pocket volumes",
                                    "train_supervision_coverage": coverage,
                                    "provenance": provenance()})
    store.close()
    return {"pairs": len(rows), "output": str(out)}
