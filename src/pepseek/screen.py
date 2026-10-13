"""Disk-backed exact screening with the same interaction backbone in both directions."""
import csv
import heapq
import json
import time
from pathlib import Path
import numpy as np
import torch
from .data import collate_chain
from .engine import load_model
from .features import FeatureStore
from .io import digest, sha256, write_json, write_csv, provenance, read_json


def _check_checkpoint(checkpoint):
    seal_path = Path(checkpoint).parent / "SELECTION_SEAL.json"
    seal = read_json(seal_path)
    if seal["checkpoint_sha256"] != sha256(checkpoint):
        raise ValueError("Checkpoint does not match its selection seal")


def _batch_encoded(records, device):
    b, c, length, dim = len(records), max(len(r["conformer_mask"]) for r in records), max(len(r["mask"]) for r in records), records[0]["h"].shape[-1]
    out = {"h": torch.zeros(b, c, length, dim, device=device),
           "mask": torch.zeros(b, length, dtype=torch.bool, device=device),
           "conformer_mask": torch.zeros(b, c, dtype=torch.bool, device=device)}
    for i, r in enumerate(records):
        nc, nl = r["h"].shape[:2]
        out["h"][i, :nc, :nl] = torch.as_tensor(r["h"], device=device)
        out["mask"][i, :nl] = torch.as_tensor(r["mask"], device=device)
        out["conformer_mask"][i, :nc] = torch.as_tensor(r["conformer_mask"], device=device)
    return out


@torch.inference_mode()
def build_index(entities, side, features, checkpoint, output, device="cpu", batch_size=8):
    _check_checkpoint(checkpoint)
    if Path(output).exists():
        raise ValueError("Index already exists; rebuild into a new versioned path")
    model, state = load_model(checkpoint, device)
    raw = FeatureStore(features, readonly=True)
    if state["identity"]["feature_recipe"] != raw.config:
        raise ValueError("Raw features differ from training recipe")
    cfg = {"kind": "encoded_index", "schema": 1, "side": side, "checkpoint_sha256": sha256(checkpoint),
           "feature_recipe": raw.config, "dtype": "float32", "source": "independent_preinteraction_features"}
    index = FeatureStore(output, cfg)
    index.db.execute("CREATE TABLE entities (id TEXT PRIMARY KEY, feature_key TEXT NOT NULL, sequence_sha256 TEXT NOT NULL)")
    count, batch = 0, []
    def emit(records):
        raw_features = [raw.get(raw.key(e, side)) for e in records]
        enc = model.encode(collate_chain(raw_features, side, device), side)
        for i, (e, f) in enumerate(zip(records, raw_features)):
            n, c = len(f["aa"]), len(f.get("conformer_mask", [True]))
            arrays = {"h": enc["h"][i, :c, :n].float().cpu().numpy(),
                      "mask": enc["mask"][i, :n].cpu().numpy(),
                      "conformer_mask": enc["conformer_mask"][i, :c].cpu().numpy()}
            key = digest({"id": e["id"], "raw_key": raw.key(e, side), "checkpoint": cfg["checkpoint_sha256"]})
            index.put(key, arrays)
            index.db.execute("INSERT INTO entities VALUES (?, ?, ?)", (e["id"], key, digest(e["sequence"])))
        index.db.commit()
    try:
        for e in entities:
            batch.append(e); count += 1
            if len(batch) == batch_size:
                emit(batch); batch = []
        if batch:
            emit(batch)
        if count == 0:
            raise ValueError("Cannot build an empty candidate index")
        index.db.execute("INSERT INTO metadata VALUES ('complete', ?)", (json.dumps({"count": count}),))
        index.db.commit()
    finally:
        index.close(); raw.close()
    receipt = {"count": count, "config": cfg, "index_sha256": sha256(output), "provenance": provenance()}
    write_json(str(output) + ".receipt.json", receipt)
    return receipt


@torch.inference_mode()
def search(queries, query_side, features, checkpoint, index_path, output, device="cpu", batch_size=8, top_k=100, all_scores=False):
    _check_checkpoint(checkpoint)
    model, state = load_model(checkpoint, device)
    raw, index = FeatureStore(features, readonly=True), FeatureStore(index_path, readonly=True)
    expected_side = "peptide" if query_side == "protein" else "protein"
    if index.config["side"] != expected_side or index.config["checkpoint_sha256"] != sha256(checkpoint):
        raise ValueError("Wrong index direction or checkpoint")
    if raw.config != state["identity"]["feature_recipe"] or index.config["feature_recipe"] != raw.config:
        raise ValueError("Feature recipe mismatch")
    if index.db.execute("SELECT value FROM metadata WHERE key='complete'").fetchone() is None:
        raise ValueError("Index is incomplete; rebuild it")
    receipt = read_json(str(index_path) + ".receipt.json")
    if receipt["index_sha256"] != sha256(index_path):
        raise ValueError("Index checksum mismatch")
    if top_k < 1 or batch_size < 1:
        raise ValueError("top_k and batch_size must be positive")
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Search output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    summary, rankings, seen = [], [], set()
    for query in queries:
        if query["id"] in seen:
            raise ValueError("Duplicate query ID")
        seen.add(query["id"])
        began = time.perf_counter()
        enc = model.encode(collate_chain([raw.get(raw.key(query, query_side))], query_side, device), query_side)
        heap = []
        stream = open(out / (digest(query["id"]) + ".scores.csv"), "w", newline="", encoding="utf-8") if all_scores else None
        writer = csv.writer(stream) if stream else None
        if writer:
            writer.writerow(["query_id", "candidate_id", "score"])
        cursor = index.db.execute("SELECT id,feature_key FROM entities ORDER BY id")
        count = 0
        try:
            while True:
                entries = cursor.fetchmany(batch_size)
                if not entries:
                    break
                candidates = _batch_encoded([index.get(key) for _, key in entries], device)
                expanded = {k: v.expand(len(entries), *v.shape[1:]) for k, v in enc.items()}
                protein, peptide = (expanded, candidates) if query_side == "protein" else (candidates, expanded)
                scores = model.interact(protein, peptide, detail=False)["score"].float().cpu().tolist()
                for (item, _), score in zip(entries, scores):
                    if not np.isfinite(score):
                        raise FloatingPointError(f"Nonfinite candidate score: {item}")
                    # Deterministic ties: descending score, then descending candidate ID.
                    candidate = (score, item)
                    if len(heap) < top_k:
                        heapq.heappush(heap, candidate)
                    elif candidate > heap[0]:
                        heapq.heapreplace(heap, candidate)
                    if writer:
                        writer.writerow([query["id"], item, score])
                count += len(entries)
        finally:
            if stream:
                stream.close()
        ranking = [{"query_id": query["id"], "rank": i + 1, "candidate_id": item, "score": score}
                   for i, (score, item) in enumerate(sorted(heap, reverse=True))]
        rankings.extend(ranking)
        elapsed = time.perf_counter() - began
        summary.append({"query_id": query["id"], "candidates_scored": count, "seconds": elapsed,
                        "pairs_per_second": count / max(elapsed, 1e-9)})
    write_csv(out / "rankings.csv", rankings)
    write_json(out / "timing.json", summary)
    write_json(out / "RECEIPT.json", {"checkpoint_sha256": sha256(checkpoint), "index_sha256": receipt["index_sha256"],
                                    "query_side": query_side, "ranking": "score desc, candidate_id desc",
                                    "score_type": "uncalibrated learned interaction score", "algorithm": "exact candidate scan through shared interaction backbone",
                                    "queries": len(summary), "provenance": provenance()})
    raw.close(); index.close()
    return summary


def evaluate_retrieval(rankings, truth, output):
    from .metrics import retrieval_metrics
    truth_rows = read_json(truth)
    if not isinstance(truth_rows, dict):
        raise ValueError("Truth must map query IDs to lists of known positive candidate IDs")
    groups = {}
    with open(rankings, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            groups.setdefault(r["query_id"], []).append((int(r["rank"]), r["candidate_id"]))
    results = []
    for query, positives in truth_rows.items():
        if query not in groups:
            raise ValueError(f"Missing ranked query: {query}")
        ids = [item for _, item in sorted(groups[query])]
        result = retrieval_metrics(ids, positives)
        for k in (1, 10, 100, 1000):
            if k > len(ids):
                result[f"recall_at_{k}"] = None
        results.append({"query_id": query, **result})
    macro = {k: float(np.mean([r[k] for r in results if r[k] is not None]))
             for k in results[0] if k not in ("query_id", "known_positives") and any(r[k] is not None for r in results)} if results else {}
    write_json(output, {"query_count": len(results), "macro": macro, "per_query": results,
                        "truth_sha256": sha256(truth), "rankings_sha256": sha256(rankings)})
    return macro
