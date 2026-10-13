"""Run an isolated synthetic end-to-end acceptance without touching scientific data."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root, project = Path(args.output).resolve(), Path(__file__).resolve().parents[1]
    root.mkdir(parents=True, exist_ok=False)
    calls = []
    def call(*argv):
        cmd = [sys.executable, "-m", "pepseek", "--threads", "2", *map(str, argv)]
        log = root / f"{len(calls):02d}_{argv[0]}.log"
        with open(log, "w", encoding="utf-8") as f:
            process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
        calls.append({"command": cmd, "returncode": process.returncode, "log": log.name})
        if process.returncode:
            raise RuntimeError(f"Acceptance failed: {log}")
    dataset, features, run = root / "fixture/dataset.json", root / "features.sqlite", root / "run"
    call("make-demo", "--output", dataset.parent)
    call("audit", "--dataset", dataset, "--output", root / "data_seal.json")
    call("prepare", "--dataset", dataset, "--features", features, "--toy-dim", "16")
    call("train", "--dataset", dataset, "--features", features, "--model-config", project / "configs/model_smoke.json",
         "--train-config", project / "configs/train_smoke.json", "--output", run, "--device", args.device)
    call("evaluate", "--dataset", dataset, "--features", features, "--checkpoint", run / "best.pt",
         "--output", root / "evaluation", "--device", args.device)
    identity_pairs = root / "prediction_pairs.jsonl"
    identity_pairs.write_text(json.dumps({"id": "new_pair", "protein_id": "protein_0", "peptide_id": "peptide_0"}) + "\n")
    call("predict", "--proteins", dataset.parent / "proteins.jsonl", "--peptides", dataset.parent / "peptides.jsonl",
         "--pairs", identity_pairs, "--features", features, "--checkpoint", run / "best.pt", "--output", root / "prediction", "--device", args.device)
    for side in ("protein", "peptide"):
        call("build-index", "--entities", dataset.parent / (side + "s.jsonl"), "--side", side, "--features", features,
             "--checkpoint", run / "best.pt", "--output", root / (side + "s.sqlite"), "--device", args.device)
        query_side = "peptide" if side == "protein" else "protein"
        query_path = root / (query_side + "_query.jsonl")
        query_path.write_text((dataset.parent / (query_side + "s.jsonl")).read_text().splitlines()[0] + "\n")
        call("search", "--queries", query_path, "--query-side", query_side, "--features", features,
             "--checkpoint", run / "best.pt", "--index", root / (side + "s.sqlite"), "--output", root / ("search_" + side),
             "--top-k", "12", "--all-scores", "--device", args.device)
        truth = root / (side + "_truth.json")
        truth.write_text(json.dumps({query_side + "_0": [side + "_0"]}))
        call("evaluate-retrieval", "--rankings", root / ("search_" + side) / "rankings.csv",
             "--truth", truth, "--output", root / ("retrieval_metrics_" + side + ".json"))
    # Same physical pair must have the same score regardless of query direction.
    pair_scores = []
    for side in ("protein", "peptide"):
        with open(root / ("search_" + side) / "rankings.csv", newline="") as f:
            r = next(r for r in csv.DictReader(f) if r["candidate_id"] == side + "_0")
            pair_scores.append(float(r["score"]))
    with open(root / "prediction/predictions.csv", newline="") as f:
        pair_scores.append(float(next(csv.DictReader(f))["score"]))
    if max(pair_scores) - min(pair_scores) > 2e-5:
        raise AssertionError(f"Detailed and bidirectional cached scores differ: {pair_scores}")
    result = {"status": "passed", "synthetic": True, "device": args.device, "commands": calls, "same_pair_score_spread": max(pair_scores) - min(pair_scores),
              "meaning": "Software acceptance; no scientific performance claim or trained production checkpoint"}
    (root / "ACCEPTANCE.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
