"""Public, explicit CLI. Importing the package never starts work or downloads assets."""
import argparse
import json
from pathlib import Path
import torch
from .io import jsonl, fasta, write_json, provenance


def entities(path):
    return fasta(path) if Path(path).suffix.lower() in (".fa", ".faa", ".fasta", ".fas") else iter(jsonl(path))


def parser():
    p = argparse.ArgumentParser(prog="pepseek", description="Unified protein-peptide interaction training and sequence screening")
    p.add_argument("--threads", type=int, default=4, help="CPU threads per process")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    a = sub.add_parser("make-demo"); a.add_argument("--output", required=True)
    a = sub.add_parser("make-labels")
    a.add_argument("--coordinates", required=True); a.add_argument("--output", required=True)
    a.add_argument("--protein-length", type=int, required=True); a.add_argument("--peptide-length", type=int, required=True)
    a.add_argument("--threshold", type=float, default=8.)
    a = sub.add_parser("audit"); a.add_argument("--dataset", required=True); a.add_argument("--output", required=True)
    a = sub.add_parser("prepare", help="Build immutable sequence-derived features from a dataset or entity list")
    a.add_argument("--dataset"); a.add_argument("--entities"); a.add_argument("--side", choices=("protein", "peptide"))
    a.add_argument("--features", required=True); a.add_argument("--esm-checkpoint"); a.add_argument("--toy-dim", type=int)
    a.add_argument("--conformers", type=int, default=0); a.add_argument("--seed", type=int, default=17)
    a.add_argument("--device", default="cpu"); a.add_argument("--window", type=int, default=1000); a.add_argument("--overlap", type=int, default=128)
    a = sub.add_parser("train")
    for name in ("dataset", "features", "model-config", "train-config", "output"):
        a.add_argument("--" + name, required=True)
    a.add_argument("--device", default="cpu"); a.add_argument("--resume", action="store_true")
    a = sub.add_parser("evaluate")
    for name in ("dataset", "features", "checkpoint", "output"):
        a.add_argument("--" + name, required=True)
    a.add_argument("--split", choices=("train", "valid", "test"), default="test")
    a.add_argument("--device", default="cpu"); a.add_argument("--batch-size", type=int, default=4)
    a = sub.add_parser("build-index")
    for name in ("entities", "features", "checkpoint", "output"):
        a.add_argument("--" + name, required=True)
    a.add_argument("--side", choices=("protein", "peptide"), required=True)
    a.add_argument("--device", default="cpu"); a.add_argument("--batch-size", type=int, default=8)
    a = sub.add_parser("predict")
    for name in ("proteins", "peptides", "pairs", "features", "checkpoint", "output"):
        a.add_argument("--" + name, required=True)
    a.add_argument("--device", default="cpu"); a.add_argument("--batch-size", type=int, default=4)
    a.add_argument("--score-only", action="store_true")
    a = sub.add_parser("search")
    for name in ("queries", "features", "checkpoint", "index", "output"):
        a.add_argument("--" + name, required=True)
    a.add_argument("--query-side", choices=("protein", "peptide"), required=True)
    a.add_argument("--device", default="cpu"); a.add_argument("--batch-size", type=int, default=8)
    a.add_argument("--top-k", type=int, default=100); a.add_argument("--all-scores", action="store_true")
    a = sub.add_parser("evaluate-retrieval")
    for name in ("rankings", "truth", "output"):
        a.add_argument("--" + name, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    torch.set_num_threads(args.threads)
    if args.command == "doctor":
        import rdkit, numpy, scipy, sys
        result = {**provenance(), "executable": sys.executable, "isolated_environment": sys.prefix != sys.base_prefix,
                  "numpy": numpy.__version__, "scipy": scipy.__version__, "rdkit": rdkit.__version__,
                  "cuda_available": torch.cuda.is_available(), "visible_gpus": torch.cuda.device_count()}
    elif args.command == "make-demo":
        from .demo import create_demo
        result = {"dataset": create_demo(args.output)}
    elif args.command == "make-labels":
        from .labels import structure_labels
        result = structure_labels(args.coordinates, args.output, args.protein_length, args.peptide_length, args.threshold)
    elif args.command == "audit":
        from .data import Dataset
        ds = Dataset(args.dataset)
        for row in ds.pairs:
            ds.labels(row)
        result = ds.seal(); write_json(args.output, result)
    elif args.command == "prepare":
        from .data import Dataset
        from .features import SequenceFeatures, FeatureStore, prepare_entities
        if bool(args.dataset) == bool(args.entities):
            raise ValueError("Use either --dataset or --entities, exclusively")
        builder = SequenceFeatures(args.esm_checkpoint, args.device, args.toy_dim, args.conformers,
                                   args.seed, args.window, args.overlap, args.threads)
        store = FeatureStore(args.features, builder.config)
        if args.dataset:
            ds = Dataset(args.dataset)
            result = {s: prepare_entities(ds.entities[s].values(), s, store, builder) for s in ("protein", "peptide")}
        else:
            if not args.side:
                raise ValueError("Entity preparation requires --side")
            result = prepare_entities(entities(args.entities), args.side, store, builder)
        store.close()
    elif args.command == "train":
        from .engine import train
        result = train(args.dataset, args.features, args.model_config, args.train_config, args.output, args.device, args.resume)
    elif args.command == "evaluate":
        from .engine import evaluate
        result = evaluate(args.dataset, args.features, args.checkpoint, args.output, args.split, args.device, args.batch_size)
    elif args.command == "build-index":
        from .screen import build_index
        result = build_index(entities(args.entities), args.side, args.features, args.checkpoint, args.output, args.device, args.batch_size)
    elif args.command == "predict":
        from .predict import predict
        result = predict(entities(args.proteins), entities(args.peptides), entities(args.pairs), args.features, args.checkpoint,
                         args.output, args.device, args.batch_size, not args.score_only)
    elif args.command == "search":
        from .screen import search
        result = search(entities(args.queries), args.query_side, args.features, args.checkpoint, args.index, args.output,
                        args.device, args.batch_size, args.top_k, args.all_scores)
    elif args.command == "evaluate-retrieval":
        from .screen import evaluate_retrieval
        result = evaluate_retrieval(args.rankings, args.truth, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
