"""Atomic artifacts and provenance. Paths inside datasets are always relative."""
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def relative_file(root, value):
    root = Path(root).resolve()
    path = (root / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Dataset path escapes its root: {value}")
    return path


def save_torch(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(obj, temp)
    os.replace(temp, path)


def load_torch(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=True)


def provenance():
    try:
        root = Path(__file__).resolve().parents[2]
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], stderr=subprocess.DEVNULL))
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return {"utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
            "git_dirty": dirty, "python": sys.version.split()[0], "torch": str(torch.__version__),
            "platform": platform.platform(), "cuda_runtime": torch.version.cuda}


def fasta(path):
    name, seq = None, []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield {"id": name, "sequence": "".join(seq)}
                name, seq = line[1:].split()[0], []
            else:
                if name is None:
                    raise ValueError("FASTA must begin with an identifier")
                seq.append(line.upper())
    if name is not None:
        yield {"id": name, "sequence": "".join(seq)}
