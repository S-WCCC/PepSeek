"""Record tracked source hashes and environment; no private data is collected."""
import argparse
import subprocess
from pathlib import Path
from pepseek.io import sha256, write_json, provenance

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
files = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
record = {"provenance": provenance(), "source_files": {name: sha256(root / name) for name in files if name and (root / name).is_file()}}
lock = root / "runtime/deployment/requirements-resolved.txt"
if lock.exists():
    record["environment_lock_sha256"] = sha256(lock)
write_json(args.output, record)
