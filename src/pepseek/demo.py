"""Synthetic acceptance fixture, deliberately excluded from scientific datasets."""
from pathlib import Path
import numpy as np
from .io import write_json, write_jsonl


def create_demo(output):
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Demo output must be empty")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    alphabet = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    p, q, pairs = [], [], []
    for i in range(12):
        ps = "".join(rng.choice(alphabet, 14 + i % 3))
        qs = "".join(rng.choice(alphabet, 5 + i % 2))
        p.append({"id": f"protein_{i}", "sequence": ps, "cluster_id": f"demo_p_{i}"})
        q.append({"id": f"peptide_{i}", "sequence": qs, "cluster_id": f"demo_q_{i}"})
        split = "train" if i < 6 else "valid" if i < 9 else "test"
        row = {"id": f"pair_{i}", "protein_id": p[-1]["id"], "peptide_id": q[-1]["id"],
               "split": split, "binding": 0 if i % 3 == 1 else 1, "source": "synthetic_software_test"}
        # Include all three modalities and explicit negative pairs.
        if i % 3 != 2:
            row["affinity"] = [{"endpoint": "pKd", "lower": 5. + i / 10, "upper": 5. + i / 10}]
        if i % 3 != 1:
            contact = np.zeros((len(ps), len(qs)), np.float32)
            contact[2:5, 1:3] = 1
            contact[-1] = np.nan
            pi = np.where(np.isfinite(contact).any(1), np.nanmax(np.nan_to_num(contact), 1), np.nan)
            qi = np.nanmax(contact, 0)
            region = pi[None]
            bins = np.where(np.isfinite(contact), np.where(contact == 1, 0, 4), -1).astype(np.int64)
            label_path = root / "labels" / f"pair_{i}.npz"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(label_path, contact=contact, protein_interface=pi, peptide_interface=qi,
                                distance_bins=bins, regions=region)
            row["structure_labels"] = str(label_path.relative_to(root)).replace("\\", "/")
        pairs.append(row)
    write_jsonl(root / "proteins.jsonl", p); write_jsonl(root / "peptides.jsonl", q)
    write_jsonl(root / "pairs.jsonl", pairs)
    write_json(root / "dataset.json", {"schema_version": 1, "proteins": "proteins.jsonl", "peptides": "peptides.jsonl",
                                       "pairs": "pairs.jsonl", "split_policy": "cluster_disjoint",
                                       "cluster_method": "synthetic unique identities; software test only"})
    write_json(root / "README.json", {"synthetic": True, "purpose": "Software pipeline acceptance only; not biological evidence"})
    return str(root / "dataset.json")
