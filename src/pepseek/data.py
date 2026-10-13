"""Strict dataset contract and variable-length batching with separate labels."""
from pathlib import Path
import numpy as np
import torch
from .features import sequence
from .io import read_json, jsonl, relative_file, sha256, digest, write_json


class Dataset:
    def __init__(self, manifest):
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        self.spec = read_json(manifest)
        if self.spec.get("schema_version") != 1:
            raise ValueError("dataset.json requires schema_version=1")
        self.entities = {}
        for side in ("protein", "peptide"):
            records = jsonl(relative_file(self.root, self.spec[side + "s"]))
            ids = [r["id"] for r in records]
            if len(set(ids)) != len(ids):
                raise ValueError(f"Duplicate {side} entity IDs")
            for e in records:
                if e["sequence"] != sequence(e["sequence"], side == "peptide",e.get("chemistry_mode")=="unknown"):
                    raise ValueError("Sequences must be uppercase and whitespace-free")
                if e.get("components") and ''.join(e['components']) != e['sequence']:
                    raise ValueError("Component boundaries do not match sequence")
            self.entities[side] = {r["id"]: r for r in records}
        self.pairs = jsonl(relative_file(self.root, self.spec["pairs"]))
        self.validate()

    def validate(self):
        seen, pair_splits, cluster_splits = set(), {}, {}
        label_signs = {}
        policy = self.spec.get("split_policy", "pair")
        if policy not in ("pair", "entity_disjoint", "cluster_disjoint"):
            raise ValueError("Unknown split_policy")
        for row in self.pairs:
            if row["id"] in seen or row["split"] not in ("train", "valid", "test"):
                raise ValueError("Duplicate item ID or invalid split")
            seen.add(row["id"])
            p, q = (self.entities[s][row[s + "_id"]] for s in ("protein", "peptide"))
            pair = (tuple(p.get('components',[p['sequence']])), digest({"sequence": q["sequence"], "smiles": q.get("smiles"), "map": q.get("atom_residue"), "chemical_identity_id":q.get('chemical_identity_id')}))
            pair_splits.setdefault(pair, set()).add(row["split"])
            if row.get("binding") is not None:
                if row["binding"] not in (0, 1):
                    raise ValueError("binding must be 0, 1, or absent")
                label_signs.setdefault(pair, set()).add(row["binding"])
            for side, e in (("protein", p), ("peptide", q)):
                if policy != "pair":
                    key = (side, "sequence", e["sequence"])
                    cluster_splits.setdefault(key, set()).add(row["split"])
                if policy == "cluster_disjoint":
                    if not e.get("cluster_id"):
                        raise ValueError("cluster_disjoint requires externally audited cluster_id for every entity")
                    cluster_splits.setdefault((side, "cluster", e["cluster_id"]), set()).add(row["split"])
            for a in row.get("affinity", []):
                if a.get("lower") is None and a.get("upper") is None:
                    raise ValueError("Affinity requires at least one interval bound")
                lo, hi = a.get("lower"), a.get("upper")
                if any(x is not None and not np.isfinite(x) for x in (lo, hi)):
                    raise ValueError("Use null for infinite interval bounds")
                if lo is not None and hi is not None and lo > hi:
                    raise ValueError("Reversed affinity interval")
            if row.get("structure_labels"):
                path = relative_file(self.root, row["structure_labels"])
                if not path.is_file():
                    raise ValueError(f"Missing labels: {path}")
            for view in row.get('structure_views',[]):
                if not relative_file(self.root,view['path']).is_file():
                    raise ValueError('Missing structural view')
        if any(len(v) > 1 for v in pair_splits.values()):
            raise ValueError("The same chemical pair crosses partitions")
        if any(len(v) > 1 for v in cluster_splits.values()):
            raise ValueError("Entity/cluster leakage across partitions")
        if any(len(v) > 1 for v in label_signs.values()):
            raise ValueError("Conflicting binary labels for a chemical pair; resolve experimental domains first")

    def split(self, name):
        return [r for r in self.pairs if r["split"] == name]

    def training_rows(self, epoch, seed=17):
        """One entry per physical pair; rotate structural observations each epoch."""
        result=[]
        for row in self.split('train'):
            r=dict(row);views=r.get('structure_views',[])
            if views:
                offset=int(digest([r['id'],seed])[:16],16)%len(views)
                r['structure_labels']=views[(offset+epoch)%len(views)]['path']
            result.append(r)
        return result

    def evaluation_rows(self, name):
        result=[]
        for row in self.split(name):
            views=row.get('structure_views',[])
            if not views:result.append(dict(row,view_weight=1.0));continue
            for i,view in enumerate(views):
                result.append(dict(row,id=row['id']+'__view'+str(i),pair_id=row['id'],structure_labels=view['path'],view_weight=1/len(views)))
        return result

    def seal(self):
        paths = {self.manifest}
        for key in ("proteins", "peptides", "pairs"):
            paths.add(relative_file(self.root, self.spec[key]))
        for row in self.pairs:
            if row.get("structure_labels"):
                paths.add(relative_file(self.root, row["structure_labels"]))
            for view in row.get('structure_views',[]):paths.add(relative_file(self.root,view['path']))
        files = {str(p.relative_to(self.root)).replace("\\", "/"): sha256(p) for p in sorted(paths)}
        return {"schema_version": 1, "files": files, "sha256": digest(files),
                "split_policy": self.spec.get("split_policy", "pair"),
                "counts": {s: len(self.split(s)) for s in ("train", "valid", "test")},
                "cluster_method": self.spec.get("cluster_method", "not provided; homology separation not established")}

    def labels(self, row):
        if not row.get('structure_labels') and row.get('structure_views'):
            row=dict(row,structure_labels=row['structure_views'][0]['path'])
        if not row.get("structure_labels"):
            return {}
        with np.load(relative_file(self.root, row["structure_labels"]), allow_pickle=False) as z:
            out = {k: z[k].copy() for k in z.files}
        accepted = {"contact", "protein_interface", "peptide_interface", "distance_bins", "regions"}
        if set(out) - accepted:
            raise ValueError("Unknown label arrays; native coordinates must be converted to labels separately")
        lp = len(self.entities["protein"][row["protein_id"]]["sequence"])
        lq = len(self.entities["peptide"][row["peptide_id"]]["sequence"])
        shapes = {"contact": (lp, lq), "distance_bins": (lp, lq),
                  "protein_interface": (lp,), "peptide_interface": (lq,)}
        for key, shape in shapes.items():
            if key in out and out[key].shape != shape:
                raise ValueError(f"Label shape mismatch: {key} for {row['id']}")
        for key in ("contact", "protein_interface", "peptide_interface", "regions"):
            if key in out:
                observed = out[key][np.isfinite(out[key])]
                if not np.isin(observed, [0, 1]).all():
                    raise ValueError("Binary structural labels must be 0/1/NaN")
        if "regions" in out and (out["regions"].ndim != 2 or out["regions"].shape[1] != lp):
            raise ValueError("regions must have shape [observed_regions, protein_length]")
        if "distance_bins" in out:
            bins = out["distance_bins"]
            if not np.isfinite(bins).all() or (bins < -1).any() or not np.equal(bins, np.floor(bins)).all():
                raise ValueError("Distance labels must be integer classes or -1 for missing")
        return out


def collate_chain(features, side, device="cpu"):
    b, length = len(features), max(len(f["aa"]) for f in features)
    dim = features[0]["esm"].shape[-1]
    out = {"esm": torch.zeros(b, length, dim), "aa": torch.full((b, length), 20, dtype=torch.long),
           "mask": torch.zeros(b, length, dtype=torch.bool), "position_features":torch.zeros(b,length,3)}
    edges, atom_features, atom_edges, atom_residue, atom_weights = [], [], [], [], []
    if side == "peptide":
        c = max(len(f["conformer_mask"]) for f in features)
        out.update(conformer_coords=torch.zeros(b, c, length, 3),
                   geometry_mask=torch.zeros(b, c, length, dtype=torch.bool),
                   conformer_mask=torch.zeros(b, c, dtype=torch.bool))
    atom_offset = 0
    for i, f in enumerate(features):
        n = len(f["aa"])
        for key in ("esm", "aa"):
            out[key][i, :n] = torch.from_numpy(f[key])
        out["mask"][i, :n] = True
        if 'position_features' in f:out['position_features'][i,:n]=torch.from_numpy(f['position_features'])
        else:
            idx=torch.arange(n);out['position_features'][i,:n]=torch.stack([idx/max(n-1,1),torch.exp(-idx/4),torch.exp(-(n-1-idx)/4)],-1)
        edges.append(torch.from_numpy(f["edges"]) + i * length)
        if side == "peptide":
            c = len(f["conformer_mask"])
            out["conformer_coords"][i, :c, :n] = torch.from_numpy(f["conformer_coords"])
            out["geometry_mask"][i, :c, :n] = torch.from_numpy(f["geometry_mask"])
            out["conformer_mask"][i, :c] = torch.from_numpy(f["conformer_mask"])
            atom_features.append(torch.from_numpy(f["atom_features"]))
            atom_edges.append(torch.from_numpy(f["atom_edges"]) + atom_offset)
            atom_weights.append(torch.from_numpy(f["atom_edge_weight"]))
            atom_residue.append(torch.from_numpy(f["atom_residue"]) + i * length)
            atom_offset += len(f["atom_features"])
    out["edges"] = torch.cat(edges, 1)
    if side == "peptide":
        out.update(atom_features=torch.cat(atom_features), atom_edges=torch.cat(atom_edges, 1),
                   atom_residue=torch.cat(atom_residue), atom_edge_weight=torch.cat(atom_weights))
    return {k: v.to(device) for k, v in out.items()}


def collate(dataset, rows, store, device="cpu"):
    return {s: collate_chain([store.get(store.key(dataset.entities[s][r[s + "_id"]], s)) for r in rows], s, device)
            for s in ("protein", "peptide")}


def positive_pairs(rows):
    # Co-complex retrieval supervision is distinct from an experimental binding class.
    return {(r["protein_id"], r["peptide_id"]) for r in rows if r.get("binding") == 1 or r.get('contrastive_positive') is True}
