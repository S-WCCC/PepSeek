"""Sequence-derived inputs only. Experimental structure labels live elsewhere.

Raw features are content-addressed SQLite blobs; no per-million-entity directory
or all-in-memory embedding matrix is required. ESM covers long chains by overlap.
"""
import io
import json
import sqlite3
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from .io import digest, sha256

AA = "ACDEFGHIKLMNPQRSTVWYX"


def sequence(value, peptide=False):
    s = value.strip().upper()
    allowed = AA[:-1] if peptide else AA + "UBZOJ"
    if not s or any(c not in allowed for c in s):
        raise ValueError("Peptides require standard 20-AA sequences; proteins also allow X/U/O/B/Z/J. Supply explicit chemical identity for modifications.")
    return s


class FeatureStore:
    def __init__(self, path, config=None, readonly=False):
        self.path = Path(path)
        if readonly:
            self.db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.path)
            self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self.db.execute("CREATE TABLE IF NOT EXISTS features (key TEXT PRIMARY KEY, payload BLOB NOT NULL, sha256 TEXT NOT NULL)")
        row = self.db.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
        if row:
            self.config = json.loads(row[0])
            if config is not None and self.config != config:
                raise ValueError("Feature recipe mismatch; create a new store instead of mixing encoders")
        elif config is not None and not readonly:
            self.config = config
            self.db.execute("INSERT INTO metadata VALUES ('config', ?)", (json.dumps(config, sort_keys=True),))
            self.db.commit()
        else:
            raise ValueError("Missing feature-store configuration")

    def key(self, entity, side):
        return digest({"side": side, "sequence": entity["sequence"],
                       "smiles": entity.get("smiles"), "atom_residue": entity.get("atom_residue"),
                       "config": self.config})

    def get(self, key):
        row = self.db.execute("SELECT payload, sha256 FROM features WHERE key=?", (key,)).fetchone()
        if row is None:
            raise KeyError(f"Missing features {key}; run prepare first")
        import hashlib
        if hashlib.sha256(row[0]).hexdigest() != row[1]:
            raise ValueError("Feature checksum failure")
        with np.load(io.BytesIO(row[0]), allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    def contains(self, key):
        return self.db.execute("SELECT 1 FROM features WHERE key=?", (key,)).fetchone() is not None

    def put(self, key, arrays):
        import hashlib
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        payload = buffer.getvalue()
        self.db.execute("INSERT OR IGNORE INTO features VALUES (?, ?, ?)", (key, payload, hashlib.sha256(payload).hexdigest()))
        self.db.commit()

    def close(self):
        self.db.close()


class SequenceFeatures:
    def __init__(self, checkpoint=None, device="cpu", toy_dim=None, conformers=0,
                 seed=17, window=1000, overlap=128, threads=1):
        if conformers < 0 or window < 2 or not 0 <= overlap < window:
            raise ValueError("Invalid feature settings")
        self.device, self.conformers, self.seed = device, conformers, seed
        self.window, self.overlap, self.threads = window, overlap, threads
        if toy_dim is not None:
            if checkpoint is not None:
                raise ValueError("Toy and ESM features cannot be combined")
            self.dim = toy_dim
            self.table = np.random.default_rng(17).normal(size=(len(AA), toy_dim)).astype(np.float32)
            source = {"backend": "toy_for_software_tests_only", "dim": toy_dim}
        else:
            if checkpoint is None or not Path(checkpoint).is_file():
                raise ValueError("An explicit local ESM-2 checkpoint is required; no implicit downloads")
            import esm
            # Trusted, explicitly provided pretrained asset. No contact-regression
            # sidecar is needed because we request residue embeddings only.
            data = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
            self.model, alphabet = esm.pretrained.load_model_and_alphabet_core(Path(checkpoint).stem, data, regression_data=None)
            del data
            self.model = self.model.eval().to(device)
            self.model.requires_grad_(False)
            self.converter = alphabet.get_batch_converter()
            self.layer = self.model.num_layers
            self.dim = self.model.embed_dim
            source = {"backend": "esm2", "checkpoint_sha256": sha256(checkpoint), "layer": self.layer, "dim": self.dim}
        self.config = {"schema": 1, **source, "window": window, "overlap": overlap,
                       "conformers": conformers, "seed": seed, "rdkit": rdBase.rdkitVersion,
                       "chemistry": "explicit_smiles_with_atom_residue_or_linear_L_free_termini_v1",
                       "protein_tokens": "ESM preserves U/O/B/Z; J maps to X; nonstandard graph AA maps to unknown",
                       "protein_graph": "bidirectional_sequence_edges_v1"}

    def embedding(self, seq):
        if hasattr(self, "table"):
            return self.table[[AA.index(a) if a in AA else 20 for a in seq]]
        out = np.zeros((len(seq), self.dim), np.float32)
        counts = np.zeros(len(seq), np.float32)
        for start in range(0, len(seq), self.window - self.overlap):
            text = seq[start:start + self.window].replace("J", "X")
            _, _, tokens = self.converter([("sequence", text)])
            with torch.inference_mode():
                h = self.model(tokens.to(self.device), repr_layers=[self.layer], return_contacts=False)["representations"][self.layer]
            out[start:start + len(text)] += h[0, 1:len(text) + 1].float().cpu().numpy()
            counts[start:start + len(text)] += 1
            if start + len(text) == len(seq):
                break
        return out / counts[:, None]

    def build(self, entity, side):
        seq = sequence(entity["sequence"], peptide=side == "peptide")
        if seq != entity["sequence"]:
            raise ValueError("Normalize entity sequences before preparing features")
        n = len(seq)
        a, b = np.arange(n - 1), np.arange(1, n)
        arrays = {"esm": self.embedding(seq).astype(np.float32),
                  "aa": np.array([AA.index(c) if c in AA else 20 for c in seq], np.int64),
                  "edges": np.array([np.r_[a, b], np.r_[b, a]], np.int64)}
        if side == "protein":
            return arrays
        if side != "peptide":
            raise ValueError("Unknown entity side")
        mol = Chem.MolFromSmiles(entity["smiles"]) if entity.get("smiles") else Chem.MolFromFASTA(seq)
        if mol is None:
            raise ValueError(f"Cannot construct chemical identity for {entity['id']}")
        if entity.get("smiles"):
            mapping = np.asarray(entity.get("atom_residue", []), np.int64)
            if len(mapping) != mol.GetNumAtoms() or (mapping < 0).any() or (mapping >= n).any():
                raise ValueError("Explicit SMILES requires a valid zero-based residue index for every atom")
        else:
            mapping = np.array([a.GetPDBResidueInfo().GetResidueNumber() - 1 for a in mol.GetAtoms()], np.int64)
        af = []
        for atom in mol.GetAtoms():
            af.append([atom.GetAtomicNum() / 100, atom.GetTotalDegree() / 4,
                       atom.GetFormalCharge() / 4, atom.GetTotalNumHs() / 4,
                       float(atom.GetIsAromatic()), float(atom.IsInRing()), atom.GetMass() / 100,
                       int(atom.GetHybridization()) / 8,
                       float(atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW),
                       float(atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW),
                       float(atom.GetAtomicNum() in (7, 8, 16)), float(atom.GetAtomicNum() == 6)])
        bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
        edges = bonds + [(j, i) for i, j in bonds]
        bond_weights = [b.GetBondTypeAsDouble() for b in mol.GetBonds()] * 2
        arrays.update(atom_features=np.asarray(af, np.float32), atom_residue=mapping,
                      atom_edges=np.asarray(edges, np.int64).T.reshape(2, -1),
                      atom_edge_weight=np.asarray(bond_weights, np.float32))
        if set(mapping.tolist()) != set(range(n)):
            raise ValueError("Every sequence residue must map to at least one chemical atom")
        coords = []
        if self.conformers:
            hydrogenated = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed, params.numThreads = self.seed, self.threads
            params.pruneRmsThresh, params.enforceChirality = 0.5, True
            ids = AllChem.EmbedMultipleConfs(hydrogenated, numConfs=self.conformers, params=params)
            for cid in ids:
                if AllChem.UFFHasAllMoleculeParams(hydrogenated):
                    status = AllChem.UFFOptimizeMolecule(hydrogenated, confId=cid, maxIters=500)
                    if status != 0:
                        continue
                xyz = np.asarray(hydrogenated.GetConformer(cid).GetPositions()[:len(mapping)], np.float32)
                if not np.isfinite(xyz).all():
                    continue
                # Heavy-atom centroids per residue; distance features are frame invariant.
                residue_xyz = np.stack([xyz[mapping == i].mean(0) for i in range(n)])
                if np.isfinite(residue_xyz).all():
                    coords.append(residue_xyz)
        if coords:
            arrays["conformer_coords"] = np.stack(coords)
            arrays["geometry_mask"] = np.ones((len(coords), n), bool)
        else:
            arrays["conformer_coords"] = np.zeros((1, n, 3), np.float32)
            arrays["geometry_mask"] = np.zeros((1, n), bool)
        arrays["conformer_mask"] = np.ones(len(arrays["conformer_coords"]), bool)
        return arrays


def prepare_entities(entities, side, store, builder):
    if builder.config != store.config:
        raise ValueError("Builder/store recipe mismatch")
    count = generated = 0
    for e in entities:
        key = store.key(e, side)
        if not store.contains(key):
            store.put(key, builder.build(e, side))
            generated += 1
        count += 1
    return {"entities": count, "new_features": generated, "recipe_sha256": digest(store.config)}
