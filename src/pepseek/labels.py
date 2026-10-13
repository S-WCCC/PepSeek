"""Convert externally mapped complex heavy atoms into masked supervision ONLY."""
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from .io import sha256, write_json


def structure_labels(source, output, protein_length, peptide_length, threshold=8., distance_edges=(4., 8., 12., 16.)):
    if Path(output).exists():
        raise ValueError("Label output already exists; never overwrite sealed labels")
    with np.load(source, allow_pickle=False) as z:
        pa, qa = np.asarray(z["protein_heavy_atom_coords"], float), np.asarray(z["peptide_heavy_atom_coords"], float)
        pi, qi = np.asarray(z["protein_atom_residue"], int), np.asarray(z["peptide_atom_residue"], int)
    for xyz, idx, length in ((pa, pi, protein_length), (qa, qi, peptide_length)):
        if xyz.shape != (len(idx), 3) or not np.isfinite(xyz).all() or not len(idx) or idx.min() < 0 or idx.max() >= length:
            raise ValueError("Invalid mapped heavy-atom coordinates; mappings must use full sequence indices")
    distance = np.full((protein_length, peptide_length), np.inf)
    for start in range(0, len(pa), 256):
        d = cdist(pa[start:start + 256], qa)
        np.minimum.at(distance, (pi[start:start + 256, None], qi[None]), d)
    valid = np.isfinite(distance)
    contact = np.where(valid, (distance < threshold).astype(float), np.nan)
    pvalid, qvalid = valid.any(1), valid.any(0)
    psite = np.where(pvalid, np.nan_to_num(contact).max(1), np.nan)
    qsite = np.where(qvalid, np.nan_to_num(contact).max(0), np.nan)
    bins = np.where(valid, np.digitize(distance, distance_edges), -1).astype(np.int64)
    positive_residues = np.where(psite == 1)[0]
    regions = []
    if len(positive_residues):
        centers = np.stack([pa[pi == i].mean(0) for i in positive_residues])
        _, components = connected_components(cdist(centers, centers) < 10., directed=False)
        for component in np.unique(components):
            membership = np.where(pvalid, 0., np.nan)
            membership[positive_residues[components == component]] = 1.
            regions.append(membership)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, contact=contact.astype(np.float32), protein_interface=psite.astype(np.float32),
                        peptide_interface=qsite.astype(np.float32), distance_bins=bins,
                        regions=np.asarray(regions, np.float32).reshape(-1, protein_length))
    record = {"source_sha256": sha256(source), "output_sha256": sha256(output),
              "contact_definition": "minimum heavy-atom distance strictly below threshold", "threshold_angstrom": threshold,
              "distance_bin_edges_angstrom": list(distance_edges), "region_definition": "connected observed-contact residues, centroid distance <10A",
              "scope": "regions observed in this complex, not all possible binding pockets", "coordinates_are_model_inputs": False}
    write_json(str(output) + ".receipt.json", record)
    return record
