# Implemented architecture: v0.1

## Inputs and reusable encodings

Protein: frozen ESM-2 residue features, amino-acid identity and relative/N/C positional features. Long ESM sequences are covered by overlapping windows, averaged at overlap; no silent truncation. The initial protein graph currently contains bidirectional sequential edges. It does not yet add predicted long-range contact edges. ESM features and subsequent global aggregation convey nonlocal information.

Peptide: the same sequence feature types plus a chemical atom graph. Standard input is a linear L-peptide with free termini; explicit SMILES with atom-to-residue mapping supports different chemistry. Atom features include element, degree, charge, H count, aromaticity, ring membership, mass, hybridization, chirality and heteroatom indicators. Bond orders weight graph messages. Atom representations are pooled into their residues.

Optional ETKDGv3 conformers are generated independently of the protein. Chirality is enforced; embedding RMS pruning is used; UFF relaxation must converge when parameters exist. Residue centroids supply intrapeptide distance features. Failed generation returns a masked, honest sequence fallback. This is an engineering conformer generator, not an exhaustive sampling of peptide free-energy landscapes; molecular validation and coverage must be audited for the intended chemistry. Independently rotating/translating a conformer leaves distance features unchanged.

## Shared candidate-conditioned backbone

Default hidden width 384; two per-chain sparse graph layers; six interaction blocks. Every interaction block:

1. Softly assigns protein residues to 16 aggregation slots and peptide residues to 8 slots.
2. Computes normalized pooled features.
3. Forms a 128-dimensional state for each slot pair using concatenation, product and absolute difference.
4. Applies bidirectional attention over these states.
5. Projects messages back to each chain and unpools them to residues with residual updates.

All conformers traverse this same backbone. An input-conditioned gate mixes their predictions; native poses do not select conformers. Final pair states plus updated residue pools generate the scoring and quantitative features. There is no independent-vector bypass in scoring.

## Heads

- Pair score: uncalibrated scalar for retrieval and supervised binary loss where labels exist.
- Quantitative: separate output dimensions for pKd, pKi, binding pIC50, functional pIC50 and pEC50. Masked interval Huber handles exact/censored measurements.
- Contact: rank-64 factorized residue-pair logits, computed in chunks.
- Interface: per-protein/per-peptide residue logits from interaction-updated states.
- Regions: multi-label protein residue memberships across region slots, trained with permutation-invariant matching of observed regions. Unmatched slots are unlabeled; they need further multi-complex evidence before interpretation. Interaction attention slots use a separate index and are not identified as region decoder slots.
- Distance: five categories, with the explicit boundaries recorded during label generation.

The gate mixes per-conformer probabilities for binary structural heads and values for score/quantitative heads. No claim of a thermodynamic free energy or Boltzmann population is made.

## Fast and detailed modes

Both modes execute the identical interaction and score computation. Fast mode omits dense structural decoding/output. Preinteraction features can be cached only after locking encoder weights; indices are bound to the entire checkpoint SHA. A new checkpoint requires rebuilding encoded indices. Raw frozen ESM/chemical features remain reusable when their recipe is unchanged.

Cross-pair interaction still costs compute. This implementation streams exact scores over every candidate; it does not claim a sublinear ANN search. Costs include ESM/conformer preprocessing, index construction, storage, cold startup verification and scoring. Benchmark these separately.

## What is deliberately separate

3D pocket meshes, docked poses and MD trajectories are not outputs of this implementation. Region/interface probabilities can be mapped to an independently supplied/predicted monomer for visualization. Homology clustering and assay curation are external data preparation steps with recorded provenance.
