# Unified quantitative and structural supervision

The `curated_dual_v1_20261013` release is built from the sealed PPIKB disposition
snapshot and the previously curated Propedia/PepBDB structural corpus. It has two
observation tables, a shared entity/pair registry, and explicit evidence links.
The preparation code is in `scripts/curation/`. Completion is established by the
data release seal and acceptance receipts, rather than by the presence of code.

## Units and identities

- A quantitative observation retains endpoint, molar bounds, inequality direction,
  assay description, source publication and original measurement ID. Repeated
  measurements remain observations of a pair, not new independent molecules.
- A structural observation retains its deposited complex, selected chains,
  assembly operation, residue mapping and masked contact/interface labels.
- A sequence library deduplicates exact sequence text. A chemical-entity library
  additionally distinguishes modifications, termini and covalent topology.
  Unresolved structural chemistry is explicitly masked; it is not reconstructed
  as an unmodified, free-terminal peptide.
- Verified joint links connect the tables. Exact compatible input identities can
  share one model row; partial-construct links retain explicit mappings and are
  placed in the same split component. Similar sequences alone do not establish
  identity of experimental samples.
- Multi-chain proteins retain separate component sequences, ESM calls, sequence
  edges and positional features. Tensor concatenation introduces no peptide bond
  between chains.

## Processing sequence

1. `build_catalog.py`: preserve the complete source ledger, split joint records
   into two supervision tables, create content-addressed identities and FASTA.
2. `map_chemistry.py`: reconstruct only the audited annotation grammar, establish
   full chiral graph isomorphism, and map supplied SMILES atoms to residues.
3. `stage_assets.py` / `acquire_assets.py`: synchronize exact coordinate snapshots;
   accept a file only when its SHA256 matches the sealed source manifest.
4. `stream_structure_labels.py`: select mapped observed heavy atoms in the same
   experimental frame; regenerate contact/interface/distance labels and per-chain
   structural libraries with residue maps and coordinate provenance.
5. `homology.py` (protein stage) and `peptide_homology_parallel.py`: actually run
   MMseqs2 and BLASTP-short searches; retain reproducible commands, alignment edges
   and receipts. Query sharding changes execution scheduling, not thresholds.
6. `assign_splits.py`: union protein homology, peptide similarity, chemical identity,
   pair identity, PDB entry, publication and verified joint-link groups. Assign
   connected components intact toward 80/10/10 using a fixed seed of 17.
7. `build_clean.py`: retain every observation disposition, assemble model-compatible
   rows, export all structure views, and report actual train/valid/test coverage.
8. Run schema, file-SHA, mapping, cross-partition and model-input acceptance before
   sealing a release. Inspect component sizes; a large biological component is
   never cut merely to achieve nominal percentages.

## Label policy

Contact means a minimum mapped heavy-atom distance **<= 8 Angstrom**. An observed
close atom pair is positive. A noncontact is supervised only when both residues
have the required standard heavy atoms; unresolved positions are NaN. Interface
negatives require complete partner coverage. Distance-bin supervision requires
complete atoms on both residues, with boundaries 4.5, 8, 12 and 16 Angstrom.
Experimental complex coordinates are used solely for supervision and provenance.

Kd, Ki, assay-dependent IC50, EC50 and Km have separate output heads. Km is an
enzyme-kinetic endpoint, not an interchangeable affinity label. No experimental
nonbinder labels are invented. Co-complex contrastive positives use a separate
field from measured binary binding labels. Unknown cross-pairs are distractors.

## Sampling and evaluation

The model manifest contains one row per input pair, a list of quantitative
observations, and a list of structural views. Training samples each pair once
per epoch, averages its quantitative observation losses, and rotates structural
views deterministically. Validation/test evaluate all views; pair scores and
quantitative predictions are recorded once, while structural metrics also retain
their per-view records and pair-macro aggregates.

Protein grouping uses >=30% alignment identity with >=80% coverage of the shorter
component to protect mapped fragments. Peptide grouping uses >=50% identity and
>=80% coverage of **both** sequences. Search-edge audits define the operational
homology criterion. They do not establish absence of every distant evolutionary
relationship. Historical split files stay intact. This new release is for new
training runs; previous project models have seen parts of the source corpus.

## Source attribution and redistribution

- PPIKB: https://ppikb.duanlab.ac/ (CC BY 4.0 as stated by the provider).
- Propedia26 supplementary data: https://github.com/LBS-UFMG/propedia26-sm
  (CC BY 4.0 in that source repository).
- PepBDB: http://huanglab.phys.hust.edu.cn/pepbdb/; preserve provider citation and
  source identifiers. The original archive is retained on the data disk, rather
  than assigned a new software license by this project.
- PDB coordinates: retain accession, source URL, revision-specific file SHA and
  selected chain/assembly identity. Citation records remain linked to each complex.

Source archives, literature evidence, local transfer manifests and credentials are
not interchangeable with the public clean data package. Public packages contain
relative paths, curation-derived tables/labels and source attribution; private
machine paths and authentication material are excluded.
