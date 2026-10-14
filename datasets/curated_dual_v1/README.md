# Curated protein–peptide data: two supervision tables

This dataset combines curated PPIKB measurements with mapped PPIKB, Propedia and
PepBDB structures. A pair may carry either or both supervision types. Splitting
one linked record into two tables does not create another independent pair.

## Download

The sealed data assets are published with release
[`curated-dual-v1-20261013`](https://github.com/S-WCCC/PepSeek/releases/tag/curated-dual-v1-20261013).
The release's `DOWNLOAD_MANIFEST.json` records every archive part and SHA256.

```bash
python scripts/download_curated_data.py --output data/curated_dual_v1 --structures
```

`clean_training` contains model-ready JSONL, every required label array, residue
maps, assignments and provenance. `structure_libraries` contains separately
indexed protein and peptide atom-coordinate views. Both archives use relative
paths. Source raw files and their original exclusions are preserved on the
development data disk; their hashes remain in the release manifest.

## Scale

| Unit | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Input pairs | 71,081 | 2,730 | 2,712 | 76,523 |
| Quantitative observations | 8,393 | 195 | 189 | 8,777 |
| Structural observations | 65,913 | 2,646 | 2,606 | 71,165 |
| Distinct supervised structural views | 65,205 | 2,605 | 2,567 | 70,377 |

Clean libraries contain **33,652 protein sequences**, **29,856 peptide
sequences**, **62,259 protein structure views** and **47,870 peptide structure
views**. Sequence strings, chemical entities, structure views and observations
are different counting units. A structure view can be an accurately mapped
fragment; it is not necessarily a full-length protein or a distinct PDB entry.

Protein homology, peptide similarity, chemical identity, PDB entries,
publications and verified links are grouped before partitioning. The largest
connected component stays in training; actual proportions differ from nominal
80/10/10. Quantitative test coverage is 120 pKd, 59 assay-dependent pIC50,
9 pKi, 1 pEC50 and 0 pKm. The latter endpoints do not support a complete
held-out quantitative benchmark. Structural evaluation has broader coverage.

This is a new fixed split for retraining/internal evaluation; older project
models have already seen parts of the source corpus. Historical split locks
remain unchanged. No scientific training is part of this data release.

## Documentation and evidence

- [Methods, schema, masks and sampling](../../docs/CURATED_DUAL_DATA.md)
- [Processing steps and source attribution](../../docs/DATA_CURATION.md)
- `DATA_CARD.json`, `audit/ACCEPTANCE.json`, `RELEASE_SEAL.json` in the data assets
- `audit/clean_dispositions.jsonl`: every eligible candidate's use/hold decision

The historical `catalog-source-v1.tar.gz` is a preparation snapshot containing
8,777 measurement and 76,442 structural candidates. It is retained for provenance
and is **not** the final clean split. Its SHA256 is
`aa4e9d4b9d80eaf60ae836c08f92e3d8a93b4c3f2877cdbdeb83b0de1a758237`.

Source attribution and original dataset terms remain applicable. The code
license does not replace those data terms.
