# Dataset contract v1

Every dataset is self-contained under one directory; manifest paths must be relative. No code path imports the historical experimental folders automatically.

`dataset.json`:

```json
{
  "schema_version": 1,
  "proteins": "proteins.jsonl",
  "peptides": "peptides.jsonl",
  "pairs": "pairs.jsonl",
  "split_policy": "cluster_disjoint",
  "cluster_method": "Record the actual algorithm, version, identity/coverage thresholds and audit here"
}
```

`proteins.jsonl`, one object per line:

```json
{"id":"protein_001","sequence":"ACDEFGHIKLMNPQRSTVWY","cluster_id":"protein_cluster_001"}
```

`peptides.jsonl`:

```json
{"id":"peptide_001","sequence":"ACDEFG","cluster_id":"peptide_cluster_001"}
```

Sequences must be uppercase with no whitespace. Proteins support X/U/O/B/Z/J, including selenocysteine in the human proteome. U/O/B/Z are preserved for ESM; J maps to X for ESM tokenization. The 21-state graph embedding maps nonstandard letters to its unknown token while the original sequence/identity is retained. Peptide sequence features currently require the 20 standard letters. For modified chemistry, provide `smiles` and a zero-based `atom_residue` array mapping every parsed SMILES atom to a sequence position. Every residue must have a mapped atom. The chemical identity is part of the cache key. Ambiguous/unknown chemical identity must be curated before feature preparation.

`pairs.jsonl`:

```json
{"id":"measurement_001","protein_id":"protein_001","peptide_id":"peptide_001","split":"train","binding":1,"source":"publication-and-row-id","affinity":[{"endpoint":"pKd","lower":7.2,"upper":7.2,"group_id":"reliable_assay_001","reliable_ranking":true}],"structure_labels":"labels/complex_001.npz"}
```

- `binding`: 1/0 only for curated positive/negative relationships; omit when unknown. Structural positives should explicitly use 1. No automatic random-negative labels are generated.
- `affinity`: optional list. Bounds are already on the specified p-endpoint scale. Exact value: equal bounds; lower/upper censoring: other bound is `null`. Unit conversion and the reversal of inequalities on converting concentration to p-values belong in audited curation.
- Ranking is enabled only for exact labels of the same protein/endpoint/reliable `group_id`, separated by at least 0.5. Rankings are formed within each batch; their availability depends on the sampler.
- `structure_labels`: optional NPZ. Missing modalities are omitted, never represented by false measurements.
- Multiple measurements can reference one entity pair, but that pair must remain in a single partition. Conflicting binary labels require domain-aware curation before this v1 schema is used.

## Structural labels

All indices map to full input sequences, including unresolved residues.

| Array | Shape | Missing value |
|---|---|---|
| `contact` | Lp × Lq | NaN |
| `protein_interface` | Lp | NaN |
| `peptide_interface` | Lq | NaN |
| `distance_bins` | Lp × Lq | -1 |
| `regions` | observed regions × Lp | NaN |

Binary observations are 0/1. NaN means no supervision. A resolved noncontact residue pair is different from an unobserved whole protein–peptide relationship.

`pepseek make-labels` accepts a separate NPZ with `protein_heavy_atom_coords`, `peptide_heavy_atom_coords` and matching zero-based `protein_atom_residue`, `peptide_atom_residue`. Coordinates must already be mapped, heavy-atom-only, in the same complex frame. The command produces minimum-heavy-atom contact labels, distance bins [4,8,12,16] Å, interfaces and connected observed-contact regions. Default contact threshold is strictly <8 Å. The source SHA, definitions and output SHA are saved. PDB/mmCIF chain/sequence alignment is upstream curation, not guessed by this command.

Native coordinates are never stored in feature inputs. Label generation and feature generation are separate commands.

## Splits and seals

- `pair`: checks chemical pair identity across partitions.
- `entity_disjoint`: additionally checks sequence identity per role.
- `cluster_disjoint`: additionally requires and checks supplied cluster IDs per role.

The audit verifies declared partition invariants, not the biological validity of a clustering method. Supply separately audited homology/peptide/source grouping when required. Test partition records are never used by the training loop for loss or checkpoint selection. File hashes can be computed without scoring test labels. `audit` may validate test label formats; it does not compute predictive performance.

## Three data categories

1. Sequence + quantitative + structure: populate both optional fields.
2. Sequence + quantitative: populate affinity, omit structure labels.
3. Sequence + structure: populate structure labels and known binding; omit affinity.

Store source accession, publication, original row, assay and transformation provenance as extra pair metadata. Such metadata does not automatically enter model features.
