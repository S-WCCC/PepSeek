# Curated protein–peptide data: two supervision tables

## Scope and counting units

This release combines the eligible, individually reviewed PPIKB observations
with the previously prepared Propedia and PepBDB structural corpus. It replaces
the three-way data organization with quantitative and structural observation
tables. A linked observation is not an additional independent interaction.

An **observation** is one retained measurement or one deposited chain-pair view.
An **input pair** is a protein component set and a peptide chemical-identity
scope. A **sequence** is an exact amino-acid string. A **structure-library entry**
is an exact mapped atom-coordinate view; several views can share a PDB entry or
sequence. Counts in these units must not be interchanged.

Unknown peptide chemistry remains explicit. Equal sequences do not establish
equal termini, stereochemistry, crosslinks or chemical modifications. Structural
observations with unresolved chemistry retain sequence and structural supervision,
with atom/conformer inputs masked. Known chemical identities require a full
stereochemistry-aware graph match and an atom-to-residue map.

## Files

| Location | Contents |
|---|---|
| `raw/` | Source workbooks, original structure-database rows, historical ledgers and source receipts |
| `supervision/quantitative.jsonl` | Candidate measurement observations, original endpoint, units, relation, assay and provenance |
| `supervision/structural.jsonl` | Candidate chain-pair views, coordinate SHA and chain identities |
| `supervision/*_clean.jsonl` | Model-eligible observations with assigned split and clean pair ID |
| `supervision/verified_links.jsonl` | Inherited association evidence and its original evidence tier |
| `libraries/` | Exact sequence libraries, component sets, chemical entities and mapped structure libraries |
| `coordinates/` | Original coordinates addressed by their content SHA |
| `maps/` | Deposited-chain, residue, assembly-operator and input-index correspondence |
| `labels/` | Contact, interface and distance arrays with missing-label masks |
| `splits/` | Actual sequence-search edges, clusters, connected components and assignments |
| `clean/` | PepSeek dataset manifest, protein/peptide entities, pair records and training examples |
| `audit/` | Source coverage, every observation's disposition, execution and validation receipts |

Association evidence remains graded. A database-and-coordinate identity match
and an independently reviewed experimental-construct match are distinct evidence
tiers; the original flags remain in the link table. These associations do not
transfer a quantitative value to an unverified chemical entity or construct.

## Structural supervision

1. Resolve the declared chain namespace against the deposited coordinate file.
2. Map exact sequences/fragments, or sufficiently matching global alignments.
   Exact residues shared by **all** optimal alignments are retained; ambiguous
   positions are masked. More than 128 optimal alignments remain unresolved.
3. An exact observed-ATOM sequence can instead map through the declared chain's
   ordered observed residues. An `X` can represent a nonstandard monomer, but
   cannot match an arbitrary standard-residue mismatch.
4. Honor explicitly recovered biological-assembly operators. Preserve component
   boundaries, altloc choice, insertion codes, monomer names and missing atoms.
5. Require an observed mapped interchain contact within 5 Å and reject severe
   atom overlap below 0.8 Å. Record the reason for every held observation.
6. Positive contact labels use minimum heavy-atom distance ≤8 Å. Negative labels
   require complete standard heavy atoms on both residues. Missing/incomplete
   negatives remain NaN. Distance bins are bounded at 4.5, 8, 12 and 16 Å;
   incompletely observed distances use class −1. Interface labels follow the
   same observed-contact and completeness rules.

Protein truncation alone is not an exclusion criterion. Correctly mapped
fragments and separately represented protein components retain structural labels.
Native complex geometry is kept in supervision and provenance, never in the
model's sequence-based scoring features.

## Split construction

The release runs MMseqs2 protein searches and BLASTP-short peptide searches.
Protein edges require ≥30% identity and ≥80% coverage of the shorter component;
peptide edges require ≥50% identity and ≥80% coverage of **each** sequence.
Connected components combine these edges with exact chemical identities, paired
identities, shared PDB identifiers, publications and inherited verified links.
The entire connected component receives one partition, with seed 17 and nominal
80/10/10 proportions. Components are allocated by observation counts, never by
activity value or model performance. Actual proportions can differ when a large
component cannot be divided. The receipt reports those actual counts.

Historical split files remain immutable provenance. This newly versioned corpus
supports retraining and internal held-out evaluation. Earlier project models
have seen parts of the source corpus; their prior exposure is not erased by a
new assignment. Evaluation of unseen external data requires a separate lock.

## Model contract and sampling

`clean/dataset.json` uses the standard PepSeek schema. Each pair has an `affinity`
list and a `structure_views` list, either of which can be empty. Quantitative
endpoints are separate: pKd, pKi, assay-dependent pIC50, pEC50 and pKm. Km denotes
an enzyme kinetic quantity rather than an interchangeable binding affinity.
Original concentration values, inequalities, approximation flags and assay
metadata remain in the quantitative supervision table.

The training loader visits one input pair per epoch, averages quantitative
observation losses within that pair, and rotates structural views. Validation
and evaluation retain every structural view with pair-balanced weights and
pair-macro summaries. Co-complex evidence is an explicit retrieval-positive
field; no experimental binary-negative label is invented from an unknown pair.

## Execution

The label-generation environment uses `requirements/curation.txt`; install
MMseqs2 and BLAST+ separately and retain their versions. The production search
used MMseqs2 13-45111 and BLAST+ 2.12.0. For a fresh corpus, run
`homology.py --stage protein` followed by `peptide_homology_parallel.py` before
assignment. The distributed release already includes fixed assignments and
search receipts, so downloading it does not require repeating the searches.

Use the versioned code checkout and the release's actual dataset seal. A typical
workflow, after configuring explicit pretrained assets, is:

```bash
pepseek audit --dataset "$DATA/clean/dataset.json" --output "$RUN/data_audit.json"
pepseek prepare --dataset "$DATA/clean/dataset.json" \
  --features "$RUN/features.sqlite" --esm-checkpoint "$ESM" --device cuda
pepseek train --dataset "$DATA/clean/dataset.json" --features "$RUN/features.sqlite" \
  --model-config "$DATA/clean/model_config.json" \
  --train-config "$DATA/clean/training_config_example.json" \
  --output "$RUN/training" --device cuda
pepseek evaluate --dataset "$DATA/clean/dataset.json" --features "$RUN/features.sqlite" \
  --checkpoint "$RUN/training/best.pt" --output "$RUN/test" --split test --device cuda
```

The supplied training configuration is an interface example, not a scientifically
optimized hyperparameter search or an authorization to start a long run. Freeze
the run plan and validation rule before training; evaluate test after selection.

## Source attribution

- PPIKB: https://ppikb.duanlab.ac/ — retain source row, PMID/DOI and measurement endpoint.
- Propedia: https://github.com/LBS-UFMG/propedia26-sm — archive/version SHA and source rows retained; repository attribution and CC BY 4.0 terms remain applicable.
- PepBDB: http://huanglab.phys.hust.edu.cn/pepbdb/ — 2020-03-18 archive SHA and original chain files retained; cite the source and retain its terms.
- wwPDB/RCSB coordinates: retain PDB entry, coordinate SHA and acquisition URL.

Project code licensing does not replace the original source datasets' terms.
