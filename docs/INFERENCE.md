# Sequence inference and proteome screening

Input FASTA must contain unique IDs; JSONL entities can additionally encode explicit peptide chemistry. The default sequence-only chemical interpretation is documented in the data contract. Queries and candidate libraries use the exact feature recipe from training.

## Detailed pair prediction

Create identity-only `query_pairs.jsonl`:

```json
{"id":"query_001","protein_id":"protein_001","peptide_id":"peptide_001"}
```

```bash
pepseek predict --proteins data/query_proteins.fasta --peptides data/query_peptides.fasta \
  --pairs data/query_pairs.jsonl --features runtime/features/v1.sqlite \
  --checkpoint runtime/runs/experiment_001/best.pt \
  --output runtime/predictions/query_001 --device cuda:0
```

Prepare missing raw features first. Output includes scalar interaction score, endpoint predictions and NPZ arrays containing contact/interface/region probabilities and conformer weights. `--score-only` omits detailed outputs while preserving score computation. Quantitative endpoints with no recorded training labels are blank in the prediction CSV. The receipt records head-level supervision coverage; a head with zero loss weight or no effective learning remains unvalidated even if some labels exist.

## Index properties

- Candidate encodings retain residue information, graph/chemical features after encoding, and conformer-specific states.
- SQLite stores compressed arrays and entity IDs on disk. Candidate data are batched into GPU/CPU memory.
- Checkpoint and preprocessing fingerprints must match. Partial indices cannot be queried. Changed checkpoint means a new encoded index.
- Queries use exact candidate scans. There is no ANN pruning or discarded low-scoring first-stage set.
- Ties use descending candidate ID after descending score, consistently recorded in receipts.
- `--all-scores` streams all scores per query to CSV. Otherwise only top K and timing are saved.
- SHA validation of a very large index has its own startup cost, separate from query timing. Cold start, preparation, indexing and scoring must all be included in end-user performance estimates.

## Human proteome

Use a versioned canonical reference FASTA and save source accession/release/URL/date/SHA alongside it. The software does not download an evolving proteome implicitly. Protein length is preserved; ESM preprocessing uses overlapping windows. Very long proteins require small scoring batches because residue-level interaction remains length-dependent.

## Retrieval evaluation

Provide `truth.json` mapping query ID to its known positive candidate IDs, and ensure these candidates actually occur in the sealed candidate library.

```bash
pepseek evaluate-retrieval --rankings runtime/searches/query_001/rankings.csv \
  --truth data/benchmark/truth.json --output runtime/searches/query_001/metrics.json
```

Metrics are query-macro Recall@K and reciprocal rank truncated to returned K. K exceeding saved ranking depth is null. Benchmark reports must include candidate count, positive coverage, query coverage, score failures and training-overlap audit. The command computes ranking metrics only; it cannot determine assay truth or unseen homology leakage.
