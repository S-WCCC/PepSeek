# PepSeek

Sequence-deployable protein–peptide screening with a unified **pool → interact → unpool** backbone and masked multi-task supervision.

The scoring, quantitative, contact, interface and region heads share candidate-conditioned interaction features. Screening runs the same score computation as detailed analysis. This repository is an engineering implementation; scientific performance must be established with a versioned dataset and a trained, selected checkpoint.

## Project layout

```text
src/pepseek/         model, features, data contract, losses, training, evaluation, screening
configs/            full-size model and explicit software-smoke configurations
tests/              gradients, masking, geometry invariance, caching and split checks
scripts/            isolated installation and end-to-end acceptance
docs/               architecture, data contract, operations and change history
requirements/       direct deployment pins; resolved environment recorded at deployment
AGENTS.md           instructions for future human/AI maintenance
.venv/              isolated Python environment (untracked)
data/               versioned scientific inputs and labels (untracked)
runtime/            assets, caches, logs, indices and immutable runs (untracked)
```

## Installation and first acceptance

On the designated development server, the checkout is `/data/PepSeek`.

```bash
cd /data/PepSeek
bash scripts/bootstrap_dev1.sh
source scripts/activate.sh
pepseek doctor
python -m pytest -q
python scripts/acceptance.py --output runtime/acceptance/run_001
```

The acceptance creates **synthetic labels and toy embeddings**, executes training/validation selection, seals a checkpoint, evaluates the held-out fixture and screens in both directions. It is a software check, not a scientific result.

For a separate CPU workstation:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[esm,test]'
```

## Train with your data

The curated PPIKB + Propedia + PepBDB release provides two supervision tables,
mapped structure labels and a fixed component-disjoint split. See
[the dataset entry](datasets/curated_dual_v1/README.md) and
[the methods and schema](docs/CURATED_DUAL_DATA.md).

```bash
python scripts/download_curated_data.py --output data/curated_dual_v1 --structures
```

`--structures` adds the separate mapped protein/peptide atom-coordinate libraries.
The default download already includes all labels required by the model. Native
complex coordinates are supervision assets, not sequence-prediction features.

Prepare a `dataset.json`, entity JSONL files and pair JSONL file using [the data contract](docs/DATA_CONTRACT.md). The caller supplies fixed partitions. `audit` checks identities and declared clusters; it does not perform homology clustering.

```bash
pepseek audit --dataset data/v1/dataset.json --output data/v1/SEAL.json
pepseek prepare --dataset data/v1/dataset.json \
  --features runtime/features/v1.sqlite \
  --esm-checkpoint runtime/assets/esm2_t33_650M_UR50D.pt \
  --conformers 8 --device cuda:0
pepseek train --dataset data/v1/dataset.json \
  --features runtime/features/v1.sqlite \
  --model-config configs/model_m.json --train-config configs/train.json \
  --output runtime/runs/experiment_001 --device cuda:0
pepseek evaluate --dataset data/v1/dataset.json \
  --features runtime/features/v1.sqlite \
  --checkpoint runtime/runs/experiment_001/best.pt \
  --split test --output runtime/evaluations/experiment_001 --device cuda:0
```

`--conformers 0` explicitly disables generated geometry. An eight-conformer recipe can also contain genuine generation failures, represented by a masked sequence fallback; successful generation does not create experimental labels. Features must have the **same recipe** at training and inference.

Training uses validation loss for checkpoint selection, includes epoch zero, and writes `SELECTION_SEAL.json` before test evaluation. Original test labels never enter training. An interrupted run uses `--resume`; a finalized run requires a new directory. See [training details](docs/TRAINING.md).

## Screen a peptide library or the human proteome

Use your fixed human canonical FASTA as the protein candidate list. The software derives its count from the file and preserves every full sequence; it never fabricates a 20,652-entry dataset. Build raw features and a checkpoint-bound encoded index once.

```bash
# Human proteins: prepare and index once. Use the training feature recipe.
pepseek prepare --entities data/human_canonical.fasta --side protein \
  --features runtime/features/v1.sqlite --esm-checkpoint runtime/assets/esm2_t33_650M_UR50D.pt --conformers 8 --device cuda:0
pepseek build-index --entities data/human_canonical.fasta --side protein \
  --features runtime/features/v1.sqlite --checkpoint runtime/runs/experiment_001/best.pt \
  --output runtime/indices/human_v1.sqlite --device cuda:0

# A query peptide: prepare, then rank the full protein index.
pepseek prepare --entities data/query_peptide.fasta --side peptide \
  --features runtime/features/v1.sqlite --esm-checkpoint runtime/assets/esm2_t33_650M_UR50D.pt --conformers 8 --device cuda:0
pepseek search --queries data/query_peptide.fasta --query-side peptide \
  --features runtime/features/v1.sqlite --checkpoint runtime/runs/experiment_001/best.pt \
  --index runtime/indices/human_v1.sqlite --output runtime/searches/query_001 \
  --top-k 100 --all-scores --device cuda:0
```

For protein → peptide screening, index the peptide library with `--side peptide` and search with `--query-side protein`. SQLite-backed indices stream candidate batches; all candidates receive the unified interaction score. Million-candidate latency and index size require measurement on the real length/conformer distribution.

For selected pairs, `pepseek predict` produces scores, endpoint predictions, interface probabilities, contact probabilities and region memberships without labels. See [inference examples](docs/INFERENCE.md).

## Documentation

- [Architecture and implemented scope](docs/ARCHITECTURE.md)
- [Data contract and label masks](docs/DATA_CONTRACT.md)
- [Training, resume and multi-GPU behavior](docs/TRAINING.md)
- [Inference and retrieval evaluation](docs/INFERENCE.md)
- [Deployment and maintenance](docs/OPERATIONS.md)
- [Change history](docs/CHANGELOG.md)
- [Dependencies and attribution](docs/THIRD_PARTY.md)
- [Agent execution rules](AGENTS.md)
