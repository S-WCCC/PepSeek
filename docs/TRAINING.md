# Training and evaluation

`configs/model_m.json` describes architecture; `configs/train.json` describes optimizer, losses and limits. Save a new configuration file for each scientific experiment. The defaults are engineering starting settings, not a claim of tuned superiority.

## Objective and batching

Masked binary pair loss + bidirectional multi-positive contrastive loss + interval Huber + reliable group ranking + contact BCE + interface BCE + matched-region BCE + distance categorical loss. Terms average over available observations within a batch and have explicit weights. Crossed unknown pairs contribute contrastive competition but are not labeled experimental non-binders.

Known train positives protect all matching pairs within a batch. The crossed-pair matrix uses the same interaction model; it is more expensive than a vector dot product. Batch size defaults to 4, pair chunk to 4; select larger values only after measuring memory on real chain lengths and conformer counts. No silent sequence truncation or topology-changing crop is performed.

Frozen ESM is a preprocessing operation, outside the optimizer. Gradients update chemical/sequence adapters, graph layers, pool–interact–unpool blocks and supervised heads. BF16 is enabled only for CUDA devices supporting it. Gradient clipping rejects nonfinite gradients.

## Selection

- Epoch zero is evaluated and can be retained.
- Checkpoint selection uses the minimum total validation loss with fixed coefficients and deterministic valid batches.
- Earlier checkpoints win ties.
- ReduceLROnPlateau observes validation loss; patience-based stopping and max steps apply.
- No test scoring occurs during training. Final `best.pt` is bound to `SELECTION_SEAL.json`.
- Validation composition, batch size and loss weights affect the selection statistic; freeze them in the experiment plan.

For a retrieval-specific experiment, use a separately predeclared query/candidate validation protocol before changing selection policy. This initial implementation does not silently switch selection metrics based on outcomes.

## Multiple GPUs / CPU processes

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone --nproc_per_node=2 \
  -m pepseek train --dataset data/v1/dataset.json \
  --features runtime/features/v1.sqlite --model-config configs/model_m.json \
  --train-config configs/train.json --output runtime/runs/multigpu_001 --device cuda
```

The engine explicitly averages gradients across processes and handles missing-head gradients. Each rank uses a local contrastive batch; this version does **not** gather global cross-rank negatives. The effective optimization batch is local batch × world size; that is not the contrastive candidate count. Epoch batches are padded with reproducible repeats so every rank takes the same number of steps. Rank zero performs validation and writes artifacts. `torchrun --... --device cpu` exercises the same process synchronization with Gloo.

## Resume and interruptions

`last.pt` stores model, optimizer, scheduler, epoch, step and immutable run identity. `--resume` requires exact data, feature recipe, model/train configuration and world size. Epoch RNG is derived from seed, epoch and rank. Resume is at the next epoch boundary; an interrupted partial epoch is not replayed. Single-process epoch-boundary reproducibility is tested separately from scientific multi-GPU reproducibility.

SIGTERM/SIGINT request graceful stopping after an active batch. Validation/checkpoint collection still runs; `max_seconds` is a soft training-loop budget, not an external hard GPU deadline. A deployment requiring hard deadlines must additionally use a process-identity-verified scheduler/watchdog. Finalized runs cannot be resumed into the same directory.

## Artifacts

`run.json`, `DATA_SEAL.json`, `history.csv/json`, `last.pt`, `best.pt`, `SELECTION_SEAL.json` (or `INTERRUPTED.json`). Training-label coverage per endpoint/head is recorded in the run identity. Evaluation writes pair/quantitative prediction CSV, per-complex structural predictions, macro structural metrics and a file-hash receipt.

Binary AUROC/AP need both classes. Constant quantitative predictions have undefined Spearman; JSON uses null. RMSE/MAE/Spearman are computed on exact quantitative labels; censored labels train through intervals and are not converted into artificial exact test values. Structural metrics are macro-averaged across complexes with defined metrics, with denominators reported.

Region evaluation uses permutation-matched Dice/IoU for observed region masks at threshold 0.5. Unobserved extra slots are unscored, so this measures recovery of observed regions rather than exhaustive detection of every physical pocket.
