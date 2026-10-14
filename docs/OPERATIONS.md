# Development 1 deployment and maintenance

## Layout and isolation

The curated-data update uses a separate versioned source directory under
`runtime/curation/curated_dual_v1_20261013/`; it does not replace the active
conformer-benchmark checkout at `/data/PepSeek`. Its dataset is
`/data/PepSeek/data/curated_dual_v1_20261013`. The final deployment receipt in that
runtime directory records the release source, environment and dataset seal.
Use the supplied `activate_curated.sh` to select that source explicitly.

Repository `/data/PepSeek`, branch `codex/unified-backbone-deployment`; origin is the user-supplied PepSeek repository. Initial upstream revision: `6602bc2938b990e0b43fa850e43191184a653708`.

`.venv` is created with the system Python 3.10 interpreter and **without system site packages**. GPU PyTorch dependencies are installed inside this environment. Nothing depends on an older Model-M, PSICHIC or comparator environment. Original projects and workloads are preserved.

`requirements/dev1.lock` records the resolved Linux/Python 3.10/CUDA 12.1 package versions from this deployment. The project itself is installed from the checked-out source commit. `requirements/dev1.txt` records the smaller direct dependency list used to bootstrap the initial environment.

Source and small documentation are tracked by Git. `data/`, `runtime/` and `.venv/` are excluded. Runtime contents:

```text
runtime/deployment/  installation log, lock, exit status, environment and source receipts
runtime/assets/      explicit pretrained assets with origin/SHA manifests
runtime/cache/       pip, torch and other caches
runtime/features/    immutable raw feature SQLite stores
runtime/indices/     checkpoint-bound encoded libraries
runtime/runs/        unique training output directories
runtime/evaluations/ selected-checkpoint evaluations
runtime/searches/    sequence screening results
runtime/acceptance/  synthetic software validation only
runtime/tmp/         temporary files on the data disk
```

## Health and installation

```bash
cd /data/PepSeek
cat .venv/pyvenv.cfg
.venv/bin/python -m pip check
.venv/bin/pepseek doctor
df -h /data
tail -30 runtime/deployment/bootstrap.stdout.log
cat runtime/deployment/bootstrap.exit.json
```

Installation uses `flock` and can run detached with logs. On SSH loss, inspect the active process and exit receipt before invoking it again. Packages already downloaded remain in this project's cache. Do not delete the environment or duplicate an active installer.

## Updating source

1. Inspect branch/status, active project processes, available disk and environment receipt.
2. Create a scoped Git commit with code, tests and change documentation.
3. Save package/source hashes; transfer to the same branch or a versioned checkout.
4. Check the transferred archive hash before unpacking. Inspect file paths; never unpack unexpected absolute/traversal paths.
5. Run relevant tests in `.venv`; architecture/data changes also require a fresh end-to-end acceptance.
6. Save results in `runtime/deployment/<update>/`, append `docs/CHANGELOG.md` and `docs/updates.jsonl`, and record the commit.
7. If tests fail, preserve logs and fix/revert the source through Git. Do not overwrite scientific runs or silently replace old data.

## Launching real jobs

Deployment acceptance does not schedule a full scientific experiment. Before real training, create an explicit run plan naming dataset seal, model config, validation policy, visible GPU UUIDs and wall-time budget. Verify live free resources. Use a persistent job manager/tmux, save PID/start time/command identity and log paths. Stop only a verified project-owned process.

Never infer that older permission to occupy a whole cluster still applies to a new scientific run. Current deployment uses bounded CPU checks by default; it does not stop any other project.

## Recovery

- Interrupted training: check `INTERRUPTED.json`, process identity and `last.pt`; run explicit `--resume` with identical inputs.
- Finalized training: a new directory is required; sealed artifacts are immutable.
- Interrupted feature preparation: restart the same recipe; existing content-addressed features are verified/reused.
- Interrupted index build: incomplete index cannot be searched; preserve it and rebuild to a new path.
- Evaluation/search failure: preserve output, then use a new output directory for the repaired run.

## Completion evidence

The final deployment receipt under `runtime/deployment/` identifies the tested source commit, environment lock, test counts and acceptance outputs. Its results establish software operability; trained model quality and million-scale throughput need their own scientific receipts.
