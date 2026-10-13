# PepSeek execution and maintenance rules

## Authority and scope

- This is the new universal protein–peptide project, deployed at `/data/PepSeek` on development 1. Historical PSICHIC/S4/Model-M datasets, split locks, controllers and runs are external, read-only provenance. Do not restart or modify them.
- Use the user's pinned development-1 SSH alias and verify its host fingerprint through the installed A100 skill. Never write keys, passwords, tokens or SSH configuration into this repository.
- The current task authorizes isolated deployment and bounded software acceptance, not a new full scientific training run. Do not stop other workloads. Never pause training 1. Check resource ownership and availability before any GPU execution.
- Do not infer permission to publish private data, upload checkpoints, or launch an unbounded job from access to a repository.

## Environment and storage

- Execute using `.venv/bin/python` and `.venv/bin/pepseek`; `include-system-site-packages` must be false. No global pip installs, no implicit dependency on another project's environment.
- Keep code in Git. Keep datasets in `data/`, assets/caches/runs/logs under `runtime/`; both are Git-ignored. Large files belong on the data disk.
- Never overwrite a sealed scientific run. New configuration/data/model version means a new directory and new SHA records.
- Check disk availability before large feature/index generation. Full preinteraction residue features are larger than one vector per molecule; estimate bytes from a pilot.

## Model and scientific contract

- External prediction inputs are sequences plus optional explicit peptide chemical identity. All scoring goes through the shared pool–interact–unpool backbone.
- Experimental complex coordinates are label-generation inputs only; never include them or native cross-chain distances in forward inputs. Contact arrays, interface masks and region targets stay separate from features.
- Missing labels are masks, not negative values. Unknown whole-pair interactions are not experimental negatives. Multiple known positives must be protected in the contrastive mask.
- Preserve endpoint units and experimental comparability. Do not mix pKd/pKi/functional pIC50/pEC50 as one interchangeable target. Only explicitly reliable within-experiment ranking groups are ordered.
- Do not change sealed splits or select on test. `cluster_disjoint` validates supplied cluster assignments; it does not prove that clustering thresholds were biologically adequate.
- Generated conformers are predictions, not experimental replicates or equilibrium populations. Missing conformers use explicit masks. Only intrapeptide geometry enters the encoder.
- Predicted region slots are not automatically physical pockets. Report native 3D pocket geometry only with an explicit structural analysis.
- Never present toy-fixture metrics as model performance. Untrained or unsupervised heads must not be presented as validated outputs.

## Before and after each update

1. Read `docs/CHANGELOG.md`, `docs/OPERATIONS.md`, Git status, and the last deployment receipt.
2. Inspect active processes before resuming anything; match command, start time, parent and job identity. PID alone is insufficient. No blind duplicate launches.
3. Preserve uncommitted user work. Make a scoped branch. Do not force-push or reset the user's branch.
4. Record the change, affected files, reason, validation and scientific impact in `docs/CHANGELOG.md`. Add a machine-readable entry to `docs/updates.jsonl`.
5. Run relevant tests in the isolated environment. For architecture/data/engine changes, also run the synthetic end-to-end acceptance into a fresh directory. Never reuse output directories to conceal a failed run.
6. Record Git commit, dependency lock, source SHA, data SHA, checkpoint SHA and test receipts. Commit source and documentation together; keep generated artifacts under `runtime/`.
7. Report what actually ran, what passed, and remaining untested scientific coverage. No SOTA, speed or completeness claim without corresponding evidence.

## Execution defaults

- No hidden downloads: ESM weights must be supplied as an explicit local asset with a recorded SHA and origin.
- CPU smoke tests are the default. Multi-GPU launch must use explicit visible devices and a documented runtime budget.
- `scripts/bootstrap_dev1.sh` uses a lock; inspect logs and exit receipt before retrying installation.
- Stop only a confirmed project-owned process. Preserve partial logs and checkpoints. Do not automatically resume unrelated paused projects.
