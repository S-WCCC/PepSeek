# Change history

Every source/config/schema change must add an entry including rationale, validation and scientific impact. Detailed command logs stay in runtime; this file stays readable and in Git.

## 2026-10-13 — 0.1.1 human-proteome input compatibility

- Added explicit protein U/O/B/Z/J token handling so selenoproteins are retained during canonical human-proteome preprocessing. The original sequence and cache identity remain intact; J-to-X tokenization and graph unknown-token mapping are recorded in the feature recipe.
- Added a regression test. This changes the feature-recipe fingerprint, so v0.1.0 raw caches cannot be silently reused; prepare a fresh cache with v0.1.1.
- Peptide chemistry continues to require a supported sequence plus explicit mapping for modified chemical identity. No experimental data or partitions were modified.

## 2026-10-13 — 0.1.0 initial isolated deployment

- Started from the user's repository initial commit; created a dedicated development branch and a fresh Python 3.10 virtual environment on the development-1 data disk.
- Imported the previously written, untested unified model draft and completed its package, data/features, masked losses, training/validation selection, checkpoint seals, evaluation, exact bidirectional screening and detailed prediction interfaces.
- Added bond-order-weighted atom messages. Distinguished interaction attention slots from independently supervised region slots.
- Added optional independent RDKit conformers, frozen ESM preparation with overlapping long-sequence windows, SQLite raw/encoded caches and strict feature/checkpoint compatibility.
- Added structural-label conversion from explicitly mapped complex atoms; native geometry is confined to label generation.
- Added missing-label, gradient-routing, invariance, caching, split and metric tests plus an isolated synthetic end-to-end acceptance.
- Added agent execution rules, data contract, architecture, deployment, operations and maintenance documentation.
- Installation initially lost its SSH session; the old pip process was confirmed absent before a locked, logged installer reused downloads and completed.
- Validation: see the versioned final deployment receipt for the completed isolated-environment checks. Initial local CPU end-to-end smoke passed before transfer. No full scientific training or million-library benchmark is claimed by this entry.

Scientific impact: new project and new architecture; no historical partitions, measurements, controllers or model results were changed.

### Acceptance hardening (same deployment)

- Stage 2: 22 unit tests and the development-1 CPU end-to-end acceptance passed; the same pair's cached bidirectional/detailed score spread was 1.79e-7.
- Stage 3: 23 tests passed after adding actual interruption/resume, checkpoint tamper detection and a test-label boundary check. Distributed workers now synchronize nonfinite-loss decisions and preserve truly unused parameter gradients.
- Real ESM test initially caught an incorrect hard-coded sequence-length assertion in the acceptance script (19 versus the actual 20). The failed log was retained; the assertion now derives length from the input. It did not change model inputs or scientific data.
- Recorded per-head training-label coverage and suppressed quantitative outputs for endpoints without recorded training labels.
- Real ESM loading explicitly uses residue representations without the optional ESM contact-regression sidecar. Our supervised contact decoder is a separate model head.
- Real ESM-2 650M / two retained RDKit conformers / full 384-wide, six-interaction-block model passed CPU forward/backward checks (8,983,285 trainable parameters). Two CPU processes also completed synchronized training and selection.
- Added permutation-matched observed-region Dice/IoU metrics and a corresponding unit test; saved the resolved Linux dependency lock. The final test run verifies these last changes.
