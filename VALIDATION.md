# Export validation

Validated on 2026-09-18, from this standalone repository, using the Python 3.10
`boyonet` test environment on Windows. This is CPU synthetic validation, not a
full dataset training run or a fresh Linux/CUDA environment installation test.

Passed:

- All 19 imported framework modules resolve inside this repository.
- Three YAMLs expand into 40 MNIST, 40 native N-MNIST T=20, and 12 sender runs.
- All 21 distinct from-scratch model configurations construct with finite weights.
- Two synthetic learning updates per from-scratch configuration.
- Both sender continuations load the shared checkpoint; upstream weights stay frozen.
- Native sender interventions preserve model parameters and RNG state; primary
  unscaled and separate threshold-normalized diagnostics both execute.
- Final native decisions, full output weights, and receiver activity are produced
  for both datasets. Receiver counts agree with layer first-spike totals.
- The 80-cell synthetic fixture produces four tables and three figure sets.
- Missing input artifacts are rejected rather than replaced by historical data.
- Python compilation, Bash syntax, and the complete dry-run command.

Re-run with `bash scripts/run_commands.sh check`. Detailed local reports are
written under ignored `tmp/doubly_reproduction/` and `tmp/package_validation/`.
Synthetic fixtures and generated plots are deliberately excluded from Git.

The full 92-run experiment suite is **not yet executed for this release**. Download
availability, GPU training completion, and final numerical results must be verified
on the training host. These checks do not establish the paper's scientific claims.
