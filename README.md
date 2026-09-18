# Doubly-constrained normalization: experiment reproduction

Self-contained training and analysis code for the Doubly experimental chapter.
Three experiment files cover **four tables and three figures**. The goal is to
generate complete new results, not match provisional numbers from older drafts.
No historical results, pretrained models, paper drafts, or private services are required.

## Installation

Formal training runs on a **Linux NVIDIA GPU host**. Python 3.10, PyTorch 2.4.1,
and torchvision 0.19.1 are the reference stack. CPU synthetic checks also work on Windows.

```bash
git clone git@github.com:BoyoChen/doubly-constrained.git
cd doubly-constrained
conda create -n doubly-constrained python=3.10 pip -y
conda activate doubly-constrained
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The final value should be `True` on the training host. For CPU-only checks, use
the PyTorch `https://download.pytorch.org/whl/cpu` index instead. Alternatively,
`conda env create -f environment.yml` installs the platform-default PyTorch distribution;
check CUDA availability before training.

## Reproduce the chapter

Run these commands from the repository root with the environment activated:

```bash
# Plan only: enumerate 92 runs; no training, downloads, or job submission.
bash scripts/run_commands.sh plan paper-v3

# CPU synthetic checks: construction, updates, checkpoint continuation,
# interventions, diagnostic artifacts, and table/figure generation.
bash scripts/run_commands.sh check

# Linux GPU: train all three groups, then generate all chapter assets.
bash scripts/run_commands.sh train paper-v3

# Rebuild tables and figures from completed runs without retraining.
bash scripts/run_commands.sh analyze paper-v3
```

`paper-v3` is a run label; choose a new label for an independent rerun. The script
executes runs sequentially. It uses neither Prefect nor an external experiment server.
Windows checks can be run without Bash:

```powershell
python scripts/validate.py --construct-models
python scripts/smoke.py
python scripts/artifact_smoke.py
```

### Data

The loaders download MNIST and N-MNIST on first training use, into `data/`.
N-MNIST uses native binary events, two polarity channels, and **20 temporal bins**;
it does not convert events into latency-encoded images. Downloads require network
access and sufficient disk space for the datasets and caches. For an offline host,
prepare the following raw N-MNIST layout in advance:

```text
data/NMNIST/Train/<digit>/*.bin
data/NMNIST/Test/<digit>/*.bin
```

Alternatively place the official `train.zip` and `test.zip` in `data/NMNIST/`;
the loader checks the configured MD5 values and extracts them. Dataset terms
remain separate from the code's MIT license; datasets are not redistributed here.

## Experiments and outputs

| Configuration under `code/experiments/` | Runs | Chapter coverage |
|---|---:|---|
| `01_mnist.yaml` | 10 conditions × 4 seeds, 100 epochs | MNIST columns in Tables 1–2; Figure 1 and Table 3 |
| `02_nmnist_t20.yaml` | 10 conditions × 4 seeds, 8 epochs | N-MNIST columns in Tables 1–2; Figure 2 and Table 4 |
| `03_sender.yaml` | 4 shared prefixes + 8 continuations, 1 + 9 epochs | Figure 3: sender interventions |

The schedule grid changes only output layer A. It includes Post every batch,
Post every other batch, None, Pre every batch, Pre every other batch, alternating,
Post-then-Pre every batch, and Post-then-Pre every other batch. Two additional
conditions apply alternating normalization to the hidden layer only or to both layers.
The Post and output-only conditions are reused across comparisons rather than trained twice.

See [PROTOCOL.md](PROTOCOL.md) for definitions and interpretation limits, and
[scripts/coverage.json](scripts/coverage.json) for the machine-readable mapping.

Results are written to `output/doubly_reproduction/<label>/chapter/`:

- `table1`–`table4`: CSV and LaTeX; `tables.md` for quick inspection.
- `figure1`–`figure3`: PDF, SVG, and PNG, sized for a single paper column.
- `all_metrics_per_seed.csv`, `figure1_per_seed.csv`, and `sources.json`: raw
  aggregate inputs, seed variation, and source hashes.
- `sender/`: sample-level interventions, per-seed summaries, and the horizontal
  version of the sender figure.
- `COMPLETE.json`: created only after all four tables and three figures succeed.

The LaTeX fragments require `booktabs`, `graphicx`, and `xcolor`. New results may
change the scientific conclusion. Figure 2 preserves the preview's axis ranges
when possible; if results exceed them, both panels expand with equal spans and
record the change in `figure2_axes.json`.

## Separate groups and interruption handling

```bash
python scripts/run_suite.py mnist --label paper-v3 --execute
python scripts/run_suite.py nmnist --label paper-v3 --execute
bash scripts/sender_commands.sh train paper-v3

# Optional: one seed within a group (useful for independent GPU jobs).
python scripts/run_suite.py mnist --seed 34000 --label paper-v3 --execute
```

For sender jobs, complete the shared prefixes before their continuations:

```bash
python scripts/run_suite.py sender --stage prefix --label paper-v3 --execute
python scripts/run_suite.py sender --stage continuation --label paper-v3 --execute
bash scripts/sender_commands.sh analyze paper-v3
```

Completed runs are skipped only when their configuration hashes match. Existing
unfinished logs cause an error; use a new label rather than overwrite them.
There is no automatic partial-training resume. Missing runs or diagnostic artifacts
cause analysis to fail; **no old or substitute values are filled in**.

## Validation status

This export is tested with CPU synthetic data, including new artifact generation,
checkpoint continuation and intervention checks. Full 92-run dataset training
has **not** been executed for this release. Test outputs remain in ignored `tmp/`
and are not experimental evidence. See [VALIDATION.md](VALIDATION.md).

The necessary framework modules are included because the model and diagnostics
import them transitively. Unrelated experiment YAMLs, orchestration services,
research notes, datasets, model weights and old results are excluded.
[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records the source snapshot and exported file hashes.

## License

The code is released under the [MIT License](LICENSE). Third-party dependencies
and datasets retain their respective licenses. Paper citation metadata will be
added when the public paper details are finalized.
