# Doubly-constrained normalization: experiment reproduction


## Installation

Formal training runs on a **Linux NVIDIA GPU host**. Python 3.10, PyTorch 2.4.1,
and torchvision 0.19.1 are the reference stack. CPU synthetic checks also work on Windows.

```bash
conda create -n doubly-constrained python=3.10 pip -y
conda activate doubly-constrained
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## Reproduce

```bash
cd code

# Train all three experiments, then generate three figure PDFs and three table CSVs.
python main.py experiments

# Or run one experiment file.
python main.py experiments/01_mnist.yaml
python main.py experiments/02_nmnist_t20.yaml
python main.py experiments/03_sender.yaml
python main.py experiments/04_lifetime_mnist.yaml
python main.py experiments/05_lifetime_nmnist.yaml
```

The sender experiment automatically trains its shared prefixes before loading
them for the continuations, then generates its intervention figure. If you ran
the files separately, assemble the complete chapter after all three finish:

```bash
python main.py experiments --analyze
```

### Data

The loaders download MNIST and N-MNIST on first training use, into `data/`.

## Experiments and outputs

### Output locations

Each run keeps its complete TensorBoard log and mechanism artifacts under its
experiment directory:

```text
logs/paper_doubly_mnist_v3/{run-name}/
logs/paper_doubly_nmnist_v3/{run-name}/
logs/paper_doubly_sender/{run-name}/
```

The sender experiment also saves the shared-prefix and final checkpoints under
`saved_models/paper_doubly_sender/`, because its two arms must continue from one
shared prefix. No other group saves model weights: the main MNIST and N-MNIST runs
and the lifetime control all set `save_final_model` and `save_best_model` to false. After all 220 runs finish, the analysis step writes the paper-ready
sample outputs directly to `result/`.

### Experiment files

| File | Dataset | Seeds | Runs | Training | Outputs |
|---|---|---:|---:|---:|---|
| `01_mnist.yaml` | MNIST | 34000–34009 | 80 | 20 epochs | Tables 1, 2, 4; Figure 1 |
| `02_nmnist_t20.yaml` | Native N-MNIST, T=20 | 35100–35109 | 80 | 20 epochs | Tables 1–4; Figure 2 |
| `03_sender.yaml` | MNIST | 40500–40503 | 12 | 1 shared + 19 continuation epochs | Figure 3 |
| `04_lifetime_mnist.yaml` | MNIST | 34000–34003 | 24 | 20 epochs, or 2 | Table 4 |
| `05_lifetime_nmnist.yaml` | Native N-MNIST, T=20 | 35100–35103 | 24 | 20 epochs, or 2 | Table 4 |

The main experiment runs ten seeds per condition; the lifetime control and the
sender study run four. `tests/validate.py` checks this table against the YAML
files, so it cannot drift from them.

MNIST follows the receiver-side-amplifier-compatible Temporal-Margin `ex767`/`ex781`
substrate; N-MNIST follows its native T=20 `ex788` fixed-amplifier-10 substrate.
Both use the same 20-epoch budget and freeze A-1 from epoch 2. Tables 1–4 use the
final epoch and full test split. Accuracy is the native spike decision;
no-decision samples are incorrect.

### Normalization conditions

`P` = pre-normalization, `Q` = post-normalization, `-` = no operation. Sequences
are indexed by learning updates, not epochs.

| Condition | Hidden layer A-1 | Output layer A | Four-update pattern |
|---|---|---|---|
| No constraint | None | None | `- - - -` |
| Post only | Q every batch | Q every batch | `Q Q Q Q` |
| Post only, 1/2 freq | Q every other batch | Q every other batch | `- Q - Q` |
| Pre only | P every batch | P every batch | `P P P P` |
| Pre only, 1/2 freq | P every other batch | P every other batch | `P - P -` |
| Doubly, alternating | Alternating P/Q | Alternating P/Q | `P Q P Q` |
| Doubly, hidden only | Alternating P/Q | Q every batch | `Q Q Q Q` |
| Doubly, output only | Q every batch | Alternating P/Q | `P Q P Q` |

### Chapter outputs

| Output | Content | Evaluation data |
|---|---|---|
| Table 1 | Six schedules applied at both layers; accuracy | Full test set |
| Table 2 | Hidden x output placement square; accuracy | Full test set |
| Table 3 | Correct/wrong earliness, gap, no-decision and accuracy | N-MNIST test set |
| Figure 1 | Hidden-unit firing-rate distribution and cumulative singular-value mass | MNIST test set |
| Figure 2 | Correct and wrong earliness | N-MNIST test set |
| Figure 3 | Sender removal and attenuation | 200 validation samples per seed |

Results are saved directly to `result/` as exactly seven files:
`figure1.pdf`--`figure3.pdf` and `table1.csv`--`table3.csv`.

### Reported metrics

| Metric | Definition |
|---|---|
| Firing rate | Mean per-unit activity of a hidden unit over the test set |
| Effective rank | `exp(-sum(p log p))`, where `p = s / sum(s)` |
| Correct earliness | First-spike earliness of the true class |
| Wrong earliness | Maximum first-spike earliness among wrong classes |
| Gap | Correct earliness − wrong earliness |

Activity and singular values use the learnable branch of output layer A; the
fixed-zero direct block and upstream A-1 activity are excluded.

### Sender intervention

| Parameter | Value |
|---|---|
| Evaluation | Epoch 20 validation subset |
| Samples | 200 per seed; 20 per class |
| Compared models | Post-only and output-only Doubly from the same seed prefix |
| Top removal | Highest-support 10% of the positive-support sender pool, rounded up |
| Random removal | Same sender pool and removal count; mask seed 849 |
| Uniform attenuation | Matches removed positive support |
| Support score | Earliness-weighted; no threshold division |

Figure 3 measures conditional decision dependence. Removal sets may differ between
models, and matched support does not imply matched timing.

All tables and figures below were generated from the 220 runs this repository
specifies, completed on 2026-09-19.

## Results

The verified outputs below are reproducible samples from the full 220-run
experiment. Every condition trained here is consumed by one of the three
figures or three tables. Figure 1 uses the recorded per-unit hidden activity
and the full output-layer weight matrices; accuracy and timing use the full
test set.

### Figures

| Figure | Content | Sample |
|---|---|---|
| Figure 1 | Hidden-unit firing rates and output-layer spectrum | [PDF](result/figure1.pdf) |
| Figure 2 | Correct and wrong evidence earliness | [PDF](result/figure2.pdf) |
| Figure 3 | Sender removal and attenuation | [PDF](result/figure3.pdf) |

### Tables

#### Table 1 — Normalization schedule

| Method | Pattern | MNIST | N-MNIST |
|---|:---:|---:|---:|
| No constraint | `- - - -` | 52.952 ± 0.827 | 18.980 ± 1.648 |
| Post only | `Q Q Q Q` | 97.948 ± 0.013 | 97.143 ± 0.091 |
| Post only, 1/2 freq | `- Q - Q` | 97.690 ± 0.104 | 97.067 ± 0.100 |
| Pre only | `P P P P` | 67.963 ± 1.735 | 44.685 ± 2.103 |
| Pre only, 1/2 freq | `P - P -` | 63.673 ± 4.892 | 45.565 ± 1.729 |
| Doubly, alternating | `P Q P Q` | **98.120 ± 0.050** | **97.430 ± 0.068** |

[CSV](result/table1.csv)

#### Table 2 — Layer placement

| Hidden | Output | MNIST | N-MNIST |
|---|---|---:|---:|
| Post | Post | 97.948 ± 0.013 | 97.143 ± 0.091 |
| Doubly | Post | 97.805 ± 0.097 (-0.142) | 96.820 ± 0.141 (-0.322) |
| Post | Doubly | 98.233 ± 0.090 (+0.285) | 97.465 ± 0.119 (+0.322) |
| Doubly | Doubly | 98.120 ± 0.050 (+0.173) | 97.430 ± 0.068 (+0.287) |

[CSV](result/table2.csv)

#### Table 4 — Layer placement against plastic lifetime

The paired difference between doubly at the output layer and doubly at the hidden
layer, in percentage points, with a 95% interval. The first row is the schedule used
everywhere else, at the ten seeds of Table 2. The other two give both layers the same
plastic span, at four seeds, which separates where the constraint acts from how long
each layer stays plastic. On N-MNIST with twenty epochs each, every network falls
silent by epoch 4, so there is nothing to compare.

| Hidden epochs plastic | Output epochs plastic | MNIST | N-MNIST |
| --- | --- | --- | --- |
| 2 | 20 | +0.386 [+0.278, +0.494] n=10 | +0.661 [+0.617, +0.705] n=10 |
| 20 | 20 | +1.163 [+0.817, +1.508] n=4 | silent |
| 2 | 2 | +6.575 [+5.145, +8.005] n=4 | +0.013 [-1.976, +2.001] n=4 |

#### Table 3 — N-MNIST decision and timing summary

| Metric | Post-only | Doubly |
|---|---:|---:|
| Accuracy | 97.143 | 97.465 (+0.322) |
| No decision | 0.173% | 0.030% (-0.142%) |
| Correct earliness | 46.559% | 52.685% (+6.126%) |
| Wrong earliness | 8.982% | 12.297% (+3.315%) |
| Gap | 37.577% | 40.388% (+2.810%) |

[CSV](result/table3.csv)

## What this reproduction does and does not give you

It retrains. Every number in `result/` comes from running the configurations in
`code/experiments/`, and `tests/validate.py` checks that those configurations are
the ones the tables describe. Two limits follow from that, and both matter if you
are checking the paper rather than building on it.

**No released weights.** Only the sender group saves checkpoints. There is no
published model for the ten main seeds, so you cannot compare weights against ours
bit for bit. Verification is by retraining and comparing the reported statistics.
Seeds are fixed, so a rerun on the same software and hardware should land very
close; a different GPU, driver or library version will not reproduce the same bits.

**No archival release.** This repository is a moving `main` branch, not a frozen
artifact with a DOI. The state the paper reports is tagged
`v1.0-icassp2027-submission`; cite that tag if you need a fixed reference. A Zenodo
or similar archival deposit is not yet in place.

**Logs are not published.** `logs/` and `saved_models/` are excluded by
`.gitignore`. Training all 220 runs is the way to regenerate them.

## License

The code is released under the [MIT License](LICENSE). Third-party dependencies
and datasets retain their respective licenses. `CITATION.cff` carries the metadata
this repository can state today; the paper reference and its DOI are added once the
paper is published.
