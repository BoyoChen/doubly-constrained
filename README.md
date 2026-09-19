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

# Train all three experiments, then generate three figure PDFs and four table CSVs.
python main.py experiments

# Or run one experiment file.
python main.py experiments/01_mnist.yaml
python main.py experiments/02_nmnist_t20.yaml
python main.py experiments/03_sender.yaml
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
`saved_models/paper_doubly_sender/`. MNIST and N-MNIST do not save model
checkpoints. After all 92 runs finish, the analysis step writes the paper-ready
sample outputs directly to `result/`.

### Experiment files

| File | Dataset | Seeds | Training | Outputs |
|---|---|---:|---:|---|
| `01_mnist.yaml` | MNIST | 34000–34003 | 20 epochs | Tables 1–3, Figure 1 |
| `02_nmnist_t20.yaml` | Native N-MNIST, T=20 | 35100–35103 | 20 epochs | Tables 1, 2, 4; Figure 2 |
| `03_sender.yaml` | MNIST | 40500–40503 | 1 shared + 19 continuation epochs | Figure 3 |

MNIST follows the receiver-side-amplifier-compatible Temporal-Margin `ex767`/`ex781`
substrate; N-MNIST follows its native T=20 `ex788` fixed-amplifier-10 substrate.
Both use the same 20-epoch budget and freeze A-1 from epoch 2. Tables 1–4 use the
final epoch and full test split. Accuracy is the native spike decision;
no-decision samples are incorrect.

### Normalization conditions

`P` = pre-normalization, `Q` = post-normalization, `-` = no operation. Sequences
are indexed by learning updates, not epochs.

| Condition | A-1 | Output layer A | Sequence at A |
|---|---|---|---|
| Post every batch | Q every batch | Q every batch | `Q Q Q Q` |
| Post every other batch | Q every batch | Q every other batch | `- Q - Q` |
| None | Q every batch | None | `- - - -` |
| Pre every batch | Q every batch | P every batch | `P P P P` |
| Pre every other batch | Q every batch | P every other batch | `P - P -` |
| Doubly, output only | Q every batch | Alternating P/Q | `P Q P Q` |
| Post then Pre, every batch | Q every batch | Q then P | `QP QP QP QP` |
| Post then Pre, every other batch | Q every batch | Q then P every other batch | `- QP - QP` |
| Doubly, hidden only | Alternating P/Q | Q every batch | `Q Q Q Q` |
| Doubly, all layers | Alternating P/Q | Alternating P/Q | `P Q P Q` |

### Chapter outputs

| Output | Content | Evaluation data |
|---|---|---|
| Table 1 | Normalization schedules; MNIST and N-MNIST accuracy | Full test set |
| Table 2 | Hidden/output/all-layer placement accuracy | Full test set |
| Figure 1, Table 3 | Output-layer receiver activity and weight spectrum | MNIST test set |
| Figure 2, Table 4 | Correct/wrong earliness, gap, no-decision and accuracy | N-MNIST test set |
| Figure 3 | Sender removal and attenuation | 200 validation samples per seed |

Results are saved directly to `result/` as exactly seven files:
`figure1.pdf`--`figure3.pdf` and `table1.csv`--`table4.csv`.

### Reported metrics

| Metric | Definition |
|---|---|
| Activity rate | Receiver first-spike count / number of samples |
| Dead fraction | Receivers with no first spike / all receivers |
| Activity variance | Population variance of receiver activity rates |
| Mean spikes | Root receiver first spikes / number of samples |
| Effective rank | `exp(-sum(p log p))`, where `p = s / sum(s)` |
| Participation ratio | `(sum(s²))² / sum(s⁴)` |
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

The formal 92-run Prefect reproduction completed on 2026-09-19. All tables and
figures below were generated from those completed runs.

## Results

The verified outputs below are reproducible samples from the full 92-run
experiment. Figure 1 uses the recorded epoch-20 receiver-activity deciles and
the full output-layer weight matrices. Accuracy and timing use the full test
set; receiver activity uses the configured 30-batch diagnostic subset.

### Figures

| Figure | Content | Sample |
|---|---|---|
| Figure 1 | Receiver activity and output-layer singular values | [PDF](result/figure1.pdf) |
| Figure 2 | Correct and wrong evidence earliness | [PDF](result/figure2.pdf) |
| Figure 3 | Sender removal and attenuation | [PDF](result/figure3.pdf) |

### Tables

#### Table 1 — Normalization schedule

| Method | Schedule | MNIST | N-MNIST |
|---|:---:|---:|---:|
| Post every batch | `Q Q Q Q` | 97.948 ± 0.013 | 97.143 ± 0.091 |
| Post every other batch | `- Q - Q` | 97.932 ± 0.145 | 97.152 ± 0.142 |
| None | `- - - -` | 75.478 ± 1.316 | 37.145 ± 3.012 |
| Pre every batch | `P P P P` | 98.110 ± 0.114 | 97.435 ± 0.060 |
| Pre every other batch | `P - P -` | 98.180 ± 0.045 | 97.420 ± 0.028 |
| Doubly alternating | `P Q P Q` | **98.233 ± 0.090** | **97.465 ± 0.119** |
| Post then pre every batch | `QP QP QP QP` | 98.100 ± 0.132 | 97.432 ± 0.079 |
| Post then pre every other batch | `- QP - QP` | 98.075 ± 0.101 | 97.422 ± 0.036 |

[CSV](result/table1.csv)

#### Table 2 — Layer placement

| Method | MNIST | N-MNIST |
|---|---:|---:|
| Post every batch | 97.948 ± 0.013 | 97.143 ± 0.091 |
| Doubly hidden only | 97.805 ± 0.097 (-0.142) | 96.820 ± 0.141 (-0.322) |
| Doubly alternating | 98.233 ± 0.090 (+0.285) | 97.465 ± 0.119 (+0.322) |
| Doubly all layers | 98.120 ± 0.050 (+0.173) | 97.430 ± 0.068 (+0.287) |

[CSV](result/table2.csv)

#### Table 3 — Activity and spectrum summary

| Method | Effective rank | Participation ratio | Dead (%) | Activity variance | Mean spikes |
|---|---:|---:|---:|---:|---:|
| Post every batch | 56.7697 ± 0.2218 | 1.0589 ± 0.0016 | 0.2500 ± 0.2887 | 0.0015 ± 0.0000 | 8.3473 ± 0.0475 |
| None | 23.3711 ± 0.0116 | 1.0077 ± 0.0000 | 0.0000 ± 0.0000 | 0.0013 ± 0.0001 | 107.4066 ± 0.0331 |
| Pre every batch | 70.3395 ± 1.1047 | 1.0772 ± 0.0052 | 0.0000 ± 0.0000 | 0.0012 ± 0.0000 | 11.9645 ± 0.0424 |
| Doubly alternating | 69.1633 ± 0.9034 | 1.0783 ± 0.0036 | 0.0000 ± 0.0000 | 0.0014 ± 0.0000 | 11.9755 ± 0.0382 |
| Post then pre every batch | 68.6510 ± 0.8676 | 1.0788 ± 0.0028 | 0.0000 ± 0.0000 | 0.0013 ± 0.0000 | 11.9602 ± 0.0616 |

[CSV](result/table3.csv)

#### Table 4 — N-MNIST decision and timing summary

| Metric | Post-only | Doubly |
|---|---:|---:|
| Accuracy | 97.143 | 97.465 (+0.322) |
| No decision | 0.173% | 0.030% (-0.142%) |
| Correct earliness | 46.559% | 52.685% (+6.126%) |
| Wrong earliness | 8.982% | 12.297% (+3.315%) |
| Gap | 37.577% | 40.388% (+2.810%) |

[CSV](result/table4.csv)

## License

The code is released under the [MIT License](LICENSE). Third-party dependencies
and datasets retain their respective licenses. Paper citation metadata will be
added when the public paper details are finalized.
