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

# Train all three experiments and generate all four tables and three figures.
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

Results are saved directly to `result/` as CSV, LaTeX,
PDF, SVG and PNG.

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

Full 92-run training has not yet been executed for this release. CPU synthetic
checks cover model construction, checkpoint continuation, diagnostics and output generation.

## License

The code is released under the [MIT License](LICENSE). Third-party dependencies
and datasets retain their respective licenses. Paper citation metadata will be
added when the public paper details are finalized.
