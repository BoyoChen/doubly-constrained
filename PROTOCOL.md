# Experiment protocol

All chapter tables use the fixed final epoch and the complete test split. MNIST
uses seeds 34000–34003; native N-MNIST T=20 uses 35100–35103. Accuracy is the native
spike decision, including no-decision samples as incorrect; no learned readout is substituted.

## Normalization and layer comparison

P denotes pre-normalization; Q denotes post-normalization; `-` means no operation.
Alternating uses `P Q P Q`. Simultaneous here means **Q then P**, matching the
implementation order. The half-frequency simultaneous control (`- QP - QP`)
matches alternating's total operation count over two updates. Cadence is measured
in learning batch updates, not epochs.

For output-only schedules, A-1 retains Post every batch and A receives the selected
schedule. The hidden-only and all-layers conditions use alternating normalization
at the named layers. A-1 is frozen from epoch 2 in all conditions. Thus the hidden
comparison reflects its shorter learning window. Layer placement alone does not
isolate supervision as a causal factor; do not claim it proves that supervision
is necessary.

## Structure and activity

Figure 1 and Table 3 use the root A learnable branch weight matrix, excluding
the fixed-zero direct block. Activity uses A's internal receiver (`hidden_nv`)
population corresponding to that matrix's columns, not upstream A-1 neurons.

- Activity rate: each receiver's first-spike count divided by test sample count.
- Dead fraction: proportion of receivers with no first spike across the test split.
- Activity variance: population variance across receiver activity rates.
- Mean spikes: total root receiver first spikes divided by sample count.
- Effective rank: `exp(-sum(p * log(p)))`, where `p = singular_values / sum(singular_values)`.
- Participation ratio: `(sum(s**2))**2 / sum(s**4)`.

Zero-weight matrices have effective rank and participation ratio zero by explicit
convention. Complete weights and per-receiver counts are retained for reanalysis.
Curves average per-seed ranked profiles; they do not assume neuron identities align across seeds.

## Correct/wrong evidence

Figure 2 and Table 4 compare N-MNIST Post with output-only alternating Doubly.
Correct earliness is the true class's first-spike earliness; wrong earliness is
the strongest wrong class's value, with silent classes assigned zero. Gap is
correct minus wrong. These are timing scores, not current amplitudes.
Timing and no-decision table entries use percentages and percentage-point changes.
Accuracy differences are absolute percentage-point differences, not relative gains.

## Sender intervention

Seeds 40500–40503 each train one shared epoch-1 prefix. Post and Doubly then load
the same seed's prefix and train nine more epochs. A-1 remains frozen; the
normalization difference and interventions act only at root A.

At epoch 10, each seed contributes a validation diagnostic subset of 200 samples,
20 per class. This is not full-test accuracy. The four primary variants are intact,
random removal, uniform attenuation, and top-sender removal. Top removal selects
the highest-support 10% (rounded up) within each model/sample's positive-support
sender pool; random removal matches the pool and removal count. Uniform attenuation
matches the removed positive support. Mask seed is 849.

Primary support retains earliness weighting but does not divide by receiver
thresholds. Threshold-normalized diagnostics are recorded separately and excluded
from the main figure. Every intervention reruns the native forward pass; model
parameters and RNG state are restored. Ground-truth labels are used only for diagnosis.

Removal sets may differ between models. Matched support is not matched timing.
Interpretation is conditional decision dependence, not a proof of feature quality
or universal robustness. CSVs retain per-seed variation; figures omit error bars.
