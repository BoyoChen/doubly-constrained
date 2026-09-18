"""Publication vector plot: adaptive Post only / Doubly, three interventions."""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'output/doubly_reproduction/paper-v3/chapter/sender'
OUT = SOURCE / 'figure'
STEM = 'ex849_sender_removal_adaptive'
VARIANTS = ['random_support', 'equal_drive_attenuation', 'top_support']


def label(value):
    return str(Decimal(f'{value:.8f}').quantize(Decimal('.1'), rounding=ROUND_HALF_UP))


def main(source=SOURCE, out=OUT, stem=STEM,
         continuation_commit='new-run',
         support_normalization='none', raw_relative='raw.csv'):
    out.mkdir(parents=True, exist_ok=True)
    per_seed = pd.read_csv(source / 'per_seed.csv')
    known = pd.read_csv(source / 'summary.csv')
    rows = []
    for (arm, variant), group in per_seed.groupby(['arm', 'variant']):
        values = group.accuracy_percent.to_numpy()
        assert len(values) == 4
        mean = values.mean()
        half = t.ppf(.975, 3) * values.std(ddof=1) / np.sqrt(4)
        expected = known[(known.arm == arm) & (known.variant == variant)].iloc[0]
        assert np.allclose([mean, mean-half, mean+half], [expected.accuracy_percent_mean,
                           expected.accuracy_percent_ci95_low, expected.accuracy_percent_ci95_high])
        rows.append({'arm': arm, 'variant': variant, 'mean_accuracy_percent': mean,
                     'ci95_low': mean-half, 'ci95_high': mean+half, 'n_training_seeds': 4})
    summary = pd.DataFrame(rows)
    font_path = Path('C:/Windows/Fonts/times.ttf')
    font_name = 'DejaVu Serif'
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({'font.family': font_name, 'font.size': 8.5,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': .55})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.1), sharey=True)
    fig.subplots_adjust(left=.065, right=.993, top=.80, bottom=.255, wspace=.20)
    for ax, arm, title, color in zip(axes, ['adaptive_post', 'adaptive_pre'],
                                    ['(a) Post only', '(b) Doubly'], ['#6C91B0', '#D99259']):
        frame = summary[summary.arm == arm].set_index('variant')
        baseline = frame.loc['identity', 'mean_accuracy_percent']
        means = frame.loc[VARIANTS, 'mean_accuracy_percent'].to_numpy()
        x = np.arange(3)
        ax.set_axisbelow(True)
        ax.bar(x, means, width=.54, color=color, edgecolor='#262626', linewidth=.55, zorder=2)
        ax.axhline(baseline, color='#444444', lw=.8, ls=(0, (4, 2.5)),
                   zorder=3, label=f'Original: {label(baseline)}%')
        ax.legend(loc='lower right', bbox_to_anchor=(1, 1.025), frameon=False, fontsize=8,
                  handlelength=2.2, handletextpad=.55, borderaxespad=0, borderpad=0)
        for xx, value in zip(x, means):
            ax.text(xx, value+3.0, label(value), ha='center', va='bottom',
                    fontsize=8, color='#111111', zorder=5)
        ax.set_title(title, loc='left', fontsize=10, weight='normal', pad=7)
        ax.set_ylim(0, 115)
        ax.set_xlim(-.6, 2.6)
        ax.set_yticks([0, 50, 100])
        ax.set_xticks(x, ['random\nremoval', 'uniform\nattenuation', 'top-sender\nremoval'], fontsize=8.5)
        ax.tick_params(axis='x', length=0, pad=4)
        ax.tick_params(axis='y', length=2.5, width=.55, labelleft=True, pad=3, labelsize=8)
        ax.spines['left'].set_color('#222222')
        ax.spines['bottom'].set_color('#222222')
    axes[0].set_ylabel('Accuracy (%)', fontsize=9, labelpad=5)
    fig.savefig(out / f'{stem}.svg', facecolor='white')
    svg = out / f'{stem}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    assert '<image' not in svg.read_text(encoding='utf-8')
    fig.savefig(out / f'{stem}.pdf', facecolor='white', metadata={
        'Title': 'Sender removal in adaptive Post only and Doubly models',
        'Subject': 'MNIST validation, epoch 10, means over four training seeds; uncertainty retained in source CSV'})
    fig.savefig(out / f'{stem}.png', dpi=240, facecolor='white')
    plt.close(fig)
    summary.to_csv(out / f'{stem}_summary.csv', index=False, encoding='utf-8-sig', float_format='%.10g')
    shutil.copyfile(source / 'per_seed.csv', out / f'{stem}_per_seed.csv')
    shutil.copyfile(source / raw_relative, out / f'{stem}_raw.csv')
    manifest = {'experiment': 'Ex849', 'continuation_commit': continuation_commit,
                'support_normalization': support_normalization,
                'dataset': 'MNIST', 'phase': 'validation', 'epoch': 10, 'adaptive_only': True,
                'samples_per_seed': 200, 'training_seeds': [40500, 40501, 40502, 40503],
                'bar': 'mean accuracy after intervention', 'whisker': 'not displayed, per user request',
                'uncertainty_data': '95% Student t interval across 4 seeds retained in summary CSV',
                'dashed_line': 'mean intact accuracy for the corresponding model',
                'figure_size_inches': [7.4, 2.1], 'style': 'compact blue/orange panels, serif text, thin strokes',
                'sources': {str((source / p).relative_to(ROOT)): hashlib.sha256((source / p).read_bytes()).hexdigest()
                            for p in ['per_seed.csv', 'summary.csv', raw_relative]},
                'controls': 'Top 10% of each model/input positive-support sender pool; random removal uses same pool and count; attenuation matches intact-forward positive-support reduction.',
                'limits': 'Removal sets and counts can differ across models. Matched support does not match timing or receiver patterns. Ground-truth labels used only for diagnosis.'}
    (out / f'{stem}_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'svg': str(svg), 'pdf': str(out / f'{stem}.pdf'), 'vector_svg': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
