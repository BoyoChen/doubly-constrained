"""Compute intervention summaries from this repository's new diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
from scipy.stats import t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VARIANTS = ['identity', 'random_support', 'equal_drive_attenuation', 'top_support']


def validate_run_manifest(info):
    if info.get('epoch') != 10 or info.get('sample_count') != 200 or info.get('support_normalization') != 'none':
        raise ValueError('Expected epoch 10, 200 diagnostic samples and unscaled support in run manifest.')
    settings=info.get('settings',{})
    if settings.get('samples_per_class') != 20 or settings.get('deletion_fraction') != .1 or settings.get('mask_seed') != 849:
        raise ValueError('Unexpected sender sampling/removal settings in run manifest.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', default='paper-v3')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.label):
        p.error('invalid label')
    out = a.output or ROOT/'output/doubly_reproduction'/a.label
    out = out.resolve()
    if not out.is_relative_to(ROOT):
        p.error('--output must be inside this repository (required by the existing figure manifest writer)')
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    frames = []
    for arm in ['adaptive_post', 'adaptive_pre']:
        for seed in range(40500, 40504):
            name = f'mnist_common_e1_{arm}_unscaled_support_seed{seed}'
            src = ROOT/'logs/paper_doubly_sender'/name/a.label/'sender_decision/epoch010/valid/native_interventions.csv'
            info_path=src.with_name('manifest.json')
            info=json.loads(info_path.read_text(encoding='utf-8'))
            validate_run_manifest(info)
            manifest.append({'path':str(info_path),'sha256':hashlib.sha256(info_path.read_bytes()).hexdigest()})
            frame = pd.read_csv(src)
            frame = frame.assign(arm=arm, seed=seed, epoch=10, phase='valid', support_normalization='none')
            frames.append(frame)
            manifest.append({'path': str(src), 'sha256': hashlib.sha256(src.read_bytes()).hexdigest()})
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[(raw.epoch == 10) & (raw.phase == 'valid') &
              (raw.support_normalization == 'none') & raw.variant.isin(VARIANTS)].copy()
    grouped = raw.groupby(['arm', 'seed', 'variant'], sort=True)
    expected = {(arm, seed, v) for arm in ['adaptive_post', 'adaptive_pre']
                for seed in range(40500,40504) for v in VARIANTS}
    if set(grouped.groups) != expected:
        raise ValueError('Missing or unexpected arm/seed/intervention cells.')
    per_seed = []
    for (arm, seed, variant), frame in grouped:
        if len(frame) != 200 or frame.sample_index.nunique() != 200:
            raise ValueError('Expected 200 distinct validation diagnostic samples per seed/intervention.')
        if len(frame.true_class.unique()) != 10 or not (frame.groupby('true_class').size() == 20).all():
            raise ValueError('Expected 20 samples per class for each intervention.')
        baseline = raw[(raw.arm == arm) & (raw.seed == seed) & (raw.variant == 'identity')]
        if not frame[['sample_index','true_class']].sort_values('sample_index').reset_index(drop=True).equals(
                baseline[['sample_index','true_class']].sort_values('sample_index').reset_index(drop=True)):
            raise ValueError('Interventions do not use the same samples and labels.')
        per_seed.append(dict(arm=arm, seed=seed, epoch=10, phase='valid', support_normalization='none', variant=variant,
            accuracy_percent=100*frame.correct.mean(), no_decision_percent=100*frame.no_decision.mean(), sample_count=200))
    per_seed = pd.DataFrame(per_seed)
    summary = []
    for (arm, variant), frame in per_seed.groupby(['arm', 'variant']):
        values=frame.accuracy_percent.to_numpy(); m=values.mean()
        ci=t.ppf(.975,3)*values.std(ddof=1)/np.sqrt(4)
        nd=frame.no_decision_percent.to_numpy();ndm=nd.mean();ndci=t.ppf(.975,3)*nd.std(ddof=1)/np.sqrt(4)
        summary.append(dict(arm=arm,epoch=10,phase='valid',support_normalization='none',variant=variant,n_training_seeds=4,
            accuracy_percent_mean=m,accuracy_percent_ci95_low=m-ci,accuracy_percent_ci95_high=m+ci,
            no_decision_percent_mean=ndm,no_decision_percent_ci95_low=ndm-ndci,no_decision_percent_ci95_high=ndm+ndci))
    raw.to_csv(out/'raw.csv',index=False)
    per_seed.to_csv(out/'per_seed.csv',index=False)
    summary=pd.DataFrame(summary);summary.to_csv(out/'summary.csv',index=False)
    (out/'sources.json').write_text(json.dumps(manifest,indent=2))
    sys.path.insert(0,str(ROOT/'code/diagnostics'))
    from plot_ex849_sender_removal_panels import main as plot
    plot(source=out,out=out/'figure',stem='ex849_sender_removal_adaptive_unscaled',raw_relative='raw.csv',
         support_normalization='none',continuation_commit='new-run-label:'+a.label)
    print(summary.pivot(index='variant',columns='arm',values='accuracy_percent_mean').to_string())
    print('Saved:',out)


if __name__ == '__main__':
    main()
