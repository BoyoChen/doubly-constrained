"""Validate complete chapter coverage and current runtime; no training."""
import argparse, copy, gc, json, sys
from pathlib import Path
import yaml
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'code'))
from modules.reproduction import settings

# The normalization contract each condition must satisfy, as
# (A-1 post, A-1 pre, A post, A pre, freq_diff). 0 = never, 1 = every update,
# 2 = every other update; freq_diff offsets pre against post.
SCHEDULES={
 'none_all_layers':(0,0,0,0,0),'post_every':(1,0,1,0,0),
 'post_half_all_layers':(2,0,2,0,0),'pre_all_layers':(0,1,0,1,0),
 'pre_half_all_layers':(0,2,0,2,1),'all_layers':(2,2,2,2,1),
 'hidden_only':(2,2,1,0,1),'alternating':(1,0,2,2,1),
}

# The lifetime control: each condition reuses a placement from SCHEDULES and
# changes only how long the layers stay plastic. life20 disables the staged
# freeze; life2 stops training after the two plastic epochs.
LIFETIME_PLACEMENTS=['post_every','hidden_only','alternating']
LIFETIME_SPANS={'life20':(False,20),'life2':(True,2)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--construct-models',action='store_true');a=p.parse_args()
    coverage=json.loads((ROOT/'code/coverage.json').read_text());report={'groups':{},'models':[],'full_training':False}
    for group,file in [('mnist','01_mnist.yaml'),('nmnist','02_nmnist_t20.yaml'),('sender','03_sender.yaml'),
                       ('lifetime_mnist','04_lifetime_mnist.yaml'),('lifetime_nmnist','05_lifetime_nmnist.yaml')]:
        raw=(ROOT/'code/experiments'/file).read_text(encoding='utf8');doc=yaml.safe_load(raw)
        assert list(doc)==['experiment_name','shared_settings','sub_experiments']
        assert not any(isinstance(e,yaml.AliasEvent) or getattr(e,'anchor',None) for e in yaml.parse(raw))
        rows=settings(group);assert len({r['sub_exp_name'] for r in rows})==len(rows)
        lifetime=group.startswith('lifetime_')
        if lifetime:
            dataset=group.split('_',1)[1];cov=coverage['lifetime']
            names={r['sub_exp_name'].rsplit('_seed',1)[0][len(dataset)+1:] for r in rows}
            assert names=={f'{p}_{s}' for p in LIFETIME_PLACEMENTS for s in LIFETIME_SPANS}
            for r in rows:
                name=r['sub_exp_name'].rsplit('_seed',1)[0][len(dataset)+1:]
                placement,span=name.rsplit('_',1)
                frozen,epochs=LIFETIME_SPANS[span];t=r['training_settings']
                assert r['seed'] in cov['seeds'][dataset] and t['max_epoch']==epochs
                assert t['training_schedule']['enabled']==frozen
                if frozen:
                    assert t['training_schedule']['stages']==[{'start_epoch':2,'frozen_cortex_ids':['A-1','A']}]
                n=r['model']['cortex_spec']['base_cortex_settings']['kernel_settings']['normalize_settings']
                assert (n['dim_0_freq_by_cortex']['A-1'],n['dim_1_freq_by_cortex']['A-1'],
                        n['dim_0_freq_by_cortex']['A'],n['dim_1_freq_by_cortex']['A'],
                        n['freq_diff'])==SCHEDULES[placement],name
        elif group!='sender':
            assert {r['sub_exp_name'].rsplit('_seed',1)[0][len(group)+1:] for r in rows}==set(SCHEDULES)
        assert len(rows)==(12 if group=='sender' else 24 if lifetime else 80);report['groups'][group]=len(rows);seen=set()
        for r in rows:
            assert 'sham' not in r['sub_exp_name']
            if lifetime:
                continue
            if group!='sender':
                ds=coverage['datasets'][group];t=r['training_settings'];d=t['diagnostic_settings']
                assert r['seed'] in ds['seeds'] and t['max_epoch']==ds['final_epoch']
                assert t['evaluation_level']>=3 and not t['save_best_model']
                assert d['normalization_mechanism_save_full_weight']
                assert d['normalization_mechanism_full_weight_epochs']==[ds['final_epoch']]
                assert d['normalization_mechanism_trajectory_epochs']==[ds['final_epoch']]
                assert 'test' in d['normalization_mechanism_trajectory_phases']
                name=r['sub_exp_name'].rsplit('_seed',1)[0][len(group)+1:]
                n=r['model']['cortex_spec']['base_cortex_settings']['kernel_settings']['normalize_settings']
                assert name in SCHEDULES, name
                assert (n['dim_0_freq_by_cortex']['A-1'],n['dim_1_freq_by_cortex']['A-1'],
                        n['dim_0_freq_by_cortex']['A'],n['dim_1_freq_by_cortex']['A'],
                        n['freq_diff'])==SCHEDULES[name],name
                if group=='nmnist':
                    assert r['data']['event_frame_settings']['temporal_bins']==20
                    assert r['model']['image_encoder_spec']['encoder']=='direct_event_bins'
                    assert r['model']['cortex_spec']['base_cortex_settings']['amplifier_settings']['log_amplifier_delta']==0.0
                else:
                    amp=r['model']['cortex_spec']['base_cortex_settings']['amplifier_settings']
                    threshold=r['model']['cortex_spec']['neuron_vectors_spec']['threshold_settings']
                    assert amp['log_amplifier_delta_by_cortex']['A']==1.25e-6
                    assert amp['target_mean_first_spike_earliness_by_cortex']['A']==.38
                    assert threshold['target_mean_first_spike_factor_by_cortex']['A']==.075
            elif 'load_from' in r['model']:
                t=r['training_settings'];s=t['diagnostic_settings']['sender_decision_study']
                assert set(r['model'])=={'load_from'} and set(r['model']['load_from'])=={'path'}
                assert t['max_epoch']==19 and t['epoch_offset']==1
                assert s['support_normalization']=='none' and s['samples_per_class']==20
                assert s['deletion_fraction']==.1 and 20 in s['epochs']
            base=r['sub_exp_name'].rsplit('_seed',1)[0]
            if a.construct_models and base not in seen and 'load_from' not in r['model']:
                import torch
                from modules.model_IO import construct_model
                torch.set_num_threads(2);print('Constructing',base,flush=True)
                m=construct_model(copy.deepcopy(r['model']),run_seed=r['seed'])
                assert all(torch.isfinite(c.kernel.weight).all() for c in m.cortex.iter_cortex_tree())
                report['models'].append(base);del m;gc.collect()
            seen.add(base)
    out=ROOT/'tmp/doubly_reproduction/validation.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
