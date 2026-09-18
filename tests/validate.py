"""Validate complete chapter coverage and current runtime; no training."""
import argparse, copy, gc, json, sys
from pathlib import Path
import yaml
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'code'))
from modules.reproduction import settings

def main():
    p=argparse.ArgumentParser();p.add_argument('--construct-models',action='store_true');a=p.parse_args()
    coverage=json.loads((ROOT/'code/coverage.json').read_text());report={'groups':{},'models':[],'full_training':False}
    for group,file in [('mnist','01_mnist.yaml'),('nmnist','02_nmnist_t20.yaml'),('sender','03_sender.yaml')]:
        raw=(ROOT/'code/experiments'/file).read_text(encoding='utf8');doc=yaml.safe_load(raw)
        assert list(doc)==['experiment_name','shared_settings','sub_experiments']
        assert not any(isinstance(e,yaml.AliasEvent) or getattr(e,'anchor',None) for e in yaml.parse(raw))
        rows=settings(group);assert len({r['sub_exp_name'] for r in rows})==len(rows)
        assert len(rows)==(12 if group=='sender' else 40);report['groups'][group]=len(rows);seen=set()
        for r in rows:
            assert 'sham' not in r['sub_exp_name']
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
                assert n['dim_0_freq_by_cortex']['A-1']==(2 if name in ['hidden_only','all_layers'] else 1)
                assert n['dim_1_freq_by_cortex']['A-1']==(2 if name in ['hidden_only','all_layers'] else 0)
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
