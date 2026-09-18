"""Real synthetic forward -> final diagnostic artifacts -> complete main table/figure pipeline.
All fixtures stay under tmp and are never paper evidence.
"""
import copy,hashlib,json,shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from modules.reproduction import settings,ROOT,HERE
from modules.paper_results import build,derive,sender_column,build_sender
import pandas as pd
from modules.model_IO import construct_model
from modules.training_helper import inspect_model
from modules.utils import move_to_device

class Writer:
    def __init__(self,path):self.log_dir=str(path);path.mkdir(parents=True,exist_ok=True)
    def __getattr__(self,name):
        if name.startswith('add_'):return lambda *a,**kw:None
        raise AttributeError(name)

def main():
    torch.set_num_threads(2)
    out=ROOT/'tmp/doubly_reproduction/artifact_smoke'
    if out.exists():shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)
    report={'synthetic_only':True,'datasets':{}}
    for ds in ['mnist','nmnist']:
        rows=settings(ds);r=rows[0]
        model=move_to_device(construct_model(copy.deepcopy(r['model']),run_seed=r['seed']),torch.device('cpu'))
        model.device=torch.device('cpu')
        # Exercise active as well as silent root neurons, with synthetic inputs only.
        for c in model.cortex.iter_cortex_tree():c.hidden_nv.log_thresholds.fill_(-5.)
        shape=(2,40,34,34) if ds=='nmnist' else (2,6,28,28)
        x=torch.rand(shape,generator=torch.Generator().manual_seed(1))
        if ds=='nmnist':x=(x>.98).float()
        loader=DataLoader(TensorDataset(x,torch.tensor([0,1])),batch_size=2)
        t=r['training_settings'];e=t['max_epoch'];w=Writer(out/ds)
        from modules import training_helper
        training_helper.cached_sample_data=None
        inspect_model(model,{'valid':loader,'test':loader},w,e,t['readout_params'],level=3,
                      diagnostic_settings=t['diagnostic_settings'],phases=['valid','test'],evaluation_micro_batch_size=2)
        art=out/ds/'mechanism_artifacts'
        arrays=[dict(np.load(art/name,allow_pickle=False)) for name in [f'native_decisions_test_epoch{e:03d}.npz',f'temporal_contribution_test_epoch{e:03d}.npz',f'root_weight_epoch{e:03d}.npz']]
        metrics,rates,sv=derive(*arrays)
        assert np.isclose(arrays[1]['root_receiver_first_spike_count'].sum(),arrays[1]['layer_first_spike_count_hidden_A'].sum())
        pred=arrays[0]['predicted_class'];decision=arrays[0]['has_decision'];labels=arrays[0]['labels']
        assert np.isclose(metrics['accuracy'],100*((pred==labels)&decision).mean())
        report['datasets'][ds]={'receivers':len(rates),'singular_values':len(sv),'metrics':metrics}
        for row in rows:
            dest=out/'logs'/row['experiment_name']/row['sub_exp_name']
            shutil.copytree(art,dest/'mechanism_artifacts',dirs_exist_ok=True)
            digest=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
            (dest/'reproduction_complete.json').write_text(json.dumps({'settings_sha256':digest,'synthetic_only':True}))
    frame=build(out/'chapter',out/'logs');assert len(frame)==80
    assert all((out/'chapter'/f'table{i}.csv').is_file() for i in range(1,5))
    # Missing artifacts must fail rather than substituting historical data.
    row=settings('mnist')[0];epoch=row['training_settings']['max_epoch'];target=out/'logs'/row['experiment_name']/row['sub_exp_name']/f'mechanism_artifacts/root_weight_epoch{epoch:03d}.npz'
    backup=target.read_bytes();target.unlink()
    try:
        try:build(out/'missing_check',out/'logs')
        except FileNotFoundError:report['missing_data_rejected']=True
        else:raise AssertionError('Missing data accepted')
    finally:target.write_bytes(backup)
    sender_rows=[]
    sender_logs=out/'sender_fixture_logs'
    for arm in ['adaptive_post','adaptive_pre']:
        path=ROOT/'tmp/doubly_reproduction/current_runtime_smoke'/f'mnist_common_e1_{arm}_unscaled_support_seed40500'/'sender_decision/epoch010/valid/native_interventions.csv'
        source=pd.read_csv(path)
        # Expand the ten real synthetic diagnostic samples only to exercise the
        # production CSV contract (four seeds, twenty samples per class).
        expanded=pd.concat([source.assign(sample_index=source.sample_index+10*i) for i in range(20)],ignore_index=True)
        for seed in range(40500,40504):
            dest=sender_logs/'paper_doubly_sender'/f'mnist_common_e1_{arm}_unscaled_support_seed{seed}'/'sender_decision/epoch020/valid'
            dest.mkdir(parents=True,exist_ok=True)
            expanded.to_csv(dest/'native_interventions.csv',index=False)
            info=json.loads(path.with_name('manifest.json').read_text())
            info['epoch']=20;info['sample_count']=200;info['settings']['samples_per_class']=20
            (dest/'manifest.json').write_text(json.dumps(info))
        for variant,group in source.groupby('variant'):
            sender_rows.append(dict(arm=arm,variant=variant,accuracy_percent_mean=100*group.correct.mean()))
    sender_summary=build_sender(logs=sender_logs)
    assert len(sender_summary)==8
    sender_column(sender_summary,out/'chapter')
    expected={f'figure{i}.pdf' for i in range(1,4)}|{f'table{i}.csv' for i in range(1,5)}
    actual={path.name for path in (out/'chapter').iterdir() if path.is_file()}
    assert actual==expected,(sorted(expected),sorted(actual))
    report['main_table_count']=4;report['figure_count']=3
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
