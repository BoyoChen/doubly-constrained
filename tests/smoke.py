"""CPU synthetic-data smoke for current runtime; no datasets, dispatch or full training."""
import copy
import gc
import json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'code'))
from modules.model_IO import construct_model, save_model
from modules.model_trainer import apply_training_schedule
from modules.sender_decision_study import parameter_fingerprint, log_sender_decision_study
from modules.utils import move_to_device, set_global_seed
from modules.reproduction import settings


class Writer:
    def __init__(self,path):
        self.log_dir=str(path);Path(path).mkdir(parents=True,exist_ok=True)
    def add_scalar(self,*a,**k):pass
    def add_text(self,*a,**k):pass


def finite(model):
    for c in model.cortex.iter_cortex_tree():
        assert torch.isfinite(c.kernel.weight).all()
        assert torch.isfinite(c.hidden_nv.log_thresholds).all()
        assert torch.isfinite(c.log_amplifier).all()


def main():
    torch.set_num_threads(2)
    out=ROOT/'tmp/doubly_reproduction/current_runtime_smoke'
    out.mkdir(parents=True,exist_ok=True)
    report={'synthetic_only':True,'full_training_executed':False,'update_checks':[], 'sender_checks':[]}
    seen=set();checkpoint=out/'shared_prefix'
    for group in ['mnist','nmnist','sender']:
        for row in settings(group,'smoke'):
            base=row['sub_exp_name'].rsplit('_seed',1)[0]
            if base in seen or 'load_from' in row['model']:continue
            seen.add(base);print('Two-update smoke:',base,flush=True)
            set_global_seed(row['seed'])
            model=move_to_device(construct_model(copy.deepcopy(row['model']),run_seed=row['seed']),torch.device('cpu'))
            model.device=torch.device('cpu')
            native=row['data']['dataset_name']=='NMNIST'
            shape=(2,40,34,34) if native else (2,6,28,28)
            images=torch.rand(shape,generator=torch.Generator().manual_seed(8494))
            if native:images=(images>.98).float()
            labels=torch.tensor([0,1]);writer=Writer(out/base)
            for epoch in [1,2]:
                apply_training_schedule(model,writer,epoch,row['training_settings']['training_schedule'])
                model.remember(images,labels,1,stdp_update_mode='two_pass_decision_contingent',simulation_forward_mode='streaming')
                finite(model)
                if base=='mnist_shared_prefix_e1' and epoch==1:
                    save_model(model,checkpoint)
            report['update_checks'].append(base)
            del model;gc.collect()
    for row in settings('sender','smoke',stage='continuation',seed=40500):
        name=row['sub_exp_name'];print('Checkpoint/continuation/intervention smoke:',name,flush=True)
        model=move_to_device(construct_model({'load_from':{'path':str(checkpoint.with_suffix('.pickle'))}},run_seed=40500),torch.device('cpu'))
        model.device=torch.device('cpu')
        before=parameter_fingerprint(model)
        writer=Writer(out/name)
        images=torch.rand((2,6,28,28),generator=torch.Generator().manual_seed(8494))
        for epoch in [2,3]:
            apply_training_schedule(model,writer,epoch,row['training_settings']['training_schedule'])
            model.remember(images,torch.tensor([0,1]),1,stdp_update_mode='two_pass_decision_contingent',simulation_forward_mode='streaming')
        finite(model);after=parameter_fingerprint(model)
        assert all(before[k]==after[k] for k in before if k.startswith('A-1/'))
        assert torch.count_nonzero(model.cortex.kernel.weight[:model.cortex.kernel.input_len])==0
        # Only this synthetic diagnostic fixture lowers thresholds to exercise nonzero masks.
        for c in model.cortex.iter_cortex_tree():c.hidden_nv.log_thresholds.fill_(-5.)
        images=torch.rand((10,6,28,28),generator=torch.Generator().manual_seed(8494))
        study=copy.deepcopy(row['training_settings']['diagnostic_settings']['sender_decision_study'])
        study.update(samples_per_class=1,micro_batch_size=2,phases=['valid'])
        frozen=parameter_fingerprint(model);rng=torch.random.get_rng_state().clone()
        log_sender_decision_study(model,{'valid':DataLoader(TensorDataset(images,torch.arange(10)))},writer,10,study)
        assert parameter_fingerprint(model)==frozen
        assert torch.equal(torch.random.get_rng_state(),rng)
        for namespace,definition in [('sender_decision','none'),('sender_decision_reference','threshold')]:
            info=json.loads((Path(writer.log_dir)/namespace/'epoch010/valid/manifest.json').read_text(encoding='utf-8'))
            assert info['support_normalization']==definition
        report['sender_checks'].append({'name':name,'checkpoint_loaded':True,'upstream_frozen':True,
            'native_interventions':True,'primary_and_reference_support':True,'parameters_and_rng_preserved':True})
        del model;gc.collect()
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
