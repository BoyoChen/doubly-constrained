"""Chapter tables, figures and sender summaries from completed experiment runs."""
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from modules.reproduction import settings, ROOT, HERE

VARIANTS = ['identity', 'random_support', 'equal_drive_attenuation', 'top_support']

def validate_run_manifest(info):
    if info.get('epoch') != 20 or info.get('sample_count') != 200 or info.get('support_normalization') != 'none':
        raise ValueError('Expected epoch 20, 200 diagnostic samples and unscaled support in run manifest.')
    settings=info.get('settings',{})
    if settings.get('samples_per_class') != 20 or settings.get('deletion_fraction') != .1 or settings.get('mask_seed') != 849:
        raise ValueError('Unexpected sender sampling/removal settings in run manifest.')


def build_sender(logs=None):
    logs = ROOT/'logs' if logs is None else logs
    frames = []
    for arm in ['adaptive_post', 'adaptive_pre']:
        for seed in range(40500, 40504):
            name = f'mnist_common_e1_{arm}_unscaled_support_seed{seed}'
            src = logs/'paper_doubly_sender'/name/'sender_decision/epoch020/valid/native_interventions.csv'
            info_path=src.with_name('manifest.json')
            info=json.loads(info_path.read_text(encoding='utf-8'))
            validate_run_manifest(info)
            frame = pd.read_csv(src)
            frame = frame.assign(arm=arm, seed=seed, epoch=20, phase='valid', support_normalization='none')
            frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[(raw.epoch == 20) & (raw.phase == 'valid') &
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
        per_seed.append(dict(arm=arm, seed=seed, epoch=20, phase='valid', support_normalization='none', variant=variant,
            accuracy_percent=100*frame.correct.mean(), no_decision_percent=100*frame.no_decision.mean(), sample_count=200))
    per_seed = pd.DataFrame(per_seed)
    summary = []
    for (arm, variant), frame in per_seed.groupby(['arm', 'variant']):
        values=frame.accuracy_percent.to_numpy(); m=values.mean()
        ci=t.ppf(.975,3)*values.std(ddof=1)/np.sqrt(4)
        nd=frame.no_decision_percent.to_numpy();ndm=nd.mean();ndci=t.ppf(.975,3)*nd.std(ddof=1)/np.sqrt(4)
        summary.append(dict(arm=arm,epoch=20,phase='valid',support_normalization='none',variant=variant,n_training_seeds=4,
            accuracy_percent_mean=m,accuracy_percent_ci95_low=m-ci,accuracy_percent_ci95_high=m+ci,
            no_decision_percent_mean=ndm,no_decision_percent_ci95_low=ndm-ndci,no_decision_percent_ci95_high=ndm+ndci))
    summary=pd.DataFrame(summary)
    print(summary.pivot(index='variant',columns='arm',values='accuracy_percent_mean').to_string())
    return summary



COLORS=['#6C91B0','#D99259','#909090','#648B70','#AB7AA2']
STRUCTURE=['post_every','none','pre_every','alternating','simultaneous']
LAYERS=['post_every','hidden_only','alternating','all_layers']

def derive(decisions,activity,weight,singular_values=None):
    n=int(activity['sample_count']);count=activity['root_receiver_first_spike_count'].astype(float)
    assert n==len(decisions['labels']) and n>0
    assert np.all((count>=0)&(count<=n))
    rates=count/n;w=weight['root_branch_weight'].astype(float)
    assert w.ndim==2 and w.shape[1]==len(rates)
    sv=np.linalg.svd(w,compute_uv=False) if singular_values is None else singular_values
    p=sv/sv.sum() if sv.sum()>0 else np.zeros_like(sv)
    rank=float(np.exp(-(p[p>0]*np.log(p[p>0])).sum())) if sv.sum()>0 else 0.
    energy=sv**2
    pr=float(energy.sum()**2/(energy**2).sum()) if energy.sum()>0 else 0.
    metrics=dict(accuracy=100*float(decisions['accuracy']),no_decision=100*(1-decisions['has_decision'].mean()),
                 correct_earliness=100*decisions['T_correct'].mean(),wrong_earliness=100*decisions['T_wrong'].mean(),
                 gap=100*decisions['T_gap'].mean(),effective_rank=rank,participation_ratio=pr,
                 dead_percent=100*(rates==0).mean(),activity_variance=rates.var(),mean_spikes=count.sum()/n,sample_count=n)
    if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite result')
    return {k:float(v) for k,v in metrics.items()},rates,sv


def _scalar_at_epoch(events, tag, epoch):
    matches = [event.value for event in events.Scalars(tag) if event.step == epoch]
    if not matches:
        raise ValueError(f'Missing TensorBoard scalar {tag} at epoch {epoch}')
    return float(matches[-1])


def derive_from_logged_summary(folder, activity, weight, epoch, singular_values=None):
    """Read the formal Prefect schema used before native_decisions was added."""
    events = EventAccumulator(str(folder), size_guidance={'scalars': 0})
    events.Reload()
    scalar = lambda tag: _scalar_at_epoch(events, tag, epoch)
    prefix = 'per_neuron_active_fraction/hidden/test/A'
    quantiles = np.asarray([scalar(f'{prefix}/q{q:02d}') for q in range(0, 101, 10)])
    weight_array = weight['root_branch_weight'].astype(float)
    receiver_count = weight_array.shape[1]
    rates = np.interp(
        np.linspace(0., 1., receiver_count),
        np.linspace(0., 1., len(quantiles)),
        quantiles,
    )
    sv = np.linalg.svd(weight_array, compute_uv=False) if singular_values is None else singular_values
    p = sv / sv.sum() if sv.sum() > 0 else np.zeros_like(sv)
    rank = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())) if sv.sum() > 0 else 0.
    energy = sv ** 2
    pr = float(energy.sum() ** 2 / (energy ** 2).sum()) if energy.sum() > 0 else 0.
    mean_activity = scalar(f'{prefix}/mean')
    activity_std = scalar(f'{prefix}/std')
    correct = scalar('hinge/mean_T_correct/test')
    wrong = scalar('hinge/mean_T_wrong/test')
    metrics = dict(
        accuracy=100 * scalar('accuracy/top_spike/test'),
        no_decision=100 * scalar('decision_distribution/predicted_fraction/test/no_decision'),
        correct_earliness=100 * correct,
        wrong_earliness=100 * wrong,
        gap=100 * (correct - wrong),
        effective_rank=rank,
        participation_ratio=pr,
        dead_percent=100 * scalar(f'{prefix}/eq_zero'),
        activity_variance=activity_std ** 2,
        mean_spikes=mean_activity * receiver_count,
        sample_count=int(activity['sample_count']),
    )
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError(f'Nonfinite logged summary in {folder}')
    return {key: float(value) for key, value in metrics.items()}, rates, sv

def build(out,logs):
    spec=json.loads((HERE/'coverage.json').read_text());records=[];curves={};spectra={}
    for dataset,ds in spec['datasets'].items():
        for row in settings(dataset):
            condition=row['sub_exp_name'].rsplit('_seed',1)[0][len(dataset)+1:]
            folder=logs/row['experiment_name']/row['sub_exp_name']
            marker=folder/'reproduction_complete.json'
            expected=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
            if json.loads(marker.read_text())['settings_sha256']!=expected:raise ValueError(f'Settings mismatch: {folder}')
            e=ds['final_epoch'];art=folder/'mechanism_artifacts'
            decision_path=art/f'native_decisions_test_epoch{e:03d}.npz'
            activity_path=art/f'temporal_contribution_test_epoch{e:03d}.npz'
            weight_path=art/f'root_weight_epoch{e:03d}.npz'
            with np.load(activity_path,allow_pickle=False) as f:activity=dict(f)
            with np.load(weight_path,allow_pickle=False) as f:weight=dict(f)
            key=hashlib.sha256(weight_path.read_bytes()).hexdigest()
            if decision_path.is_file() and 'root_receiver_first_spike_count' in activity:
                with np.load(decision_path,allow_pickle=False) as f:decisions=dict(f)
                values,rates,sv=derive(decisions,activity,weight,singular_values=spectra.get(key))
            else:
                values,rates,sv=derive_from_logged_summary(
                    folder,activity,weight,e,singular_values=spectra.get(key))
            spectra[key]=sv
            records.append(dict(dataset=dataset,condition=condition,seed=row['seed'],epoch=e,phase='test',**values))
            curves.setdefault((dataset,condition),[]).append((rates,sv))
    out.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(records)
    grouped=frame.groupby(['dataset','condition']);mean=grouped.mean(numeric_only=True);sd=grouped.std(numeric_only=True,ddof=1)
    labels={c['id']:c['label'] for c in spec['conditions']};schedule=[c['id'] for c in spec['conditions'][:8]];tables={}
    for number,conditions in [(1,schedule),(2,LAYERS)]:
        rows=[]
        for c in conditions:
            r={'method':labels[c]}
            if number==1:r['schedule']=next(x['schedule'] for x in spec['conditions'] if x['id']==c)
            for ds in ['mnist','nmnist']:
                v=mean.loc[(ds,c),'accuracy'];r[ds]=f'{v:.3f} ± {sd.loc[(ds,c),"accuracy"]:.3f}'
                if number==2 and c!='post_every':r[ds]+=f' ({v-mean.loc[(ds,"post_every"),"accuracy"]:+.3f})'
            rows.append(r)
        tables[number]=pd.DataFrame(rows)
    metrics=['effective_rank','participation_ratio','dead_percent','activity_variance','mean_spikes']
    tables[3]=pd.DataFrame([{'method':labels[c],**{k:f'{mean.loc[("mnist",c),k]:.4f} ± {sd.loc[("mnist",c),k]:.4f}' for k in metrics}} for c in STRUCTURE])
    rows=[]
    for k in ['accuracy','no_decision','correct_earliness','wrong_earliness','gap']:
        a=mean.loc[('nmnist','post_every'),k];b=mean.loc[('nmnist','alternating'),k];suffix='' if k=='accuracy' else '%'
        rows.append({'metric':k,'Post-only':f'{a:.3f}{suffix}','Doubly':f'{b:.3f}{suffix} ({b-a:+.3f}{suffix})'})
    tables[4]=pd.DataFrame(rows)
    for num,table in tables.items():
        table.to_csv(out/f'table{num}.csv',index=False)
    plt.rcParams.update({'font.family':'serif','font.size':8,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,1,figsize=(3.4,4.2))
    for c,color in zip(STRUCTURE,[COLORS[0],COLORS[2],COLORS[3],COLORS[1],COLORS[4]]):
        pairs=curves[('mnist',c)]
        for ax,index in zip(axes,[0,1]):
            data=np.stack([np.sort(p[index])[::-1] for p in pairs])
            ax.plot(np.arange(1,data.shape[1]+1),data.mean(0),label=labels[c],color=color)
    axes[0].set(xlabel='Receiver rank',ylabel='First-spike probability');axes[0].legend(fontsize=6)
    axes[1].set(xlabel='Singular-value rank',ylabel='Singular value');axes[1].set_yscale('symlog',linthresh=1e-4)
    fig.tight_layout();save(fig,out/'figure1')
    fig,axes=plt.subplots(1,2,figsize=(3.4,2.35))
    limits=[(.35,.55),(.05,.25)];ticksets=[[.4,.5],[.1,.2]]
    evidence=[[mean.loc[('nmnist',c),k]/100 for c in ['post_every','alternating']] for k in ['correct_earliness','wrong_earliness']]
    if any(not all(lo<=v<=hi for v in values) for values,(lo,hi) in zip(evidence,limits)):
        span=max(.2,max(max(v)-min(v) for v in evidence)+.08)
        limits=[(min(v)-.04,min(v)-.04+span) for v in evidence]
        ticksets=[np.linspace(lo,hi,3) for lo,hi in limits]
    for ax,k,ylim,ticks in zip(axes,['correct_earliness','wrong_earliness'],limits,ticksets):
        v=[mean.loc[('nmnist',c),k]/100 for c in ['post_every','alternating']]
        ax.bar([0,1],v,color=COLORS[:2],width=.55)
        ax.set(ylim=ylim,yticks=ticks,xticks=[0,1],xticklabels=['Post-only','Doubly'],title=k.replace('_',' ').capitalize())
        ax.tick_params(axis='x',labelrotation=30)
        for x,y in enumerate(v):ax.annotate(f'{y:.4f}',(x,y),xytext=(0,3),textcoords='offset points',ha='center',fontsize=7)
    fig.tight_layout();save(fig,out/'figure2')
    return frame

def save(fig,path):
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)

def sender_column(table,out):
    fig,axes=plt.subplots(2,1,figsize=(3.4,4.1))
    variants=['random_support','equal_drive_attenuation','top_support']
    for ax,arm,title,color in zip(axes,['adaptive_post','adaptive_pre'],['Post-only','Doubly'],COLORS):
        frame=table[table.arm==arm].set_index('variant');baseline=frame.loc['identity','accuracy_percent_mean']
        values=frame.loc[variants,'accuracy_percent_mean'].to_numpy()
        ax.bar(range(3),values,color=color,width=.55)
        ax.axhline(baseline,color='#444444',linestyle='--',linewidth=.8,label=f'Original {baseline:.2f}%')
        ax.set(ylim=(0,115),yticks=[0,50,100],ylabel='Accuracy (%)',title=title,
               xticks=[0,1,2],xticklabels=['Random\nremoval','Uniform\nattenuation','Top-sender\nremoval'])
        ax.legend(fontsize=7,loc='upper right')
        for x,y in enumerate(values):ax.text(x,y+2,f'{y:.2f}',ha='center',fontsize=7)
    fig.tight_layout();save(fig,out/'figure3')

def generate_results(sender_only=False):
    out=ROOT/'result'
    if not sender_only and out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)
    if not sender_only:
        build(out,ROOT/'logs')
    for row in settings('sender'):
        marker=ROOT/'logs'/row['experiment_name']/row['sub_exp_name']/'reproduction_complete.json'
        expected=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
        if json.loads(marker.read_text())['settings_sha256']!=expected:
            raise ValueError(f'Sender settings mismatch: {marker}')
    sender_summary=build_sender()
    sender_column(sender_summary,out)
    if not sender_only:
        expected={f'figure{i}.pdf' for i in range(1,4)}|{f'table{i}.csv' for i in range(1,5)}
        actual={path.name for path in out.iterdir() if path.is_file()}
        if actual != expected:
            raise RuntimeError(f'Unexpected result files: expected {sorted(expected)}, got {sorted(actual)}')
    print('Results:',out)
