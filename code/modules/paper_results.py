"""Chapter tables, figures and sender summaries from completed experiment runs."""
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
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



# STIXGeneral ships with matplotlib and is Times-metric-compatible, so the
# figures match the paper's body text and render identically off this machine.
FONT={'font.family':'STIXGeneral','mathtext.fontset':'stix','pdf.fonttype':42,
      'font.size':7,'axes.labelsize':7.2}
COLORS=['#6C91B0','#D99259','#909090','#648B70','#AB7AA2']
# Figure 1 uses a separately validated set: all-pairs CVD dE 9.2,
# normal-vision 24.0 on the light surface.
# Its neutral is the reference condition and deliberately carries no chroma.
BLUE,ORANGE,AQUA,NEUTRAL='#2a78d6','#eb6834','#1baf7a','#5f5f5f'
DASH=(0,(4.5,1.8))
# Figure 1: the four every-batch schedules, in the order Table 1 lists them.
FIGURE1=[('none_all_layers','No constraint',NEUTRAL,'-',1.5),
         ('post_every','Post only',BLUE,'-',1.3),
         ('pre_all_layers','Pre only',AQUA,DASH,1.4),
         ('all_layers','Doubly',ORANGE,'-',1.6)]
# Table 1: every schedule, applied at both layers.
SCHEDULE=['none_all_layers','post_every','post_half_all_layers',
          'pre_all_layers','pre_half_all_layers','all_layers']
# Table 2: the hidden x output placement square.
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

def figure1(out,curves,units,mean):
    """(a) how the hidden units distribute their firing rate, which is what post
    normalization controls; (b) how the output weight matrix distributes its
    singular-value mass, which is what pre normalization controls."""
    ink,muted,grid='#1a1a1a','#6b6b6b','#dcdcdc'
    def style(ax):
        for side in ('top','right'):ax.spines[side].set_visible(False)
        for side in ('left','bottom'):
            ax.spines[side].set_linewidth(.5);ax.spines[side].set_color(muted)
        ax.tick_params(width=.5,length=2.2,colors=muted,labelsize=6.2)
        for label in ax.get_xticklabels()+ax.get_yticklabels():label.set_color(ink)
        ax.set_axisbelow(True)
    pooled=np.concatenate([np.concatenate(units[('mnist',c)]) for c,*_ in FIGURE1])
    bins=np.linspace(0.,float(np.quantile(pooled,.999)),26)
    heat=[]
    for c,*_ in FIGURE1:
        v=np.concatenate(units[('mnist',c)])
        counts,_=np.histogram(np.clip(v,bins[0],np.nextafter(bins[-1],bins[0])),bins=bins)
        heat.append(counts/counts.sum()*100)
    heat=np.asarray(heat)
    fig=plt.figure(figsize=(3.4,3.3))
    ax_a=fig.add_axes([.250,.655,.585,.285]);ax_c=fig.add_axes([.850,.655,.024,.285])
    ax_b=fig.add_axes([.250,.130,.710,.365])
    mesh=ax_a.pcolormesh(bins,np.arange(len(FIGURE1)+1),np.ma.masked_less_equal(heat,0.),
                         cmap='YlGnBu',norm=mcolors.LogNorm(vmin=.2,vmax=float(heat.max())),
                         edgecolors='white',linewidth=.2)
    ax_a.set_yticks(np.arange(len(FIGURE1))+.5)
    ax_a.set_yticklabels([label for _,label,*_ in FIGURE1],fontsize=6.5)
    ax_a.invert_yaxis();ax_a.set_xlim(bins[0],bins[-1])
    ax_a.set_xticks(np.linspace(0,bins[-1],4))
    ax_a.set_xticklabels([f'{v:.2f}' for v in np.linspace(0,bins[-1],4)])
    ax_a.set_xlabel('Mean firing rate per hidden unit',labelpad=1.5)
    style(ax_a);ax_a.spines['left'].set_visible(False);ax_a.tick_params(left=False)
    cbar=fig.colorbar(mesh,cax=ax_c)
    cbar.set_label('Units per bin (%)',fontsize=6.,labelpad=1.5,color=ink)
    cbar.ax.tick_params(labelsize=5.6,width=.4,length=1.8,colors=muted)
    cbar.outline.set_linewidth(.4);cbar.outline.set_edgecolor(muted)
    handles=[]
    for c,label,color,dash,width in FIGURE1:
        sv=np.stack([np.sort(pair[1])[::-1] for pair in curves[('mnist',c)]]).mean(0)
        line,=ax_b.plot(np.arange(1,len(sv)+1),np.cumsum(sv)/sv.sum(),color=color,
                        linestyle=dash,linewidth=width,solid_capstyle='round',
                        dash_capstyle='round')
        handles.append((line,f"{label}  (rank {mean.loc[('mnist',c),'effective_rank']:.0f})"))
    style(ax_b);ax_b.grid(True,color=grid,linewidth=.35)
    ax_b.set_xscale('log');ax_b.set_xlim(1,len(sv));ax_b.minorticks_off()
    ticks=[x for x in (1,2,5,10,20,50,100,200) if x<=len(sv)]
    ax_b.set_xticks(ticks);ax_b.set_xticklabels([str(x) for x in ticks])
    ax_b.set_ylim(0,1.02);ax_b.set_yticks([0,.25,.5,.75,1.])
    ax_b.set_xlabel('Singular-value rank',labelpad=1.5)
    ax_b.set_ylabel('Cumulative share of' + chr(10) + 'singular-value mass',labelpad=3)
    ax_b.legend([h for h,_ in handles],[l for _,l in handles],loc='upper left',
                bbox_to_anchor=(-.02,1.04),ncol=2,frameon=False,fontsize=6.,
                handlelength=2.,columnspacing=.6,handletextpad=.35,labelspacing=.2,
                labelcolor=ink)
    fig.text(.012,.945,'(a)',fontsize=7.5,color=ink,weight='bold')
    fig.text(.012,.505,'(b)',fontsize=7.5,color=ink,weight='bold')
    fig.savefig((out/'figure1').with_suffix('.pdf'));plt.close(fig)


def build(out,logs):
    spec=json.loads((HERE/'coverage.json').read_text());records=[];curves={};spectra={};units={}
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
            # Real per-unit means over the test set, so the mass at exactly
            # zero survives the binning in Figure 1(a).
            units.setdefault((dataset,condition),[]).append(
                activity['sender_activity_mean'].astype(float))
    out.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(records)
    grouped=frame.groupby(['dataset','condition']);mean=grouped.mean(numeric_only=True);sd=grouped.std(numeric_only=True,ddof=1)
    info={c['id']:c for c in spec['conditions']};tables={}
    cell=lambda ds,c:f'{mean.loc[(ds,c),"accuracy"]:.3f} ± {sd.loc[(ds,c),"accuracy"]:.3f}'
    delta=lambda ds,c:f' ({mean.loc[(ds,c),"accuracy"]-mean.loc[(ds,"post_every"),"accuracy"]:+.3f})'
    tables[1]=pd.DataFrame([{'method':info[c]['label'],'schedule':info[c]['schedule'],
                             **{ds:cell(ds,c) for ds in ['mnist','nmnist']}} for c in SCHEDULE])
    tables[2]=pd.DataFrame([{'hidden':info[c]['hidden'],'output':info[c]['output'],
                             **{ds:cell(ds,c)+('' if c=='post_every' else delta(ds,c))
                                for ds in ['mnist','nmnist']}} for c in LAYERS])
    rows=[]
    for k in ['accuracy','no_decision','correct_earliness','wrong_earliness','gap']:
        a_=mean.loc[('nmnist','post_every'),k];b_=mean.loc[('nmnist','alternating'),k];suffix='' if k=='accuracy' else '%'
        rows.append({'metric':k,'Post-only':f'{a_:.3f}{suffix}','Doubly':f'{b_:.3f}{suffix} ({b_-a_:+.3f}{suffix})'})
    tables[3]=pd.DataFrame(rows)
    for num,table in tables.items():
        table.to_csv(out/f'table{num}.csv',index=False)
    plt.rcParams.update(FONT)
    figure1(out,curves,units,mean)
    # Paired slope chart: one line per seed, so the reader sees every paired
    # difference rather than two bars on a truncated axis.
    fig,axes=plt.subplots(1,2,figsize=(3.4,1.75))
    ev=frame[frame.dataset=='nmnist'].set_index(['condition','seed'])
    seeds=sorted(frame[(frame.dataset=='nmnist')&(frame.condition=='post_every')].seed)
    pairs=[([ev.loc[('post_every',s),k]/100 for s in seeds],
            [ev.loc[('alternating',s),k]/100 for s in seeds])
           for k in ['correct_earliness','wrong_earliness']]
    # One round span for both panels, so the two slopes are directly comparable
    # and both axes read off round numbers (e.g. 0.45-0.55 against 0.05-0.15).
    NICE=[.005,.01,.02,.025,.05,.1,.2,.25,.5,1.]
    SPAN=next(s for s in NICE if s>=max(max(x+y)-min(x+y) for x,y in pairs)*1.25)
    STEP=SPAN/2
    DEC=max(0,-int(np.floor(np.log10(STEP))))
    limits=[]
    for a,b in pairs:
        lo=np.floor(min(a+b)/STEP)*STEP
        while lo+SPAN<max(a+b):lo+=STEP
        limits.append((lo,lo+SPAN))
    for ax,(a,b),(lo,hi),title in zip(axes,pairs,limits,
                                      ['Correct earliness','Wrong earliness']):
        for lo_,hi_ in zip(a,b):
            ax.plot([0,1],[lo_,hi_],color='#b8b8b8',linewidth=.7,zorder=1,
                    solid_capstyle='round')
        ax.scatter([0]*len(a),a,s=13,color=BLUE,zorder=3,linewidths=0)
        ax.scatter([1]*len(b),b,s=13,color=ORANGE,zorder=3,linewidths=0)
        for x,vals,col,off in ((0,a,BLUE,-11),(1,b,ORANGE,6)):
            m=float(np.mean(vals))
            ax.plot([x-.22,x+.22],[m,m],color=col,linewidth=1.5,zorder=2,
                    solid_capstyle='butt')
            ax.annotate(f'{m:.4f}',(x,m),xytext=(0,off),textcoords='offset points',
                        ha='center',fontsize=6.3,color=col)
        ticks=np.round(np.arange(lo,hi+STEP/2,STEP),DEC+1)
        ax.set(xlim=(-.5,1.5),ylim=(lo,hi),xticks=[0,1],
               xticklabels=['Post-only','Doubly'],title=title)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{v:.{DEC}f}' for v in ticks])
        for side in ('top','right'):ax.spines[side].set_visible(False)
        for side in ('left','bottom'):
            ax.spines[side].set_linewidth(.5);ax.spines[side].set_color('#6b6b6b')
        ax.tick_params(width=.5,length=2.2,labelsize=6.2)
    fig.tight_layout();save(fig,out/'figure2')
    return frame

def save(fig,path):
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)

def sender_column(table,out):
    plt.rcParams.update(FONT)
    fig,axes=plt.subplots(2,1,figsize=(3.4,4.1))
    variants=['random_support','equal_drive_attenuation','top_support']
    for ax,arm,title,color in zip(axes,['adaptive_post','adaptive_pre'],['Post-only','Doubly'],COLORS):
        frame=table[table.arm==arm].set_index('variant');baseline=frame.loc['identity','accuracy_percent_mean']
        values=frame.loc[variants,'accuracy_percent_mean'].to_numpy()
        ax.bar(range(3),values,color=color,width=.55)
        ax.axhline(baseline,color='#444444',linestyle='--',linewidth=.8,label=f'Original {baseline:.2f}%')
        ax.set(ylim=(0,115),yticks=[0,50,100],ylabel='Accuracy (%)',title=title,
               xticks=[0,1,2],xticklabels=['Random\nremoval','Uniform\nattenuation','Top-sender\nremoval'])
        ax.annotate(f'Original {baseline:.2f}%',xy=(1,baseline),xycoords=('axes fraction','data'),
                    xytext=(-3,3),textcoords='offset points',ha='right',va='bottom',
                    fontsize=6.5,color='#444444')
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
        expected={f'figure{i}.pdf' for i in range(1,4)}|{f'table{i}.csv' for i in range(1,4)}
        actual={path.name for path in out.iterdir() if path.is_file()}
        if actual != expected:
            raise RuntimeError(f'Unexpected result files: expected {sorted(expected)}, got {sorted(actual)}')
    print('Results:',out)
