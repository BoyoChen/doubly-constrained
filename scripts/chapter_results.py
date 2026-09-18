"""Build four tables and three figures from new completed runs, without old-data fallback."""
import argparse, hashlib, json, re, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_suite import settings, ROOT, HERE

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

def build(label,out,logs):
    spec=json.loads((HERE/'coverage.json').read_text());sources=[];records=[];curves={};spectra={}
    for dataset,ds in spec['datasets'].items():
        for row in settings(dataset,label):
            condition=row['sub_exp_name'].rsplit('_seed',1)[0][len(dataset)+1:]
            folder=logs/row['experiment_name']/row['sub_exp_name']/label
            marker=folder/'reproduction_complete.json'
            expected=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
            if json.loads(marker.read_text())['settings_sha256']!=expected:raise ValueError(f'Settings mismatch: {folder}')
            e=ds['final_epoch'];art=folder/'mechanism_artifacts'
            paths=[art/f'native_decisions_test_epoch{e:03d}.npz',art/f'temporal_contribution_test_epoch{e:03d}.npz',art/f'root_weight_epoch{e:03d}.npz']
            arrays=[]
            for path in paths:
                with np.load(path,allow_pickle=False) as f:arrays.append(dict(f))
                sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            key=sources[-1]['sha256']
            values,rates,sv=derive(*arrays,singular_values=spectra.get(key))
            spectra[key]=sv
            records.append(dict(dataset=dataset,condition=condition,seed=row['seed'],epoch=e,phase='test',**values))
            curves.setdefault((dataset,condition),[]).append((rates,sv))
    out.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(records);frame.to_csv(out/'all_metrics_per_seed.csv',index=False)
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
        rendered=table.copy()
        for col in rendered:
            rendered[col]=rendered[col].map(lambda v:str(v).replace('_',r'\_').replace('%',r'\%').replace('±',r'$\pm$'))
        for ds in ['mnist','nmnist']:
            if num==1:
                best=max(mean.loc[(ds,c),'accuracy'] for c in schedule)
                for i,c in enumerate(schedule):
                    if mean.loc[(ds,c),'accuracy']==best:rendered.loc[i,ds]=r'\textbf{'+rendered.loc[i,ds]+'}'
            elif num==2:
                for i,c in enumerate(LAYERS[1:],1):
                    delta=mean.loc[(ds,c),'accuracy']-mean.loc[(ds,'post_every'),'accuracy']
                    color='green!45!black' if delta>0 else 'red!70!black' if delta<0 else 'black'
                    rendered.loc[i,ds]=re.sub(r'(\([+-][0-9.]+\))',lambda m:r'\textcolor{'+color+'}{'+m[0]+'}',rendered.loc[i,ds])
        tex=rendered.to_latex(index=False,escape=False)
        tex=r'\resizebox{\columnwidth}{!}{%'+ '\n'+tex+'}\n'
        (out/f'table{num}.tex').write_text(tex,encoding='utf8')
    plt.rcParams.update({'font.family':'serif','font.size':8,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,1,figsize=(3.4,4.2));curve_rows=[]
    for c,color in zip(STRUCTURE,[COLORS[0],COLORS[2],COLORS[3],COLORS[1],COLORS[4]]):
        pairs=curves[('mnist',c)]
        for seed,(rates,sv) in zip(spec['datasets']['mnist']['seeds'],pairs):
            for k,values in [('activity',np.sort(rates)[::-1]),('singular_value',sv)]:
                curve_rows.extend(dict(condition=c,seed=seed,quantity=k,rank=i+1,value=float(v)) for i,v in enumerate(values))
        for ax,index in zip(axes,[0,1]):
            data=np.stack([np.sort(p[index])[::-1] for p in pairs])
            ax.plot(np.arange(1,data.shape[1]+1),data.mean(0),label=labels[c],color=color)
    axes[0].set(xlabel='Receiver rank',ylabel='First-spike probability');axes[0].legend(fontsize=6)
    axes[1].set(xlabel='Singular-value rank',ylabel='Singular value');axes[1].set_yscale('symlog',linthresh=1e-4)
    fig.tight_layout();save(fig,out/'figure1');pd.DataFrame(curve_rows).to_csv(out/'figure1_per_seed.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(3.4,2.35))
    limits=[(.35,.55),(.05,.25)];ticksets=[[.4,.5],[.1,.2]]
    evidence=[[mean.loc[('nmnist',c),k]/100 for c in ['post_every','alternating']] for k in ['correct_earliness','wrong_earliness']]
    if any(not all(lo<=v<=hi for v in values) for values,(lo,hi) in zip(evidence,limits)):
        span=max(.2,max(max(v)-min(v) for v in evidence)+.08)
        limits=[(min(v)-.04,min(v)-.04+span) for v in evidence]
        ticksets=[np.linspace(lo,hi,3) for lo,hi in limits]
    (out/'figure2_axes.json').write_text(json.dumps({'limits':limits,'equal_span':True,'changed_from_preview':limits!=[(.35,.55),(.05,.25)]},indent=2))
    for ax,k,ylim,ticks in zip(axes,['correct_earliness','wrong_earliness'],limits,ticksets):
        v=[mean.loc[('nmnist',c),k]/100 for c in ['post_every','alternating']]
        ax.bar([0,1],v,color=COLORS[:2],width=.55)
        ax.set(ylim=ylim,yticks=ticks,xticks=[0,1],xticklabels=['Post-only','Doubly'],title=k.replace('_',' ').capitalize())
        ax.tick_params(axis='x',labelrotation=30)
        for x,y in enumerate(v):ax.annotate(f'{y:.4f}',(x,y),xytext=(0,3),textcoords='offset points',ha='center',fontsize=7)
    fig.tight_layout();save(fig,out/'figure2')
    (out/'sources.json').write_text(json.dumps(sources,indent=2))
    def markdown(table):
        rows=[list(table.columns),['---']*len(table.columns),*table.astype(str).values.tolist()]
        return '\n'.join('| '+' | '.join(row)+' |' for row in rows)
    (out/'tables.md').write_text('\n\n'.join(f'Table {i}\n\n'+markdown(t) for i,t in tables.items()),encoding='utf8')
    return frame

def save(fig,path):
    for ext in ['pdf','svg','png']:fig.savefig(path.with_suffix('.'+ext),bbox_inches='tight',dpi=220)
    plt.close(fig)

def sender_column(source,out):
    table=pd.read_csv(source/'summary.csv')
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

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',default='paper-v3');a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.label):p.error('Invalid label')
    out=ROOT/'output/doubly_reproduction'/a.label/'chapter'
    complete=out/'COMPLETE.json'
    if complete.exists():complete.unlink()
    build(a.label,out,ROOT/'logs')
    for row in settings('sender',a.label):
        marker=ROOT/'logs'/row['experiment_name']/row['sub_exp_name']/a.label/'reproduction_complete.json'
        expected=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
        if json.loads(marker.read_text())['settings_sha256']!=expected:raise ValueError(f'Sender settings mismatch: {marker}')
    subprocess.run([sys.executable,str(HERE/'sender_results.py'),'--label',a.label,'--output',str(out/'sender')],check=True)
    sender_column(out/'sender',out)
    complete.write_text(json.dumps({'label':a.label,'tables':4,'figures':3,'historical_fallback':False},indent=2))
    print('Complete chapter assets:',out)
if __name__=='__main__':main()
