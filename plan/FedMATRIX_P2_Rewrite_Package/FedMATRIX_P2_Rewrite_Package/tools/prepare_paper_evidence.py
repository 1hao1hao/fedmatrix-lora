#!/usr/bin/env python3
"""Offline manuscript evidence preparation. Never launches training.

Reads the published P2/B CSVs and the local original P/A pools. Writes new
presentation files only; original data are never overwritten.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DISTS = ('iid', 'dirichlet:1.0', 'dirichlet:0.1')
STAGES = ('early', 'middle')
SEEDS = (2024, 2025, 2026)
METHODS = ('Full', 'Server-Proxy', 'Layer-Level', 'No-Depth')
LABELS = {'iid': 'IID', 'dirichlet:1.0': r'Dir. $1.0$', 'dirichlet:0.1': r'Dir. $0.1$'}
STAGE_LABELS = {'early': 'Initial', 'middle': 'Warm-up'}

def mask(s: str) -> tuple[int, ...]:
    value = tuple(sorted(int(x) for x in json.loads(s)))
    if len(value) != 12 or len(set(value)) != 12:
        raise ValueError('Expected 12 distinct target indices')
    return value

def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')

def cellkey(row) -> tuple:
    return (str(row['distribution']), str(row['checkpoint']),
            int(row['seed']), int(row['client_id']))

def mean_sd(x) -> tuple[float, float]:
    a = np.asarray(list(x), dtype=float)
    if len(a) != 3 or not np.isfinite(a).all():
        raise ValueError('Exactly three finite seed-level values are required')
    return float(a.mean()), float(a.std(ddof=1))

def texnum(mu: float, sd: float, decimals: int = 3, signed: bool = False) -> str:
    fmt = ('+' if signed else '') + '.' + str(decimals) + 'f'
    return '$' + format(mu, fmt) + r'\pm' + format(sd, '.' + str(decimals) + 'f') + '$'

def prep_p2(root: Path, gen: Path) -> tuple[pd.DataFrame, dict]:
    p = root / 'paper_assets/05_figures/experiment_p2_20261027'
    data = pd.read_csv(p / 'all_results.csv')
    expected = {'Local-Matrix', 'Local-Layer', 'Proxy-Matrix', 'Proxy-Layer'}
    if len(data) != 360 or set(data.method) != expected or set(data.status) != {'feasible'}:
        raise ValueError('P2 no longer matches the locked 360-record protocol')
    for name, expected_value in [('matrix_count',12),('rank',8),('train_steps',20)]:
        if not (pd.to_numeric(data[name]) == expected_value).all():
            raise ValueError('P2 protocol mismatch: ' + name)
    if set(data.seed) != set(SEEDS) or set(data.distribution) != set(DISTS):
        raise ValueError('P2 seeds/partitions mismatch')
    bymask = {}
    pairs = []
    for (dist, stage, seed, client), part in data.groupby(['distribution','checkpoint','seed','client_id']):
        if len(part) != 4 or part.trainable_parameters.nunique() != 1:
            raise ValueError('Unequal parameter counts or incomplete P2 cell')
        if np.ptp(part.val_loss_before.to_numpy(float)) > 1e-6:
            raise ValueError('P2 starting losses are not matched')
        if not np.isfinite(part[['val_loss_before','val_loss_after','val_loss_reduction']].to_numpy(float)).all():
            raise ValueError('Nonfinite P2 outcome')
        if not np.allclose(part.val_loss_before-part.val_loss_after,part.val_loss_reduction,atol=1e-8,rtol=0):
            raise ValueError('Inconsistent P2 loss reduction')
        for _, row in part.iterrows():
            key = cellkey(row) + (mask(row['mask']),)
            vals = tuple(float(row[n]) for n in ('val_loss_before','val_loss_after','val_loss_reduction'))
            if key in bymask and not np.allclose(bymask[key], vals, atol=1e-8, rtol=0):
                raise ValueError('Conflicting outcomes for a duplicate P2 mask')
            bymask[key] = vals
        gains = part.set_index('method').val_loss_reduction
        for source in ('Local','Proxy'):
            pairs.append({'distribution':dist,'checkpoint':stage,'seed':int(seed),'client_id':int(client),
                          'source':source,'difference':float(gains[source+'-Matrix']-gains[source+'-Layer'])})
    pairs = pd.DataFrame(pairs)
    seedmeans = pairs.groupby(['distribution','checkpoint','source','seed'],as_index=False).difference.mean()
    if len(seedmeans) != 36 or int((seedmeans.difference > 0).sum()) != 35 or int((pairs.difference > 0).sum()) != 161:
        raise ValueError('P2 aggregate counts differ from the locked manuscript text; report rather than rewrite claims')
    summary = pd.read_csv(p / 'paired_summary.csv')
    outrows = []
    tex = [r'\begin{table*}[t]',r'\centering',
           r'\caption{Complete matrix-minus-layer differences in 20-step held-out validation-loss reduction. Entries are mean $\pm$ sample standard deviation over three within-seed client means.}',
           r'\label{tab:fm-p2-complete}',r'\small',r'\begin{tabular}{@{}llcc@{}}',r'\toprule',
           r'Partition & Checkpoint & Local scores & Proxy scores \\',r'\midrule']
    for dist in DISTS:
        for stage in STAGES:
            stats = []
            for source in ('Local','Proxy'):
                s = seedmeans[(seedmeans.distribution==dist)&(seedmeans.checkpoint==stage)&(seedmeans.source==source)]
                mu, sd = mean_sd(s.difference)
                old = summary[(summary.distribution==dist)&(summary.checkpoint==stage)&(summary.source==source)]
                if len(old)!=1 or abs(mu-float(old.iloc[0].mean_matrix_minus_layer))>1e-8:
                    raise ValueError('P2 published summary does not reproduce from all_results')
                stats.append(texnum(mu,sd,5,True))
                outrows.append({'distribution':dist,'checkpoint':stage,'source':source,'mean':mu,'std':sd})
            tex.append(f'{LABELS[dist]} & {STAGE_LABELS[stage]} & {stats[0]} & {stats[1]} '+r'\\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    write(gen/'fm_p2_complete_table.tex','\n'.join(tex)+'\n')
    pairs.to_csv(gen/'p2_verified_client_pairs.csv',index=False)
    seedmeans.to_csv(gen/'p2_verified_seed_pairs.csv',index=False)
    return data,bymask

def prep_b(root: Path, gen: Path, fig: Path) -> None:
    p = root/'paper_assets/05_figures/experiment_20261025/P_A_B_review'
    curves=pd.read_csv(p/'B_round_curves.csv')
    published=pd.read_csv(p/'B_three_seed_summary.csv').set_index('method')
    expected={'Full':(.906,.9086666666666666), 'Server-Proxy':(.9046666666666666,.908),
              'Layer-Level':(.9006666666666666,.9073333333333333), 'No-Depth':(.9046666666666666,.9093333333333334)}
    fig.mkdir(parents=True,exist_ok=True)
    plot,ax=plt.subplots(figsize=(3.4,2.35))
    styles=[('black','-',1.6),('0.25','--',1.1),('0.5','-.',1.1),('0.65',':',1.4)]
    grid=None
    for method,(color,ls,lw) in zip(METHODS,styles):
        runs=[]; final=[]; best=[]
        for seed in SEEDS:
            run=curves[(curves.method==method)&(curves.seed==seed)].sort_values('round')
            if len(run)!=80 or run.run_id.nunique()!=1 or list(run['round'])!=list(range(1,81)):
                raise ValueError('B must contain one complete 80-round run per method and seed')
            x=run.cumulative_matrix_budget.to_numpy(float)
            if not np.isfinite(x).all() or not np.all(np.diff(x)>0) or abs(x[-1]-7200)>1e-8:
                raise ValueError('Invalid B budget grid')
            if grid is None: grid=x
            elif not np.allclose(grid,x,rtol=0,atol=1e-8):
                raise ValueError('B budget grids differ: no interpolation is authorized by this patch')
            y=run.accuracy.to_numpy(float)
            if not np.isfinite(y).all() or not ((0<=y)&(y<=1)).all():
                raise ValueError('Invalid B accuracy')
            runs.append(y); final.append(y[-1]); best.append(y.max())
        fm,fs=mean_sd(final); bm,bs=mean_sd(best)
        if not np.allclose((fm,bm),expected[method],atol=1e-8,rtol=0):
            raise ValueError('B means do not match the locked table/text')
        for field,value in [('final_accuracy_mean',fm),('final_accuracy_std',fs),('best_accuracy_mean',bm),('best_accuracy_std',bs)]:
            if abs(float(published.loc[method,field])-value)>1e-8:
                raise ValueError('B summary not reproducible: '+method+' '+field)
        ax.plot(grid,100*np.mean(runs,axis=0),color=color,linestyle=ls,linewidth=lw,label=method)
    ax.set(xlim=(0,7200),ylim=(0,100),xlabel='Cumulative matrix-equivalent budget',ylabel='Validation Accuracy (%)')
    ax.set_xticks([0,1800,3600,5400,7200])
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(7.5);ax.yaxis.label.set_size(7.5)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(fontsize=6.8,frameon=False,loc='lower right',ncol=2)
    plot.tight_layout(pad=.4)
    plot.savefig(fig/'fig6_ablation_budget.pdf',bbox_inches='tight',pad_inches=.02)
    plt.close(plot)
    write(gen/'b_verification.json',json.dumps({'runs':12,'rounds_per_run':80,'budget_per_run':7200,'source':str(p)},indent=2)+'\n')

def pending_a(gen: Path, reason: str) -> None:
    write(gen/'fm_a_verified_table.tex',r'''\providecommand{\FMAdepthMin}{\FMtodo{A minimum pending}}
\providecommand{\FMAdepthMax}{\FMtodo{A maximum pending}}
\providecommand{\FMApositive}{\FMtodo{A positive count pending}}
\providecommand{\FMAtotal}{90}
\providecommand{\FMAassociationSentence}{\FMtodo{Paired-RNG A statistics required.}}
\begin{table}[t]\centering
\caption{Scoring reliability: paired-RNG statistics pending.}
\label{tab:fm-score-reliability}
\fbox{\parbox{.88\columnwidth}{\FMtodo{Original 12-configuration records are needed for offline recomputation. Do not reuse stale A summary values.}}}
\end{table}
''')
    write(gen/'fm_p_corrected_table.tex',r'''\begin{table}[t]\centering
\caption{Scoring-source control: paired-RNG statistics pending.}
\label{tab:fm-source-control}
\fbox{\parbox{.88\columnwidth}{\FMtodo{Offline correction of the original Local/Local-Mean/Proxy records is pending.}}}
\end{table}
''')
    write(gen/'a_status.json',json.dumps({'status':'blocked','reason':reason,'training_started':False},indent=2)+'\n')

def prep_a(root: Path, gen: Path, bymask: dict) -> None:
    base=root/'paper_assets/02_experiments/matrix_score_20261025/full'
    files=[]
    for dist in DISTS:
        for seed in SEEDS:
            folder=base/f"cifar100_{dist.replace(':','_')}_seed{seed}"
            for stage in STAGES:
                for client in range(5):
                    files.append((dist,seed,stage,client,folder/'short_train'/f'{stage}_client{client}.csv'))
    missing=[str(x[-1]) for x in files if not x[-1].is_file()]
    if missing:
        pending_a(gen,f'{len(missing)} original pool files unavailable; first missing: {missing[0]}')
        return
    corrected=[]; correlations=[]; source_pairs=[]; count_replaced=0
    for dist,seed,stage,client,path in files:
        pool=pd.read_csv(path)
        if len(pool)!=12 or pool.config_id.nunique()!=12:
            raise ValueError('Original A must have 12 configurations per cell')
        masks=[mask(x) for x in pool.matrix_indices]
        if len(set(masks))!=12:
            raise ValueError('Duplicate configuration in original A pool')
        gains=[]; sources={}
        for pos,(_,row) in enumerate(pool.iterrows()):
            key=(dist,stage,seed,client,masks[pos])
            if cellkey(row)!=(dist,stage,seed,client):
                raise ValueError('A source metadata mismatch')
            if int(row.matrix_count)!=12 or int(row['rank'])!=8 or int(row.train_steps)!=20:
                raise ValueError('A source protocol mismatch')
            if row.model_construction!='full_adapters_forward_unselected_frozen_classifier_trainable':
                raise ValueError('A local-model construction mismatch')
            vals=(float(row.val_loss_before),float(row.val_loss_after),float(row.val_loss_reduction))
            corrected_from='original_nonfirst'
            if key in bymask:
                new=bymask[key]
                if abs(new[0]-vals[0])>1e-6:
                    raise ValueError('A/P2 starting loss mismatch')
                if int(row.config_id)>0 and not np.allclose(new,vals,rtol=0,atol=1e-6):
                    raise ValueError('Non-first A/P2 outcomes differ; do not silently mix protocols')
                vals=new; corrected_from='p2_matched_mask'
                count_replaced+=int(row.config_id==0)
            elif int(row.config_id)==0:
                raise ValueError('Missing P2 replacement for original A first mask')
            if not np.isfinite(vals).all() or abs(vals[0]-vals[1]-vals[2])>1e-8:
                raise ValueError('Invalid corrected A outcome')
            out=row.to_dict()
            out.update(val_loss_before=vals[0],val_loss_after=vals[1],val_loss_reduction=vals[2],
                       correction_source=corrected_from,original_pool_path=str(path))
            corrected.append(out);gains.append(vals[2])
            for name in str(row.method).split('+'):
                if name in ('Local','Local-Mean','Server-Proxy'):
                    if name in sources: raise ValueError('Duplicate scoring-source assignment')
                    sources[name]=vals[2]
        for method,column in [('raw','pred_local_raw'),('depth','pred_local_depth')]:
            coefficient=float(spearmanr(pool[column].to_numpy(float),gains).statistic)
            if not np.isfinite(coefficient): raise ValueError('Undefined Spearman coefficient')
            correlations.append({'distribution':dist,'checkpoint':stage,'seed':seed,'client_id':client,
                                 'method':method,'rho':coefficient,'n_configs':12})
        if set(sources)!= {'Local','Local-Mean','Server-Proxy'}:
            raise ValueError('Incomplete original P source comparison')
        for comp in ('Server-Proxy','Local-Mean'):
            source_pairs.append({'distribution':dist,'checkpoint':stage,'seed':seed,'client_id':client,
                                 'comparator':comp,'difference':sources['Local']-sources[comp]})
    if count_replaced!=90 or len(corrected)!=1080 or len(correlations)!=180:
        raise ValueError('Incomplete corrected A records')
    c=pd.DataFrame(correlations)
    sm=c.groupby(['distribution','checkpoint','method','seed'],as_index=False).rho.mean()
    cells=pd.DataFrame(source_pairs)
    psm=cells.groupby(['distribution','checkpoint','comparator','seed'],as_index=False).difference.mean()
    summary=[]; rows=[]; prows=[]
    for dist in DISTS:
        for stage in STAGES:
            values=[]
            for method in ('raw','depth'):
                sub=sm[(sm.distribution==dist)&(sm.checkpoint==stage)&(sm.method==method)]
                mu,sd=mean_sd(sub.rho)
                summary.append({'distribution':dist,'checkpoint':stage,'method':method,'mean':mu,'std':sd})
                values.append(texnum(mu,sd,3))
            rows.append(f'{LABELS[dist]} & {STAGE_LABELS[stage]} & {values[0]} & {values[1]} '+r'\\')
            vals=[]
            for comp in ('Server-Proxy','Local-Mean'):
                sub=psm[(psm.distribution==dist)&(psm.checkpoint==stage)&(psm.comparator==comp)]
                mu,sd=mean_sd(sub.difference); vals.append(texnum(mu,sd,5,True))
            prows.append(f'{LABELS[dist]} & {STAGE_LABELS[stage]} & {vals[0]} & {vals[1]} '+r'\\')
    sdf=pd.DataFrame(summary); depth=sdf[sdf.method=='depth']['mean']
    npositive=int((c[c.method=='depth'].rho>0).sum())
    interpretation=('The positive setting-level associations support using the profiled utility to prioritize configurations in this controlled protocol.'
                    if (depth>0).all() else
                    'The setting-dependent associations characterize the ranking information supplied by the score in this controlled protocol.')
    macros=(r'\newcommand{\FMAdepthMin}{'+f'{depth.min():.3f}'+r'}'+'\n'+
            r'\newcommand{\FMAdepthMax}{'+f'{depth.max():.3f}'+r'}'+'\n'+
            r'\newcommand{\FMApositive}{'+str(npositive)+r'}'+'\n'+
            r'\newcommand{\FMAtotal}{90}'+'\n'+
            r'\newcommand{\FMAassociationSentence}{'+interpretation+r'}'+'\n')
    table=[r'\begin{table}[t]',r'\centering',
           r"\caption{Agreement between predicted configuration utility and observed 20-step validation-loss reduction, measured by Spearman's $r_s$. Raw uses gradient-squared matrix scores; Calibrated additionally applies the positional factor. Entries are three-seed mean $\pm$ sample standard deviation of client-averaged coefficients.}",
           r'\label{tab:fm-score-reliability}',r'\small',r'\setlength{\tabcolsep}{3pt}',
           r'\begin{tabular}{@{}llcc@{}}',r'\toprule',r'Partition & State & Raw & Calibrated \\',r'\midrule',
           *rows,r'\bottomrule',r'\end{tabular}',r'\end{table}']
    write(gen/'fm_a_verified_table.tex',macros+'\n'.join(table)+'\n')
    ptable=[r'\begin{table*}[t]',r'\centering',
            r'\caption{Complete scoring-source control after paired-random-state correction. Values are Local-minus-comparator differences in held-out validation-loss reduction, with three-seed sample standard deviations.}',
            r'\label{tab:fm-source-control}',r'\small',r'\begin{tabular}{@{}llcc@{}}',r'\toprule',
            r'Partition & State & Local minus Proxy & Local minus Local-Mean \\',r'\midrule',
            *prows,r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    write(gen/'fm_p_corrected_table.tex','\n'.join(ptable)+'\n')
    pd.DataFrame(corrected).to_csv(gen/'a_corrected_configuration_records.csv',index=False)
    c.to_csv(gen/'a_corrected_cell_correlations.csv',index=False)
    sdf.to_csv(gen/'a_corrected_summary.csv',index=False)
    cells.to_csv(gen/'p_corrected_client_differences.csv',index=False)
    psm.to_csv(gen/'p_corrected_seed_differences.csv',index=False)
    write(gen/'a_status.json',json.dumps({'status':'complete','configuration_records':1080,
          'client_checkpoint_cells':90,'first_records_replaced':90,'positive_calibrated_cells':npositive,
          'depth_mean_min':float(depth.min()),'depth_mean_max':float(depth.max()),'training_started':False},indent=2)+'\n')

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    a=parser.parse_args();root=a.repo.resolve()
    paper=root/'paper_assets/04_manuscript';gen=paper/'generated';fig=paper/'figures'
    gen.mkdir(parents=True,exist_ok=True)
    _,bymask=prep_p2(root,gen)
    prep_b(root,gen,fig)
    try:
        prep_a(root,gen,bymask)
    except Exception as exc:
        pending_a(gen,repr(exc))
        raise
    print('P2/B verified; A status:',(gen/'a_status.json').read_text())
    print('Only derivative tables/figures written. No training was started.')

if __name__=='__main__':
    main()
