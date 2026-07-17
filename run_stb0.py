#!/usr/bin/env python3
"""STB-0: successful-translation test of a two-coordinate semantic/boundary profile.

WMT20 MQM professional annotations are split by dimension. For each translation
unit, one rater is held out as the success witness; all other raters supply the
semantic and boundary coordinates. Document-grouped CV and a model-free
admission rule compare one-dimensional semantic clearance against two-dimensional
semantic-plus-boundary clearance.
"""
from __future__ import annotations
import argparse, csv, html, json, math, random, re, sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED=20260717
random.seed(SEED); np.random.seed(SEED)
TAG=re.compile(r'</?v>')
SEMANTIC={'Accuracy','Non-translation'}

def topcat(cat):
    cat=str(cat or '').strip()
    if cat.lower() in {'','no-error'}: return 'No-error'
    if cat.lower().startswith('non-translation'): return 'Non-translation'
    return cat.split('/',1)[0]

def weight(cat, sev):
    top=topcat(cat); sev=str(sev or '').lower()
    if top=='No-error' or sev=='no-error': return 0.0
    if top=='Non-translation': return 25.0
    if sev=='major': return 5.0
    if sev=='minor': return 0.1 if cat=='Fluency/Punctuation' else 1.0
    return 0.0

@dataclass
class Rating:
    lp:str; system:str; doc:str; doc_id:str; seg_id:str; rater:str; source:str; target:str
    semantic:float=0.0; boundary:float=0.0; total:float=0.0
    cats:Counter=field(default_factory=Counter)

def load(path:Path, lp:str):
    rows={}; meta=Counter()
    with path.open(encoding='utf-8',newline='') as f:
        rd=csv.DictReader(f,delimiter='\t')
        need={'system','doc','doc_id','seg_id','rater','source','target','category','severity'}
        if not need.issubset(rd.fieldnames or []): raise ValueError(f'missing {need-set(rd.fieldnames or [])}')
        for x in rd:
            target=TAG.sub('',x['target']).strip()
            key=(lp,x['system'],x['doc'],x['doc_id'],x['seg_id'],x['rater'],x['source'],target)
            r=rows.setdefault(key,Rating(*key))
            w=weight(x['category'],x['severity']); top=topcat(x['category'])
            if w:
                r.cats[x['category']]+=1
                if top in SEMANTIC: r.semantic+=w
                else: r.boundary+=w
                r.total+=w
            meta['annotation_rows']+=1
    df=pd.DataFrame([{**r.__dict__,'cats':json.dumps(r.cats,sort_keys=True,ensure_ascii=False),
                      'unit_id':f'{r.lp}|{r.system}|{r.doc_id}|{r.seg_id}',
                      'group_id':f'{r.lp}|{r.doc_id}'} for r in rows.values()])
    return df, {'lp':lp,'annotation_rows':int(meta['annotation_rows']),'ratings':len(df),
                'raters':sorted(df.rater.unique()),'systems':int(df.system.nunique()),
                'documents':int(df.group_id.nunique())}

def cross_rater(df):
    out=[]
    for uid,g in df.groupby('unit_id',sort=False):
        if g.rater.nunique()<3: continue
        for _,h in g.iterrows():
            o=g[g.rater!=h.rater]
            s=float(o.semantic.mean()); b=float(o.boundary.mean())
            out.append({'unit_id':uid,'group_id':h.group_id,'lp':h.lp,'system':h.system,
                'doc_id':h.doc_id,'seg_id':h.seg_id,'heldout_rater':h.rater,
                'source':h.source,'target':h.target,'semantic_feature':s,
                'boundary_feature':b,'collapsed_feature':s+b,
                'heldout_semantic':float(h.semantic),'heldout_boundary':float(h.boundary),
                'heldout_total':float(h.total),'strict_success':int(h.total==0),
                'usable_success':int(h.total<=1),'heldout_categories':h.cats})
    return pd.DataFrame(out)

def rank_corr(a,b):
    v=stats.spearmanr(a,b).statistic
    return float(v) if np.isfinite(v) else None

def rule(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=int)
    tp=int(((p==1)&(y==1)).sum()); fp=int(((p==1)&(y==0)).sum())
    tn=int(((p==0)&(y==0)).sum()); fn=int(((p==0)&(y==1)).sum())
    return {'tp':tp,'fp':fp,'tn':tn,'fn':fn,'coverage':float(p.mean()),
            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
            'false_success_rate':fp/(tp+fp) if tp+fp else None}

def rules(df):
    y=df.strict_success.to_numpy()
    sem=(df.semantic_feature==0).astype(int).to_numpy()
    typed=((df.semantic_feature==0)&(df.boundary_feature==0)).astype(int).to_numpy()
    rs={'semantic_clear_1d':rule(y,sem),'typed_clear_2d':rule(y,typed)}
    es=sem!=y; et=typed!=y
    b=int((es&~et).sum()); c=int((~es&et).sum())
    rs['mcnemar']={'semantic_wrong_typed_right':b,'semantic_right_typed_wrong':c,
                   'p':float(stats.binomtest(min(b,c),b+c,0.5).pvalue) if b+c else 1.0}
    return rs

def bootstrap(df,n=500):
    rng=np.random.default_rng(SEED); gs=df.group_id.unique(); by={g:df[df.group_id==g] for g in gs}; vals=[]
    for _ in range(n):
        x=pd.concat([by[g] for g in rng.choice(gs,len(gs),replace=True)],ignore_index=True)
        r=rules(x); vals.append(r['semantic_clear_1d']['false_success_rate']-r['typed_clear_2d']['false_success_rate'])
    q=np.quantile(vals,[.025,.5,.975])
    return {'mean':float(np.mean(vals)),'median':float(q[1]),'ci95':[float(q[0]),float(q[2])],'n':n}

def cv(df,features,outcome='strict_success'):
    X=df[features]; y=df[outcome].to_numpy(); groups=df.group_id.to_numpy(); pred=np.full(len(df),np.nan)
    folds=GroupKFold(min(5,len(np.unique(groups))))
    for tr,te in folds.split(X,y,groups):
        if len(np.unique(y[tr]))<2: pred[te]=y[tr].mean(); continue
        m=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),
                    ('lr',LogisticRegression(max_iter=2000,random_state=SEED))])
        m.fit(X.iloc[tr],y[tr]); pred[te]=m.predict_proba(X.iloc[te])[:,1]
    return {'roc_auc':float(roc_auc_score(y,pred)),'average_precision':float(average_precision_score(y,pred)),
            'brier':float(brier_score_loss(y,pred)),'pred':pred}

def quadrant(df):
    masks={'S0_B0':(df.semantic_feature==0)&(df.boundary_feature==0),
           'S0_Bplus':(df.semantic_feature==0)&(df.boundary_feature>0),
           'Splus_B0':(df.semantic_feature>0)&(df.boundary_feature==0),
           'Splus_Bplus':(df.semantic_feature>0)&(df.boundary_feature>0)}
    return {k:{'n':int(m.sum()),'share':float(m.mean()),'strict_success':float(df.loc[m,'strict_success'].mean()) if m.any() else None,
               'usable_success':float(df.loc[m,'usable_success'].mean()) if m.any() else None,
               'heldout_mean_penalty':float(df.loc[m,'heldout_total'].mean()) if m.any() else None} for k,m in masks.items()}

def save_report(out,summary,metrics,examples):
    r=summary['rules']; q=summary['quadrants']; boot=summary['bootstrap']; m={x['model']:x for x in metrics}
    s=r['semantic_clear_1d']; t=r['typed_clear_2d']
    lines=f'''# STB-0 — Successful Translation Benchmark

## Result

This executed pilot tests whether successful translation is better represented by two independently retained coordinates—semantic preservation and boundary/target-regime realization—than by a semantic-only gate.

- Language pairs: {', '.join(summary['language_pairs'])}
- Unique translation units: {summary['n_units']:,}
- Leave-one-rater-out examples: {summary['n_examples']:,}
- Documents: {summary['n_documents']:,}

## Main findings

1. S/B rank correlation: **{summary['semantic_boundary_spearman']:.3f}**.
2. Boundary-only cohort (S=0, B>0): **{100*summary['boundary_only_share']:.1f}%**.
3. Held-out strict success: **{100*q['S0_B0']['strict_success']:.1f}%** for S=0,B=0 versus **{100*q['S0_Bplus']['strict_success']:.1f}%** for S=0,B>0.
4. False-success rate: semantic-only **{100*s['false_success_rate']:.1f}%**; typed 2D **{100*t['false_success_rate']:.1f}%**. Reduction **{100*boot['mean']:.1f} points**, document-bootstrap 95% CI **[{100*boot['ci95'][0]:.1f}, {100*boot['ci95'][1]:.1f}]**.
5. Strict-success ROC-AUC: semantic-only **{m['semantic_only_1d']['roc_auc']:.3f}**, collapsed scalar **{m['collapsed_total_1d']['roc_auc']:.3f}**, typed 2D **{m['typed_2d']['roc_auc']:.3f}**.

## Admission trade-off

| Rule | Coverage | Precision | Recall | False-success |
|---|---:|---:|---:|---:|
| S=0 | {100*s['coverage']:.1f}% | {100*s['precision']:.1f}% | {100*s['recall']:.1f}% | {100*s['false_success_rate']:.1f}% |
| S=0 and B=0 | {100*t['coverage']:.1f}% | {100*t['precision']:.1f}% | {100*t['recall']:.1f}% | {100*t['false_success_rate']:.1f}% |

McNemar exact p={r['mcnemar']['p']:.4g}; semantic wrong/typed right={r['mcnemar']['semantic_wrong_typed_right']:,}; semantic right/typed wrong={r['mcnemar']['semantic_right_typed_wrong']:,}.

## Interpretation boundary

This is a cross-rater construct-validity result. The coordinates are derived from professional MQM annotations, not automatically inferred. It can support or refute the claim that translation success contains an independently observable semantic dimension and boundary/realization dimension. It does not calibrate POINT/RE-DERIVE/BORROW/WEAK/REFUSE or validate full Regime Convergence.
'''
    (out/'report.md').write_text(lines,encoding='utf-8')
    escaped=html.escape(lines)
    (out/'report.html').write_text(f"<!doctype html><meta charset=utf-8><title>STB-0</title><style>body{{font:16px system-ui;max-width:980px;margin:40px auto;padding:0 20px;line-height:1.55}}pre{{white-space:pre-wrap;background:#f5f6f8;padding:20px;border:1px solid #ddd}}</style><h1>STB-0</h1><img style='max-width:100%' src='../figures/quadrants.png'><pre>{escaped}</pre>",encoding='utf-8')

def tbool(x): return bool(x)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',action='append',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    out=Path(a.out); res=out/'results'; fig=out/'figures'; res.mkdir(parents=True,exist_ok=True); fig.mkdir(exist_ok=True)
    frames=[]; sources=[]
    for spec in a.input:
        lp,path=spec.split('=',1); d,m=load(Path(path),lp); frames.append(d); sources.append(m)
    ratings=pd.concat(frames,ignore_index=True); x=cross_rater(ratings)
    if len(x)<1000: raise RuntimeError('insufficient cross-rater sample')
    r=rules(x); b=bootstrap(x); q=quadrant(x)
    specs={'semantic_only_1d':['semantic_feature'],'collapsed_total_1d':['collapsed_feature'],
           'typed_2d':['semantic_feature','boundary_feature'],'typed_2d_interaction':['semantic_feature','boundary_feature']}
    metrics=[]
    for name,fs in specs.items():
        z=cv(x,fs); x['p_'+name]=z.pop('pred'); metrics.append({'model':name,**z})
    ex=x[(x.semantic_feature==0)&(x.boundary_feature>0)&(x.strict_success==0)].sort_values(['heldout_boundary','boundary_feature'],ascending=False).head(50)
    summary={'schema':'stb0-summary-v1','seed':SEED,'sources':sources,'language_pairs':sorted(x.lp.unique()),
             'n_units':int(x.unit_id.nunique()),'n_examples':len(x),'n_documents':int(x.group_id.nunique()),
             'semantic_boundary_spearman':rank_corr(x.semantic_feature,x.boundary_feature),
             'boundary_only_share':float(((x.semantic_feature==0)&(x.boundary_feature>0)).mean()),
             'quadrants':q,'rules':r,'bootstrap':b,'models':metrics,
             'claims':{'two_coordinates_observed':q['S0_Bplus']['n']>100,
                       'typed_rule_reduces_false_success':tbool(r['typed_clear_2d']['false_success_rate']<r['semantic_clear_1d']['false_success_rate']),
                       'automatic_boundary_estimator_validated':False,'full_rc_validated':False}}
    json.dump(summary,(res/'summary.json').open('w'),indent=2,ensure_ascii=False,allow_nan=False)
    ratings.to_csv(res/'rater_dimensions.csv',index=False); x.to_csv(res/'cross_rater_predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(res/'model_metrics.csv',index=False)
    ex[['lp','system','doc_id','seg_id','heldout_rater','source','target','semantic_feature','boundary_feature','heldout_semantic','heldout_boundary','heldout_total','heldout_categories']].to_csv(res/'semantic_clear_boundary_failures.csv',index=False)
    vals=[q[k]['strict_success'] for k in ['S0_B0','S0_Bplus','Splus_B0','Splus_Bplus']]
    ns=[q[k]['n'] for k in ['S0_B0','S0_Bplus','Splus_B0','Splus_Bplus']]
    f,ax=plt.subplots(figsize=(8,4.6)); bars=ax.bar(['S=0,B=0','S=0,B>0','S>0,B=0','S>0,B>0'],vals); ax.set_ylim(0,1); ax.set_ylabel('Held-out strict-success rate'); ax.set_title('Two-coordinate translation profile')
    for bar,n in zip(bars,ns): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.02,f'n={n}',ha='center',fontsize=8)
    f.tight_layout(); f.savefig(fig/'quadrants.png',dpi=180); plt.close(f)
    save_report(res,summary,metrics,ex)
    print(json.dumps({'units':summary['n_units'],'examples':summary['n_examples'],'false_success_semantic':r['semantic_clear_1d']['false_success_rate'],'false_success_typed':r['typed_clear_2d']['false_success_rate'],'models':metrics},indent=2))

if __name__=='__main__': main()
