"""Independently check every retained fit and summarize the fixed experiment.

No fitting, seed selection, thresholds, or external evaluation are performed.
Requires the private cache created by diagnose_controls.py; outputs aggregates.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import sys
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr

WORK = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent

def read(p): return json.loads(p.read_text())
def digest(b): return hashlib.sha256(b).hexdigest()
def array_digest(a,dtype='<f8'): return digest(np.asarray(a,dtype=dtype).tobytes())

def rank_auc(y,s,w=None):
    """Independent pair counting at exact score ties, without sklearn/rankdata."""
    w=np.ones(len(y)) if w is None else w
    order=np.argsort(s,kind='stable');ss=s[order];yy=y[order];ww=w[order]
    starts=np.r_[0,np.flatnonzero(ss[1:]!=ss[:-1])+1]
    positive=np.add.reduceat(ww*yy,starts)
    negative=np.add.reduceat(ww*(1-yy),starts)
    return float(np.sum(positive*(np.cumsum(negative)-negative/2))/(positive.sum()*negative.sum()))

def ap(y,s):
    order=np.argsort(-s,kind='stable');ss=s[order];yy=y[order]
    ends=np.r_[np.flatnonzero(ss[1:]!=ss[:-1]),len(y)-1]
    tp=np.cumsum(yy)[ends];precision=tp/(ends+1)
    return float(np.sum(np.diff(np.r_[0,tp])*precision)/sum(y))

def summary(values):
    a=np.array(values)
    return {'n':len(a),'mean':float(a.mean()),'sd':float(a.std(ddof=1)),
            'min':float(a.min()),'q025':float(np.quantile(a,.025)),
            'median':float(np.median(a)),'q975':float(np.quantile(a,.975)),'max':float(a.max())}

def main():
    protocol=read(WORK/'PROTOCOL.json');start=read(WORK/'execution-start.json')
    completion=read(WORK/'execution-complete.json')
    expected=list(range(1000,1099));assert start['seeds']==expected and completion['attempts']==99
    results=[read(WORK/'additional-fits'/f'{s}-result.json') for s in expected]
    assert [r['seed'] for r in results]==expected
    assert set(p.name for p in (WORK/'additional-fits').glob('*-result.json'))=={f'{s}-result.json' for s in expected}
    assert all(r['status']=='completed' for r in results), 'Report failed attempts before summarizing'
    prep=read(WORK/'preparation-receipt.json')
    x,z,y,v,dw=[np.load(WORK/'private-cache'/f'{n}.npy',mmap_mode='r') for n in ['x','z','y','v','domain_weights']]
    assert array_digest(x)==prep['training_features_sha256'] and array_digest(z)==prep['validation_features_sha256']
    assert array_digest(y,'<i8')==prep['training_labels_sha256'] and array_digest(v,'<i8')==prep['validation_labels_sha256']
    assert read(WORK/'feature-order.json')[20]=='is_https'
    verification=[];rows=[]
    for r in results:
        seed=r['seed'];labels=np.random.Generator(np.random.PCG64(seed)).permutation(y)
        assert array_digest(labels,'<i8')==r['consumed_labels_sha256']
        assert r['consumed_features_sha256']==prep['training_features_sha256']
        assert r['protocol_sha256']==digest((WORK/'PROTOCOL.json').read_bytes())
        assert r['analysis_source_sha256']==start['source_sha256']==digest((WORK/'diagnose_controls.py').read_bytes())
        assert r['parameters']==protocol['classifier'] and r['class_counts']==np.bincount(y).tolist()
        beta=np.array(r['coefficients']);logits=np.einsum('ij,j->i',z,beta)+r['intercept'];scores=expit(logits)
        assert array_digest(scores)==r['score_sha256']
        pv=np.random.Generator(np.random.PCG64(protocol['random_validation_seed']+seed)).permutation(v)
        calculated={'auc':rank_auc(v,scores),'ap':ap(v,scores),
          'without_https_auc':rank_auc(v,logits-z[:,20]*beta[20]),
          'randomized_validation_auc':rank_auc(pv,scores),
          'domain_equal_weighted_auc':rank_auc(v,scores,dw)}
        error=max(abs(calculated[k]-r[k]) for k in calculated)
        assert error<1e-11,(seed,error)
        verification.append({'seed':seed,'max_metric_difference':error,'scores_digest_matches':True})
        rows.append({'seed':seed,**{k:r[k] for k in calculated},'is_https_coefficient':r['coefficients'][20],
                     'iterations':r['iterations'],'warnings':len(r['warnings']),'coefficient_l2':r['coefficient_l2']})
    saved=read(WORK/'saved-fit-diagnosis.json');old=saved['saved_fits']
    strict=[read(WORK/'convergence'/f'{s}-result.json') for s in range(42,47)]
    for s,r in zip(old,strict):
        assert s['seed']==r['seed'] and s['model_sha256']==r['initial_model_sha256']
        assert r['protocol_sha256']==digest((WORK/'CONVERGENCE_PROTOCOL.json').read_bytes())
        assert r['source_sha256']==digest((WORK/'check_convergence.py').read_bytes())
        labels=np.random.Generator(np.random.PCG64(r['seed'])).permutation(y)
        assert array_digest(labels,'<i8')==r['input_labels_sha256']
        beta=np.array(r['coefficients']);logits=np.einsum('ij,j->i',z,beta)+r['intercept'][0]
        assert abs(rank_auc(v,logits)-r['auc_after'])<1e-12
        assert abs(s['auc']-r['auc_before'])<1e-12
    result={'experiment':protocol['study'],'completed':len(results),
      'warnings':sum(len(r['warnings']) for r in results),
      'iterations_min':min(r['iterations'] for r in results),'iterations_max':max(r['iterations'] for r in results),
      'all_below_iteration_limit':all(r['iterations']<5000 for r in results),
      'auc':summary([r['auc'] for r in results]),'ap':summary([r['ap'] for r in results]),
      'without_https_auc':summary([r['without_https_auc'] for r in results]),
      'randomized_validation_auc':summary([r['randomized_validation_auc'] for r in results]),
      'domain_equal_weighted_auc':summary([r['domain_equal_weighted_auc'] for r in results]),
      'at_or_above_original_max_auc':sum(r['auc']>=old[-1]['auc'] for r in results),
      'original_max_auc':old[-1]['auc'],
      'auc_https_coefficient_spearman':float(spearmanr([r['auc'] for r in results],[r['coefficients'][20] for r in results]).statistic),
      'coefficient_sign_groups':{name:{'n':len(g),'auc':summary([r['auc'] for r in g]) if len(g)>1 else None}
          for name,g in [('negative',[r for r in results if r['coefficients'][20]<0]),
                         ('positive',[r for r in results if r['coefficients'][20]>0]),
                         ('zero',[r for r in results if r['coefficients'][20]==0])]},
      'mean_absolute_auc_departure':float(np.mean([abs(r['auc']-.5) for r in results])),
      'mean_absolute_auc_departure_without_https':float(np.mean([abs(r['without_https_auc']-.5) for r in results])),
      'convergence':{'n':len(strict),'warnings':sum(len(r['warnings']) for r in strict),
         'max_absolute_auc_change':max(abs(r['auc_difference']) for r in strict),
         'max_iterations':max(r['iterations'] for r in strict),
         'max_mean_loss_kkt_residual':max(r['kkt']['max_residual_mean_loss'] for r in strict)},
      'scope':'Conditional algorithmic sensitivity on one fixed dataset; empirical quantiles, not population confidence intervals; not a leakage p-value.'}
    (WORK/'diagnostic-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (WORK/'diagnostic-verification.json').write_text(json.dumps({'status':'verified','fits_checked':99,'strict_fits_checked':5,
      'independent_auc_implementation':'weighted pair counting with half credit at ties',
      'maximum_metric_difference':max(r['max_metric_difference'] for r in verification),'checks':verification},indent=2)+'\n')
    with (WORK/'permutation-summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fixed=[]
    for s,r in zip(old,strict):
        fixed.append({'seed':s['seed'],'original_auc':s['auc'],'without_https_auc':s['without_https_auc'],
         'is_https_coefficient':s['features'][20]['coefficient'],'refined_auc':r['auc_after'],
         'refined_minus_original_auc':r['auc_difference'],'original_iterations':s['iterations'],
         'refinement_iterations':r['iterations'],'refinement_warnings':len(r['warnings'])})
    with (WORK/'original-controls-summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(fixed[0]));w.writeheader();w.writerows(fixed)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
