from pathlib import Path
import json,sys,time,warnings,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from diagnose_controls import kkt,digest,array_digest,write,timestamp
WORK=Path(__file__).resolve().parent
OLD=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else None

def check(args):
 seed,old=args
 original=json.loads((Path(old)/f'seed-{seed}/evidence/model.json').read_text())
 x,z,y,v=[np.load(WORK/'private-cache'/f'{n}.npy',mmap_mode='r') for n in ['x','z','y','v']]
 labels=np.random.Generator(np.random.PCG64(seed)).permutation(y)
 params=original['parameters']['classifier'];params.update(tol=1e-7,max_iter=20000,warm_start=True)
 model=LogisticRegression(**params)
 model.classes_=np.array([0,1]);model.coef_=np.array(original['state']['coefficients']);model.intercept_=np.array(original['state']['intercept']);model.n_features_in_=25
 before=roc_auc_score(v, expit(np.einsum('ij,j->i',z,model.coef_[0])+model.intercept_[0]))
 start=time.monotonic()
 receipt={'seed':seed,'started':timestamp(),'input_labels_sha256':array_digest(labels,'<i8'),'parameters':params,
  'initial_model_sha256':digest((Path(old)/f'seed-{seed}/evidence/model.json').read_bytes()),'protocol_sha256':digest((WORK/'CONVERGENCE_PROTOCOL.json').read_bytes()),'source_sha256':digest(Path(__file__).read_bytes())}
 write(WORK/'convergence'/f'{seed}-started.json',receipt)
 with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always');model.fit(x,labels)
  logits=np.einsum('ij,j->i',z,model.coef_[0])+model.intercept_[0]
  after=float(roc_auc_score(v,logits))
  result={**receipt,'finished':timestamp(),'elapsed_seconds':time.monotonic()-start,'auc_before':float(before),'auc_after':after,
   'auc_difference':after-before,'without_https_auc':float(roc_auc_score(v,logits-z[:,20]*model.coef_[0,20])),
   'iterations':int(model.n_iter_[0]),'coefficients':model.coef_[0].tolist(),'intercept':model.intercept_.tolist(),
   'kkt':kkt(x,labels,model.coef_[0],model.intercept_[0]),'warnings':[{'type':w.category.__name__,'message':str(w.message)} for w in caught]}
 write(WORK/'convergence'/f'{seed}-result.json',result)
 return {k:result[k] for k in ['seed','auc_before','auc_after','auc_difference','iterations','elapsed_seconds','kkt','warnings']}
if __name__=='__main__':
 (WORK/'convergence').mkdir()
 with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
  for f in as_completed([pool.submit(check,(s,str(OLD))) for s in range(42,47)]):print(json.dumps(f.result()),flush=True)
