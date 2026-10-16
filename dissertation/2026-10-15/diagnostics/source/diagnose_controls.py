"""Retrospective, reproducible analysis of frozen shuffled-label controls.

Only aggregate results and this source are suitable for publication. The cache,
input partitions and original prediction rows stay local. No external test data
are loaded, and no operating threshold is selected.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np
import scipy
from scipy.special import expit
from scipy.stats import rankdata
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def digest(value):
    return hashlib.sha256(value).hexdigest()


def array_digest(value, dtype='<f8'):
    return digest(np.asarray(value, dtype=dtype).tobytes(order='C'))


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def auc_rank(y, score):
    """Mann–Whitney AUC with average ranks for exact ties."""
    n1 = int(y.sum())
    ranks = rankdata(score)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def kkt(x, y, beta, intercept, c=1.0):
    """Stationarity residual for sum(weighted log loss) + ||beta||_1 / C."""
    weights = len(y) / (2 * np.bincount(y)[y])
    residual = weights * (expit(x @ beta + intercept) - y)
    gradient = x.T @ residual
    violation = np.where(beta != 0, np.abs(gradient + np.sign(beta) / c),
                         np.maximum(np.abs(gradient) - 1 / c, 0))
    return {'max_coefficient_residual_sum_loss': float(violation.max()),
            'intercept_residual_sum_loss': float(abs(residual.sum())),
            'max_residual_mean_loss': float(max(violation.max(), abs(residual.sum())) / weights.sum())}


def self_test():
    y = np.array([0, 0, 1, 1])
    for score in [np.array([0., 0., 1., 1.]), np.array([1., 0., 0., 1.]), np.ones(4)]:
        assert abs(auc_rank(y, score) - roc_auc_score(y, score)) < 1e-15
        assert abs(auc_rank(y, -score) + auc_rank(y, score) - 1) < 1e-15
    assert kkt(np.zeros((4, 2)), y, np.zeros(2), 0)['max_residual_mean_loss'] == 0
    x = np.array([[-1.], [-1.], [1.], [1.]])
    assert kkt(x, y, np.zeros(1), 0)['max_coefficient_residual_sum_loss'] == 1
    print('Metric and stationarity checks passed', flush=True)


def prepare(args):
    work, source, data, old = map(Path, [args.work, args.source, args.data, args.old])
    protocol = json.loads((work / 'PROTOCOL.json').read_text())
    sys.path.insert(0, str(source / 'src'))
    from automated_phishing_detection.url_features import FEATURE_NAMES, extract_url_features
    from automated_phishing_detection.gmm_monitor import _require_runtime
    _require_runtime()
    self_test()
    partitions = []
    for name in ['train', 'validation']:
        content = (data / (name + '.jsonl')).read_bytes()
        assert digest(content) == protocol[name + '_sha256'], name
        rows = [json.loads(line) for line in content.splitlines()]
        assert all(row['split'] == name for row in rows)
        partitions.append(rows)
    train, validation = partitions
    assert len(train) == 166248 and len(validation) == 32695
    assert not ({r['registrable_domain'] for r in train} & {r['registrable_domain'] for r in validation})
    assert not ({r['record_id'] for r in train} & {r['record_id'] for r in validation})
    assert not ({r['canonical_url_sha256'] for r in train} & {r['canonical_url_sha256'] for r in validation})
    y, v = [np.array([r['is_phishing'] for r in rows], dtype=np.int64) for rows in partitions]
    x, z = [np.array([extract_url_features(r['raw_url']) for r in rows], dtype=np.float64)
            for rows in partitions]
    assert x.shape == (166248, 25) and z.shape == (32695, 25)
    scaler = StandardScaler().fit(x)
    xs, zs = scaler.transform(x), scaler.transform(z)
    domains = Counter(r['registrable_domain'] for r in validation)
    dw = np.array([1 / domains[r['registrable_domain']] for r in validation])
    https = FEATURE_NAMES.index('is_https')
    results = []
    with threadpool_limits(limits=1):
        for seed in range(42, 47):
            directory = old / f'seed-{seed}'
            receipt = json.loads((directory / 'fit-call-prepared.json').read_text())
            model_bytes = (directory / 'evidence/model.json').read_bytes()
            model = json.loads(model_bytes)
            state = model['state']
            labels = np.random.Generator(np.random.PCG64(seed)).permutation(y)
            indices = np.random.Generator(np.random.PCG64(seed)).permutation(np.arange(len(y), dtype=np.int64))
            assert receipt['ordered_record_ids'] == [r['record_id'] for r in train]
            assert np.array_equal(labels, receipt['consumed_labels'])
            assert array_digest(labels, '<i8') == receipt['consumed_labels_sha256']
            assert array_digest(y, '<i8') == receipt['original_labels_sha256']
            assert array_digest(indices, '<i8') == receipt['permutation_indices_sha256']
            assert array_digest(x) == receipt['source_features']['sha256']
            assert array_digest(xs) == receipt['scaled_features']['sha256']
            assert np.array_equal(scaler.mean_, state['scaler']['mean'])
            assert np.array_equal(scaler.scale_, state['scaler']['scale'])
            predictions = [json.loads(line) for line in (directory / 'evidence/validation-predictions.jsonl').read_text().splitlines()]
            assert [r['record_id'] for r in predictions] == [r['record_id'] for r in validation]
            assert np.array_equal([r['label'] for r in predictions], v)
            stored = np.array([r['probability'] for r in predictions])
            beta, intercept = np.array(state['coefficients'][0]), state['intercept'][0]
            contributions = zs * beta
            logits = np.einsum('ij,j->i', zs, beta) + intercept
            scores = expit(logits)
            assert np.max(np.abs(scores - stored)) < 1e-12
            published = json.loads((old / f'seed-{seed}-summary.json').read_text())['result']['score_metrics']
            assert abs(roc_auc_score(v, stored) - published['roc_auc']) < 1e-12
            assert abs(average_precision_score(v, stored) - published['average_precision']) < 1e-12
            assert abs(auc_rank(v, logits) - published['roc_auc']) < 1e-6
            class_delta = contributions[v == 1].mean(axis=0) - contributions[v == 0].mean(axis=0)
            feature_details = []
            for j, name in enumerate(FEATURE_NAMES):
                feature_details.append({'feature': name, 'coefficient': float(beta[j]),
                    'positive_minus_negative_mean_logit_contribution': float(class_delta[j]),
                    'component_auc': float(roc_auc_score(v, contributions[:, j])),
                    'auc_with_component_removed': float(roc_auc_score(v, logits - contributions[:, j]))})
            result = {'seed': seed, 'model_sha256': digest(model_bytes),
                'all_input_identity_checks_passed': True, 'iterations': state['n_iter'][0],
                'max_probability_reconstruction_error': float(np.max(np.abs(scores - stored))),
                'auc': published['roc_auc'], 'ap': published['average_precision'],
                'rank_sum_auc': auc_rank(v, logits),
                'reverse_logit_auc': auc_rank(v, -logits),
                'without_https_auc': float(roc_auc_score(v, logits - contributions[:, https])),
                'https_only_auc': float(roc_auc_score(v, contributions[:, https])),
                'domain_equal_weighted_auc': float(roc_auc_score(v, stored, sample_weight=dw)),
                'distinct_probabilities': int(len(np.unique(stored))),
                'score_quantiles': np.quantile(stored, [0, .25, .5, .75, 1]).tolist(),
                'coefficient_l2': float(np.linalg.norm(beta)),
                'kkt': kkt(xs, labels, beta, intercept), 'features': feature_details}
            results.append(result)
            print('saved', seed, {k: result[k] for k in ['auc', 'without_https_auc', 'https_only_auc', 'kkt']}, flush=True)
    report = {'protocol_sha256': digest((work/'PROTOCOL.json').read_bytes()),
        'source_commit': protocol['source_commit'], 'runtime': {'python': platform.python_version(),
            'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
            'blas': np.__config__.CONFIG['Build Dependencies']['blas']['name']},
        'partitions': {'training_rows': len(y), 'validation_rows': len(v),
            'train_class_counts': np.bincount(y).tolist(), 'validation_class_counts': np.bincount(v).tolist(),
            'train_validation_domain_id_and_canonical_url_disjoint': True,
            'validation_domains': len(domains), 'largest_validation_domain': max(domains.values()),
            'validation_unique_feature_vectors': int(len(np.unique(z, axis=0))),
            'https_by_validation_class': {str(label): {'n': int((v==label).sum()),
                'https': int(z[v==label, https].sum()), 'fraction': float(z[v==label, https].mean())}
                for label in [0,1]}}, 'saved_fits': results}
    write(work/'saved-fit-diagnosis.json', report)
    cache = work/'private-cache'
    cache.mkdir()
    for name, values in [('x', xs), ('z', zs), ('y', y), ('v', v), ('domain_weights', dw)]:
        np.save(cache/(name+'.npy'), values)
    write(cache/'scaler.json', {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()})
    write(work/'feature-order.json', list(FEATURE_NAMES))
    write(work/'preparation-receipt.json', {'created': timestamp(), 'protocol_sha256': report['protocol_sha256'],
        'analysis_source_sha256': digest(Path(__file__).read_bytes()),
        'training_features_sha256': array_digest(xs), 'validation_features_sha256': array_digest(zs),
        'training_labels_sha256': array_digest(y,'<i8'), 'validation_labels_sha256': array_digest(v,'<i8'),
        'input_train_sha256': protocol['train_sha256'], 'input_validation_sha256': protocol['validation_sha256']})


def fit_one(arguments):
    seed, work_string = arguments
    work = Path(work_string)
    cache, output = work/'private-cache', work/'additional-fits'
    protocol = json.loads((work/'PROTOCOL.json').read_text())
    x,z,y,v,dw = [np.load(cache/(n+'.npy'), mmap_mode='r') for n in ['x','z','y','v','domain_weights']]
    start = time.monotonic()
    indices = np.random.Generator(np.random.PCG64(seed)).permutation(len(y))
    labels = np.asarray(y[indices], dtype=np.int64)
    assert np.array_equal(labels, np.random.Generator(np.random.PCG64(seed)).permutation(y))
    assert np.array_equal(np.bincount(labels), np.bincount(y))
    labels.flags.writeable = False
    receipt = {'seed': seed, 'started': timestamp(), 'protocol_sha256': digest((work/'PROTOCOL.json').read_bytes()),
        'consumed_labels_sha256': array_digest(labels,'<i8'), 'permutation_indices_sha256': array_digest(indices,'<i8'),
        'consumed_features_sha256': array_digest(x), 'class_counts': np.bincount(labels).tolist(),
        'parameters': protocol['classifier'], 'analysis_source_sha256': digest(Path(__file__).read_bytes())}
    write(output/f'{seed}-started.json', receipt)
    try:
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            model = LogisticRegression(**protocol['classifier']).fit(x, labels)
            beta, intercept = model.coef_[0], model.intercept_[0]
            logits = np.einsum('ij,j->i', z, beta) + intercept
            score = expit(logits)
            pv = np.random.Generator(np.random.PCG64(protocol['random_validation_seed']+seed)).permutation(v)
            result = {**receipt, 'finished': timestamp(), 'status': 'completed',
                'elapsed_seconds': time.monotonic()-start,
                'warnings': [{'type': w.category.__name__, 'message': str(w.message)} for w in observed],
                'iterations': int(model.n_iter_[0]), 'coefficients': beta.tolist(), 'intercept': float(intercept),
                'auc': float(roc_auc_score(v,score)), 'ap': float(average_precision_score(v,score)),
                'without_https_auc': float(roc_auc_score(v, logits-z[:,20]*beta[20])),
                'randomized_validation_auc': float(roc_auc_score(pv,score)),
                'domain_equal_weighted_auc': float(roc_auc_score(v,score,sample_weight=dw)),
                'score_sha256': array_digest(score), 'coefficient_l2': float(np.linalg.norm(beta)),
                'kkt': kkt(x,labels,beta,intercept)}
            assert abs(result['auc']-auc_rank(v,score)) < 1e-12
        write(output/f'{seed}-result.json', result)
        return {'seed': seed, 'auc': result['auc'], 'seconds': result['elapsed_seconds'], 'warnings': len(result['warnings'])}
    except Exception as error:
        result = {**receipt, 'finished': timestamp(), 'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)}
        write(output/f'{seed}-result.json', result)
        return {'seed':seed,'status':'failed','error_type':type(error).__name__}


def run(args):
    work=Path(args.work).resolve()
    protocol=json.loads((work/'PROTOCOL.json').read_text())
    (work/'additional-fits').mkdir()
    seeds=list(range(protocol['additional_seed_start'], protocol['additional_seed_stop_exclusive']))
    write(work/'execution-start.json', {'started':timestamp(),'seeds':seeds,'source_sha256':digest(Path(__file__).read_bytes()),
        'protocol_sha256':digest((work/'PROTOCOL.json').read_bytes())})
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        tasks=[pool.submit(fit_one,(seed,str(work))) for seed in seeds]
        for i,future in enumerate(as_completed(tasks),1): print(i,json.dumps(future.result()),flush=True)
    write(work/'execution-complete.json', {'finished':timestamp(),'attempts':len(seeds)})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','run','self-test'])
    parser.add_argument('--work',default=str(Path(__file__).resolve().parent))
    parser.add_argument('--source')
    parser.add_argument('--data')
    parser.add_argument('--old')
    args=parser.parse_args()
    if args.action=='self-test':self_test()
    elif args.action=='prepare':prepare(args)
    else:run(args)
