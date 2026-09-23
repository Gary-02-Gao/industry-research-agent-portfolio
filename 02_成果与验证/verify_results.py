#!/usr/bin/env python3
"""Verify portfolio integrity and aggregate saved scoring decisions; no network."""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / '02_成果与验证'

# Local environments and repository metadata are not delivery artifacts.
LOCAL_DIRS = {'.git', '.venv', '.runtime', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', 'node_modules'}

def delivery_files():
    for directory, dirs, files in os.walk(ROOT, followlinks=False):
        for name in dirs:
            path = Path(directory) / name
            require(not path.is_symlink(), 'Linked directory: ' + str(path.relative_to(ROOT)))
        dirs[:] = [name for name in dirs if name not in LOCAL_DIRS]
        for name in files:
            path = Path(directory) / name
            if name == '.DS_Store' or name.endswith('.pyc') or path == ROOT / 'MANIFEST.sha256':
                continue
            yield str(path.relative_to(ROOT))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read(name):
    return json.loads((EVIDENCE / name).read_text(encoding='utf-8'))

def main():
    manifest = ROOT / 'MANIFEST.sha256'
    require(manifest.is_file(), 'MANIFEST.sha256 is missing')
    expected = set()
    for line in manifest.read_text(encoding='utf-8').splitlines():
        digest, rel = line.split('  ', 1)
        path = ROOT / rel
        require(not Path(rel).is_absolute() and '..' not in Path(rel).parts, 'Unsafe manifest path')
        require(path.is_file() and not path.is_symlink(), 'Missing or linked file: ' + rel)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, 'Hash mismatch: ' + rel)
        require(rel not in expected, 'Duplicate manifest entry: ' + rel)
        expected.add(rel)
    actual = set(delivery_files())
    require(actual == expected, 'Manifest file set mismatch; extra=' + str(sorted(actual - expected)) + '; missing=' + str(sorted(expected - actual)))
    for row in read('source-manifest.json')['files']:
        require(hashlib.sha256((ROOT / row['delivery']).read_bytes()).hexdigest() == row['sha256'], 'Source snapshot mismatch')
    dataset = read('dataset-summary.json')
    targets = {'single_document_fact': 40, 'same_document_multi_hop': 30, 'cross_document_comparison': 20, 'unanswerable': 10}
    require(dataset['category_counts'] == targets and dataset['case_count'] == 100, 'Dataset category counts mismatch')
    require({k:v['count'] for k,v in dataset['splits'].items()} == {'dev':40,'validation':30,'test':30}, 'Split counts mismatch')
    for category,total in targets.items():
        require(sum(v['category_counts'][category] for v in dataset['splits'].values()) == total, 'Split category total mismatch')
    for split in dataset['splits'].values():
        require(sum(split['category_counts'].values()) == split['count'], 'Split quota mismatch')
    retrieval = read('retrieval-audit.json')['cases']
    require(len(retrieval)==20 and len({r['row'] for r in retrieval})==20, 'Retrieval row count mismatch')
    answerable = [r for r in retrieval if r['answerable']]
    require(len(answerable)==18, 'Retrieval denominator mismatch')
    for row in answerable:
        require(0 <= row['covered_evidence_at_10'] <= row['gold_evidence_count'] and row['gold_evidence_count'] > 0, 'Invalid evidence counts')
    recall = sum(r['covered_evidence_at_10']/r['gold_evidence_count'] for r in answerable)/len(answerable)
    precision = sum(r['first_result_relevant'] for r in answerable)/len(answerable)
    summary = read('public-evaluation.json')
    require(math.isclose(recall, 71/72, abs_tol=1e-12), 'Recall aggregate mismatch')
    require(math.isclose(precision, 13/18, abs_tol=1e-12), 'Precision aggregate mismatch')
    require(abs(recall-summary['retrieval']['recall_at_10']['candidate']) < 0.0005, 'Published recall mismatch')
    route = read('routing-audit.json')['cases']
    require(len(route)==200 and len({r['row'] for r in route})==200, 'Routing row count mismatch')
    correct = sum(r['selection_exact'] for r in route)
    valid = sum(r['schema_valid'] for r in route)
    require(correct==193 and valid==200, 'Routing score mismatch')
    require(all(r['forbidden_tool_count']==0 for r in route), 'Forbidden selection observed')
    require(correct/len(route)==summary['routing']['metrics']['tool_selection_accuracy'], 'Published route mismatch')
    print(json.dumps({'integrity_files':len(expected),'dataset_cases':100,'retrieval_cases':20,'retrieval_answerable':18,'recall_at_10':round(recall,6),'precision_at_1':round(precision,6),'tool_selection':f'{correct}/200','schema_valid':f'{valid}/200','status':'PASS','scope':'Saved-statistic verification; no model inference or semantic rejudging.'}, ensure_ascii=False,indent=2))

if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print('FAIL: '+str(exc), file=sys.stderr)
        raise SystemExit(1)
