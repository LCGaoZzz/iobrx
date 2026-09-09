"""Quantify Salmon differences without changing outputs or declaring tolerance.

All cross-arm pairs and all within-arm repeat pairs are retained. Transcript
identifiers must match; rows are aligned by identifier solely for comparison.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def describe(left, right):
    if not left.index.is_unique or not right.index.is_unique or set(left.index) != set(right.index):
        raise ValueError('Transcript identifiers differ or are not unique')
    right = right.loc[left.index]
    result = {}
    for column in ('Length', 'EffectiveLength', 'TPM', 'NumReads'):
        a, b = left[column].to_numpy(), right[column].to_numpy()
        delta = np.abs(a - b)
        result[column] = {'different_values': int(np.count_nonzero(delta)),
                          'max_absolute_difference': float(delta.max()),
                          'mean_absolute_difference': float(delta.mean()),
                          'l1_difference': float(delta.sum()),
                          'pearson_r': float(np.corrcoef(a, b)[0, 1])}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir', type=Path, required=True)
    args = parser.parse_args()
    runs = json.loads((args.case_dir / 'runs.json').read_text())
    loaded = {}
    for run in runs:
        key = (run['arm'], run['repeat'])
        loaded[key] = {p.parent.name: pd.read_csv(p, sep='\t', index_col='Name')
                       for p in Path(run['output']).rglob('quant.sf')}
    records = []
    for a, b in itertools.combinations(loaded, 2):
        group = 'between_arms' if a[0] != b[0] else f'within_{a[0]}'
        for sample in loaded[a]:
            records.append({'group': group, 'left': a, 'right': b, 'sample': sample,
                            'columns': describe(loaded[a][sample], loaded[b][sample])})
    report = {'interpretation': 'Observed differences; no post-hoc pass tolerance. Multithreaded within-arm variability is measured separately.',
              'comparisons': records}
    (args.case_dir / 'numeric_variation.json').write_text(json.dumps(report, indent=2) + '\n')
    for group in sorted(set(r['group'] for r in records)):
        selected = [r for r in records if r['group'] == group]
        print(group, 'max TPM difference:', max(r['columns']['TPM']['max_absolute_difference'] for r in selected),
              'minimum TPM correlation:', min(r['columns']['TPM']['pearson_r'] for r in selected))


if __name__ == '__main__':
    main()
