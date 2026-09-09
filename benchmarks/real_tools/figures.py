"""Small-multiple figures from measured repetitions and actual tool products."""
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = ('#818B92', '#237C79', '#CBD7D8')
TITLES = {'fastq_qc': 'FASTQ quality control', 'batch_salmon': 'Salmon quantification',
          'batch_star_count': 'STAR alignment', 'extract_hla_read': 'HLA read extraction',
          'spechla': 'HLA allele typing'}


def read_count(path):
    with gzip.open(path, 'rb') as stream:
        return sum(1 for _ in stream) // 4


def make_figure(case, output, baseline, revision=2):
    name = case['case']['function']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.labelsize': 9, 'axes.titlesize': 10, 'axes.titleweight': 'normal',
                         'axes.linewidth': .7, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, (ax, detail) = plt.subplots(1, 2, figsize=(10.4 if name == 'spechla' else 8.8, 3.5),
                                   gridspec_kw={'width_ratios': [1, 2.05 if name == 'spechla' else 1.3]})
    for index, arm in enumerate(('iobrpy', 'iobrx')):
        values = np.array([r['wall_seconds'] for r in case['runs'] if r['arm'] == arm])
        median = np.median(values)
        if revision == 0:
            ax.bar(index, median, color=COLORS[index], width=.6)
        else:
            ax.vlines(index, values.min(), values.max(), color=COLORS[index], lw=1.2)
            ax.scatter(index + np.linspace(-.10, .10, len(values)), values,
                       color=COLORS[index], s=22, zorder=3)
            ax.hlines(median, index - .18, index + .18, color='#26383B', lw=1.6)
        ax.text(index, values.max() * 1.035, f'{median:.2f} s', ha='center', va='bottom', fontsize=9)
    ax.set_xticks([0, 1], ['IOBRpy', 'iobrx'])
    ax.set_xlim(-.55, 1.55)
    ax.set_ylim(0, max(r['wall_seconds'] for r in case['runs']) * 1.25)
    ax.set_ylabel('Wall time (s)')
    ax.set_title('A   Fresh process', loc='left', pad=12)
    if revision == 2:
        ax.text(0, -.23, '3 alternating repetitions per arm\nPoints: runs; line: median; range: min–max',
                transform=ax.transAxes, fontsize=8, color='#556467')
    output, baseline = Path(output), Path(baseline)
    if name == 'fastq_qc':
        reports = sorted(output.glob('*_fastp.json'))
        labels, before, after = [], [], []
        for p in reports:
            data = json.loads(p.read_text())['summary']
            labels.append(p.name.split('_fastp')[0])
            before.append(data['before_filtering']['total_reads'] / 2000)
            after.append(data['after_filtering']['total_reads'] / 2000)
        x = np.arange(len(labels))
        detail.bar(x - .17, before, .32, color=COLORS[0], label='Input')
        detail.bar(x + .17, after, .32, color=COLORS[1], label='Passed')
        detail.set_xticks(x, labels)
        detail.set_ylabel('Read pairs (thousands)')
        detail.set_ylim(0, max(before) * 1.22)
        detail.legend(frameon=False, fontsize=8, ncol=2, loc='upper right')
        detail.set_title('B   Retained read pairs', loc='left', pad=12)
    elif name == 'batch_salmon':
        a = pd.read_csv(next(baseline.rglob('ERR188044/quant.sf')), sep='\t', index_col='Name')
        b = pd.read_csv(next(output.rglob('ERR188044/quant.sf')), sep='\t', index_col='Name').loc[a.index]
        detail.scatter(np.log10(1+a.TPM), np.log10(1+b.TPM), s=3, alpha=.23, color=COLORS[1], rasterized=True)
        lim = max(np.log10(1+a.TPM).max(), np.log10(1+b.TPM).max())
        detail.plot([0, lim], [0, lim], color='#A7B3B5', lw=.7, zorder=0)
        detail.set_xlabel('IOBRpy log₁₀(TPM + 1)')
        detail.set_ylabel('iobrx log₁₀(TPM + 1)')
        detail.set_title('B   ERR188044 · full transcript index', loc='left', pad=12)
        detail.text(.04, .96, f'Pearson r = {np.corrcoef(a.TPM, b.TPM)[0,1]:.6f}\n4-thread output is not byte-identical',
                    transform=detail.transAxes, va='top', fontsize=8)
    elif name == 'batch_star_count':
        labels, fractions = [], []
        for p in sorted(output.glob('*Log.final.out')):
            values = dict(line.strip().split('|', 1) for line in p.read_text().splitlines() if '|' in line)
            values = {k.strip(): v.strip() for k, v in values.items()}
            unique = float(values['Uniquely mapped reads %'].rstrip('%'))
            multi = float(values['% of reads mapped to multiple loci'].rstrip('%'))
            labels.append(p.name.split('_Log')[0]);fractions.append([unique, multi, 100 - unique - multi])
        bottom = np.zeros(len(labels))
        for i, label in enumerate(('Unique', 'Multiple loci', 'Other')):
            vals = np.array(fractions)[:, i]
            detail.bar(labels, vals, bottom=bottom, color=(COLORS[1], COLORS[0], COLORS[2])[i], width=.5, label=label)
            bottom += vals
        detail.set_ylim(0, 105)
        detail.set_ylabel('Input read pairs (%)')
        detail.legend(frameon=False, fontsize=8, loc='upper center',
                      bbox_to_anchor=(.5, -.13), ncol=3)
        detail.set_title('B   Yeast mapping outcomes', loc='left', pad=12)
    elif name == 'extract_hla_read':
        reads = sorted(output.rglob('*.fq.gz')) + sorted(output.rglob('*.fastq.gz'))
        paired = [p for p in reads if ('1.f' in p.name or 'R1' in p.name or '_1' in p.name)]
        count = read_count(paired[0] if paired else reads[0])
        detail.barh(['Input pairs', 'Extracted pairs'], [52771, count], color=COLORS[:2], height=.45)
        for y, n in enumerate((52771, count)):
            detail.text(n + 900, y, f'{n:,}', va='center', fontsize=9)
        detail.set_xlim(0, 67000)
        detail.set_xlabel('Read pairs')
        from matplotlib.ticker import StrMethodFormatter
        detail.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        detail.invert_yaxis()
        detail.set_title('B   NA06985 · chr6 alignment', loc='left', pad=12)
    else:
        rows = [line.split('\t') for line in next(output.rglob('hla.result.txt')).read_text().splitlines()
                if line and not line.startswith('#')]
        alleles = rows[1][1:]
        table = detail.table(cellText=[[alleles[i], alleles[i+1]] for i in range(0, 16, 2)],
                             colLabels=['Allele 1', 'Allele 2'], loc='center', cellLoc='left')
        table.auto_set_font_size(False);table.set_fontsize(8.6);table.scale(1, 1.45)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('white')
            cell.set_facecolor('#E7EFEF' if row == 0 else ('#F4F7F7' if row % 2 else 'white'))
        detail.set_axis_off()
        detail.set_title('B   NA06985 · exon-mode calls', loc='left', pad=12)
        detail.text(0, -.15, 'IPD-IMGT/HLA 3.38.0 · agreement with the original wrapper\nThis example does not establish genotype accuracy.',
                    transform=detail.transAxes, fontsize=8, color='#556467')
    for axes in (ax, detail):
        axes.spines[['top', 'right']].set_visible(False)
        for spine in ('left', 'bottom'):
            axes.spines[spine].set_color('#697679')
        axes.tick_params(width=.7, length=3)
        if revision == 0:
            axes.grid(axis='y', alpha=.2)
    fig.suptitle(TITLES[name], x=.07, y=1.02, ha='left', fontsize=12, fontweight='bold')
    fig.subplots_adjust(left=.08, right=.94, bottom=.25, top=.87, wspace=.45)
    return fig
