#!/usr/bin/env python3
"""Build results_ledger.csv: one row per (table, run, task, metric) from every results_*.json.
Machine-readable index of ALL evaluation artifacts (review/repro-eng.md P0 item)."""
import csv, json, glob, os

rows = []
for path in sorted(glob.glob('results_v2*/**/results_*.json', recursive=True) +
                   glob.glob('results_v2*/results_*.json')):
    table = path.split('/')[0]
    parts = path.split('/')
    run = parts[1] if len(parts) > 3 and parts[-2] != parts[0] else '_root'
    run = os.path.dirname(path).replace(table + '/', '').replace(table, '_root') or '_root'
    try:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
    except Exception as e:
        rows.append([table, run, 'PARSE_ERROR', str(e)[:60], '', '', path])
        continue
    ns = d.get('n-samples', {}) or {}
    for task, metrics in (d.get('results', {}) or {}).items():
        n_eff = (ns.get(task) or {}).get('effective', '')
        for mk, mv in metrics.items():
            if isinstance(mv, (int, float)):
                rows.append([table, run, task, mk, round(float(mv), 6), n_eff, path])

with open('results_ledger.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['table', 'run', 'task', 'metric', 'value', 'n_samples', 'source_file'])
    w.writerows(rows)
print(f'ledger rows: {len(rows)}')
