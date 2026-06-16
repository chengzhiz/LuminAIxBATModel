#!/usr/bin/env python3
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_DIR = ROOT / "FloorSupport" / "dataset" / "floor_support_downloaded"
CSV_DIR = ROOT / "FloorSupport" / "dataset" / "floor_support_raw"

out = {}
json_files = list(JSON_DIR.rglob('*.json'))
csv_files = list(CSV_DIR.rglob('*.csv'))

# Precompute name stems and numeric ids
def stem(p: Path):
    return p.stem.lower()

def numeric_tokens(s: str):
    return re.findall(r"\d+", s)

json_index = {p: stem(p) for p in json_files}
csv_index = {p: stem(p) for p in csv_files}

# Invert maps for quick lookup by stem
csv_by_stem = {}
for p, s in csv_index.items():
    csv_by_stem.setdefault(s, []).append(str(p.relative_to(ROOT)))

# Also build list for substring search
csv_paths = [str(p.relative_to(ROOT)) for p in csv_files]

mappings = []
for jp in json_files:
    jstem = json_index[jp]
    cand = set()
    # exact stem match
    if jstem in csv_by_stem:
        cand.update(csv_by_stem[jstem])
    # substring match
    for cp in csv_paths:
        if jstem in cp.lower():
            cand.add(cp)
    # numeric token intersection
    jnums = set(numeric_tokens(jp.name))
    if jnums:
        for p in csv_files:
            pnums = set(numeric_tokens(p.name))
            if jnums & pnums:
                cand.add(str(p.relative_to(ROOT)))

    mappings.append({
        'json': str(jp.relative_to(ROOT)),
        'candidates': sorted(cand),
        'candidate_count': len(cand)
    })

# Sort by candidate_count desc
mappings_sorted = sorted(mappings, key=lambda x: x['candidate_count'], reverse=True)

SAMPLE_N = 20
sample = mappings_sorted[:SAMPLE_N]

summary = {
    'total_json': len(json_files),
    'total_csv': len(csv_files),
    'matched_json_with_candidates': sum(1 for m in mappings if m['candidate_count']>0),
    'unmatched_json': sum(1 for m in mappings if m['candidate_count']==0),
}

out_data = {
    'summary': summary,
    'sample_mappings': sample,
}

out_file = ROOT / 'mapping_sample.json'
with out_file.open('w', encoding='utf-8') as f:
    json.dump(out_data, f, indent=2)

print(f"Wrote sample to {out_file}")
print(json.dumps(summary, indent=2))
print('\nSample entries:')
for item in sample:
    print('-', item['json'], '->', item['candidate_count'], 'candidates')
    for c in item['candidates'][:5]:
        print('   ', c)

print('\nDone')
