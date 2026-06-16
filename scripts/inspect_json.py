#!/usr/bin/env python3
import json, re, sys
from pathlib import Path

def summarize(p):
    text=Path(p).read_text(encoding='utf-8')
    text = re.sub(r'ISODate\("(.*?)"\)', r'"\1"', text)
    text = re.sub(r'\bNaN\b', 'null', text)
    obj=json.loads(text)
    print('FILE:', p)
    print('\nTop-level keys and types:')
    for k in sorted(obj.keys()):
        v=obj[k]
        if isinstance(v, list):
            info=f'list(len={len(v)})'
            if v:
                info += f', first_type={type(v[0]).__name__}'
        elif isinstance(v, dict):
            info=f'dict(len={len(v)})'
        else:
            info=f'{type(v).__name__}'
        print(f' - {k}: {info}')
    if 'bodyFrames' in obj and isinstance(obj['bodyFrames'], list) and obj['bodyFrames']:
        bf = obj['bodyFrames'][0]
        print('\nSample bodyFrames[0] keys and types:')
        if isinstance(bf, dict):
            for kk in sorted(bf.keys()):
                vv = bf[kk]
                t = type(vv).__name__
                s = f' - {kk}: {t}'
                if isinstance(vv, list): s += f' (len={len(vv)})'
                if isinstance(vv, dict): s += f' (len={len(vv)})'
                print(s)
            if 'bodyFrameHuman' in bf and isinstance(bf['bodyFrameHuman'], list):
                print('\nFirst 6 entries in bodyFrameHuman:')
                for entry in bf['bodyFrameHuman'][:6]:
                    if isinstance(entry, dict):
                        k_val = entry.get('k')
                        v_val = entry.get('v')
                        print(f'  k={k_val}, v_len={len(v_val) if isinstance(v_val, list) else "?"}')

if __name__=='__main__':
    p=sys.argv[1] if len(sys.argv)>1 else 'FloorSupport/dataset/floor_support_downloaded/FloorSupport/D-HN_Dual-Hand/D-HN_C1/66b10a3d9c7fc5c19aa98097.json'
    summarize(p)
