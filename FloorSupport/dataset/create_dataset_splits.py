import random
import shutil
from pathlib import Path

root = Path('FloorSupport_raw')
classes = ['d_ft_dual_feet', 'd_hn_dual_hand', 's_ft_single_feet']

for split in ['train', 'val', 'test']:
    for cls in classes:
        (root / split / cls).mkdir(parents=True, exist_ok=True)

random.seed(42)
for cls in classes:
    src_dir = root / cls / 'bodyframes'
    files = sorted([str(p) for p in src_dir.glob('*.csv')])
    if not files:
        continue

    random.shuffle(files)
    n = len(files)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val

    splits = {
        'train': files[:n_train],
        'val': files[n_train:n_train + n_val],
        'test': files[n_train + n_val:n_train + n_val + n_test],
    }

    for split, split_files in splits.items():
        dst_dir = root / split / cls
        for src_file in split_files:
            shutil.copy2(src_file, dst_dir / Path(src_file).name)

    print(f"{cls}: total={n}, train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
