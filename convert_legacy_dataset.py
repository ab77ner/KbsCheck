import argparse
import json
import random
from pathlib import Path
from collections import defaultdict, Counter
from src.features import sample_to_features, LABELS


def read_jsonl(path):
    rows=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            try: rows.append(json.loads(line))
            except Exception: continue
    return rows


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def stratified_val_split(rows, val_ratio=0.10, seed=42):
    random.seed(seed)
    groups=defaultdict(list)
    for r in rows:
        groups[r.get('label_name', str(r.get('label')))].append(r)
    train=[]; val=[]
    for _, items in groups.items():
        random.shuffle(items)
        n_val=max(1, int(len(items)*val_ratio)) if len(items)>1 else 0
        val.extend(items[:n_val]); train.extend(items[n_val:])
    random.shuffle(train); random.shuffle(val)
    return train, val


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train_raw', required=True)
    ap.add_argument('--test_raw', required=True)
    ap.add_argument('--out_dir', default='data')
    ap.add_argument('--seq_len', type=int, default=20)
    ap.add_argument('--val_ratio', type=float, default=0.10)
    args=ap.parse_args()
    train_raw=read_jsonl(args.train_raw)
    test_raw=read_jsonl(args.test_raw)
    train_converted=[sample_to_features(x, seq_len=args.seq_len) for x in train_raw]
    test_converted=[sample_to_features(x, seq_len=args.seq_len) for x in test_raw]
    train_final, val_final = stratified_val_split(train_converted, args.val_ratio)
    out=Path(args.out_dir)
    write_jsonl(train_final, out/'train.jsonl')
    write_jsonl(val_final, out/'val.jsonl')
    write_jsonl(test_converted, out/'test.jsonl')
    meta={
        'labels': LABELS,
        'train': len(train_final),
        'val': len(val_final),
        'test': len(test_converted),
        'train_distribution': Counter([r['label_name'] for r in train_final]),
        'val_distribution': Counter([r['label_name'] for r in val_final]),
        'test_distribution': Counter([r['label_name'] for r in test_converted]),
    }
    with open(out/'dataset_summary.json','w',encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
