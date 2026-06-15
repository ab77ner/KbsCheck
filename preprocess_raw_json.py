import argparse
import json
from pathlib import Path
from src.features import parse_audit_event, build_sequences


def read_events(path):
    events=[]
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            ev=parse_audit_event(line)
            if ev is not None: events.append(ev)
    events.sort(key=lambda x: x.get('timestamp',0.0))
    return events


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--seq_len', type=int, default=20)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--label', type=int, default=None)
    args=ap.parse_args()
    events=read_events(args.input)
    rows=build_sequences(events, seq_len=args.seq_len, stride=args.stride, default_label=args.label)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output,'w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+'\n')
    print(f'events={len(events)}, samples={len(rows)}, output={args.output}')

if __name__ == '__main__':
    main()
