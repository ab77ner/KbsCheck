#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将新增 Kubernetes 原始审计日志转换为本文模型的数据集输入格式。

输入：
  realtime/raw/*.jsonl

输出：
  realtime/dataset/new_samples.jsonl

每行格式：
{
  "features": [[...], [...], ...],
  "label": int,
  "meta": {...}
}

示例：
python convert_new_audit_to_dataset.py \
  --input_dir realtime/raw \
  --output realtime/dataset/new_samples.jsonl \
  --state_file realtime/converter_state.json \
  --seq_len 20 \
  --stride 1
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from src.features import parse_audit_event, event_to_feature, rule_label_event


def load_state(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(path: Path, state: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def iter_files(input_dir: Path, pattern: str):
    return sorted(input_dir.glob(pattern), key=lambda p: (p.stat().st_mtime, str(p)))


def read_new_events(input_file: Path, offset: int):
    events = []
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(offset)
        while True:
            line = f.readline()
            if not line:
                break
            pos = f.tell()
            ev = parse_audit_event(line.strip())
            if ev:
                ev["_source_file"] = str(input_file)
                ev["_source_offset"] = pos
                events.append(ev)
        new_offset = f.tell()
    return events, new_offset


def build_rows(events: List[Dict], seq_len: int, stride: int, force_label: Optional[int]):
    rows = []
    if len(events) < seq_len:
        return rows

    events = sorted(events, key=lambda x: x.get("timestamp", 0.0))
    features = []
    prev = None
    for ev in events:
        features.append(event_to_feature(ev, prev))
        prev = ev

    for i in range(0, len(features) - seq_len + 1, stride):
        seq_feats = features[i:i + seq_len]
        seq_events = events[i:i + seq_len]

        if force_label is not None:
            label = int(force_label)
        else:
            # 规则伪标签：窗口内若出现高风险事件，则取最高风险类别
            labels = [int(rule_label_event(e)) for e in seq_events]
            label = max(labels) if labels else 0

        rows.append({
            "features": seq_feats,
            "label": label,
            "meta": {
                "event_count": len(seq_events),
                "first_ts": seq_events[0].get("timestamp", 0.0),
                "last_ts": seq_events[-1].get("timestamp", 0.0),
                "source_file_start": seq_events[0].get("_source_file", ""),
                "source_file_end": seq_events[-1].get("_source_file", ""),
                "source_offset_start": seq_events[0].get("_source_offset", 0),
                "source_offset_end": seq_events[-1].get("_source_offset", 0),
                "users": sorted(list({str(e.get("user", "unknown")) for e in seq_events}))[:20],
                "resources": sorted(list({str(e.get("resource", "unknown")) for e in seq_events}))[:20],
            }
        })
    return rows


def append_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="realtime/raw")
    ap.add_argument("--pattern", default="*.jsonl")
    ap.add_argument("--output", default="realtime/dataset/new_samples.jsonl")
    ap.add_argument("--state_file", default="realtime/converter_state.json")
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--label", type=int, default=None, help="指定所有新增样本标签；不指定时使用规则伪标签")
    ap.add_argument("--min_events", type=int, default=20)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    state_file = Path(args.state_file)
    state = load_state(state_file)

    all_events = []
    touched_files = 0

    for fp in iter_files(input_dir, args.pattern):
        key = str(fp.resolve())
        old_offset = int(state.get(key, {}).get("offset", 0))
        events, new_offset = read_new_events(fp, old_offset)
        if new_offset != old_offset:
            state[key] = {"offset": new_offset}
        if events:
            all_events.extend(events)
            touched_files += 1

    if len(all_events) < args.min_events:
        save_state(state_file, state)
        print(json.dumps({
            "status": "not_enough_new_events",
            "new_events": len(all_events),
            "min_events": args.min_events,
            "touched_files": touched_files
        }, ensure_ascii=False))
        return

    rows = build_rows(all_events, args.seq_len, args.stride, args.label)
    append_jsonl(Path(args.output), rows)
    save_state(state_file, state)

    counts = {}
    for r in rows:
        k = str(r["label"])
        counts[k] = counts.get(k, 0) + 1

    print(json.dumps({
        "status": "converted",
        "new_events": len(all_events),
        "new_samples": len(rows),
        "output": args.output,
        "label_counts": counts
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
