#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时记录并保存 Kubernetes 原始审计日志。

功能：
- tail Kubernetes API Server audit.log；
- 持续保存新增 JSON 审计日志到 realtime/raw；
- 支持 JSON 行校验；
- 支持 offset 状态保存，重启后续采；
- 支持 audit.log 轮转；
- 支持输出文件按大小滚动。

示例：
python collect_k8s_audit_logs.py \
  --source /var/log/kubernetes/audit.log \
  --out_dir realtime/raw \
  --state_file realtime/collector_state.json
"""

import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_state(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def get_inode(path: Path):
    try:
        return os.stat(path).st_ino
    except FileNotFoundError:
        return None


def is_valid_json_line(line: str):
    try:
        obj = json.loads(line)
        return isinstance(obj, dict)
    except Exception:
        return False


class RollingWriter:
    def __init__(self, out_dir: Path, prefix="audit_raw", max_mb=128):
        self.out_dir = out_dir
        self.prefix = prefix
        self.max_bytes = max_mb * 1024 * 1024
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fp = None
        self.path = None
        self.open_new()

    def open_new(self):
        if self.fp:
            self.fp.close()
        self.path = self.out_dir / f"{self.prefix}_{utc_now()}.jsonl"
        self.fp = open(self.path, "a", encoding="utf-8")

    def write(self, line: str):
        if self.fp.tell() >= self.max_bytes:
            self.open_new()
        self.fp.write(line.rstrip("\n") + "\n")
        self.fp.flush()

    def close(self):
        if self.fp:
            self.fp.close()


def collect(source: Path, out_dir: Path, state_file: Path, poll_interval: float, from_beginning: bool, validate_json: bool, max_mb: int):
    state = load_state(state_file)
    writer = RollingWriter(out_dir, max_mb=max_mb)

    fp = None
    current_inode = None

    try:
        while True:
            if not source.exists():
                print(json.dumps({"event": "waiting_source", "source": str(source)}, ensure_ascii=False))
                time.sleep(poll_interval)
                continue

            inode = get_inode(source)
            if fp is None or inode != current_inode:
                if fp:
                    fp.close()
                fp = open(source, "r", encoding="utf-8", errors="ignore")
                current_inode = inode

                saved = state.get(str(source), {})
                saved_inode = saved.get("inode")
                saved_offset = int(saved.get("offset", 0))

                if saved_inode == inode:
                    fp.seek(saved_offset)
                elif from_beginning:
                    fp.seek(0)
                else:
                    fp.seek(0, os.SEEK_END)

                print(json.dumps({
                    "event": "open_source",
                    "source": str(source),
                    "inode": inode,
                    "offset": fp.tell(),
                    "output": str(writer.path)
                }, ensure_ascii=False))

            line = fp.readline()
            if not line:
                try:
                    if fp.tell() > source.stat().st_size:
                        fp.seek(0)
                except FileNotFoundError:
                    pass
                time.sleep(poll_interval)
                continue

            if validate_json and not is_valid_json_line(line):
                state[str(source)] = {"inode": current_inode, "offset": fp.tell(), "last_update": utc_now()}
                save_state(state_file, state)
                continue

            writer.write(line)
            state[str(source)] = {
                "inode": current_inode,
                "offset": fp.tell(),
                "last_update": utc_now(),
                "current_output": str(writer.path)
            }
            save_state(state_file, state)

    finally:
        if fp:
            fp.close()
        writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/var/log/kubernetes/audit.log")
    ap.add_argument("--out_dir", default="realtime/raw")
    ap.add_argument("--state_file", default="realtime/collector_state.json")
    ap.add_argument("--poll_interval", type=float, default=0.5)
    ap.add_argument("--from_beginning", action="store_true")
    ap.add_argument("--no_validate_json", action="store_true")
    ap.add_argument("--max_mb", type=int, default=128)
    args = ap.parse_args()

    collect(
        source=Path(args.source),
        out_dir=Path(args.out_dir),
        state_file=Path(args.state_file),
        poll_interval=args.poll_interval,
        from_beginning=args.from_beginning,
        validate_json=not args.no_validate_json,
        max_mb=args.max_mb,
    )


if __name__ == "__main__":
    main()
