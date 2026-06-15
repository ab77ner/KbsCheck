#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
周期性转换新增 Kubernetes 审计日志，并用新增样本微调模型生成新权重。

流程：
1. 调用 convert_new_audit_to_dataset.py；
2. 检查新增样本数是否达到阈值；
3. 加载 base_ckpt 或 latest_finetuned_model.pt；
4. 使用新增样本进行少量 epoch 微调；
5. 保存 finetuned_model_*.pt 和 latest_finetuned_model.pt。

示例：
python scheduled_finetune.py \
  --base_ckpt outputs/best_model.pt \
  --raw_dir realtime/raw \
  --dataset_path realtime/dataset/new_samples.jsonl \
  --interval_seconds 600 \
  --min_new_samples 128 \
  --finetune_epochs 3
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.dataset import AuditDataset
from src.model import CNNBiLSTMAttention
from src.utils import get_device, ensure_dir, save_json


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def count_jsonl(path: Path):
    if not path.exists():
        return 0
    n = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return n


def run_converter(args):
    cmd = [
        sys.executable, "convert_new_audit_to_dataset.py",
        "--input_dir", args.raw_dir,
        "--output", args.dataset_path,
        "--state_file", args.converter_state,
        "--seq_len", str(args.seq_len),
        "--stride", str(args.stride),
        "--min_events", str(args.seq_len),
    ]
    if args.force_label is not None:
        cmd.extend(["--label", str(args.force_label)])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.returncode != 0:
        print(proc.stderr.strip())
    return proc.returncode == 0


def load_model(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    class_names = ckpt.get("class_names", cfg.get("class_names", [0, 1, 2, 3, 4]))
    input_dim = ckpt.get("input_dim", data_cfg.get("input_dim", 19))
    seq_len = ckpt.get("seq_len", data_cfg.get("seq_len", 20))

    model = CNNBiLSTMAttention(
        input_dim=input_dim,
        embed_dim=model_cfg.get("embed_dim", 64),
        conv_channels=model_cfg.get("conv_channels", 96),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_classes=len(class_names),
        dropout=model_cfg.get("dropout", 0.3),
    ).to(device)

    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    return model, ckpt, class_names, input_dim, seq_len


def finetune_once(args, ckpt_path: Path, device, last_seen_count: int):
    total_count = count_jsonl(Path(args.dataset_path))
    new_count = total_count - last_seen_count

    if new_count < args.min_new_samples:
        print(json.dumps({
            "status": "skip_finetune",
            "new_samples_since_last": new_count,
            "min_new_samples": args.min_new_samples,
            "total_samples": total_count
        }, ensure_ascii=False))
        return ckpt_path, last_seen_count

    model, ckpt, class_names, input_dim, seq_len = load_model(ckpt_path, device)

    ds = AuditDataset(args.dataset_path, seq_len=seq_len, input_dim=input_dim)
    start = max(0, len(ds) - new_count)
    subset = Subset(ds, list(range(start, len(ds))))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(1, args.finetune_epochs + 1):
        batch_losses = []
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).long()
            logits = model(x)
            loss = criterion(logits, y)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batch_losses.append(float(loss.item()))
        avg_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        losses.append(avg_loss)
        print(json.dumps({"finetune_epoch": epoch, "loss": avg_loss}, ensure_ascii=False))

    ensure_dir(args.output_dir)
    new_ckpt = dict(ckpt)
    new_ckpt["model_state"] = model.state_dict()
    new_ckpt["finetune_info"] = {
        "created_at": utc_now(),
        "base_ckpt": str(ckpt_path),
        "dataset_path": args.dataset_path,
        "new_samples": new_count,
        "epochs": args.finetune_epochs,
        "losses": losses,
    }

    out_path = Path(args.output_dir) / f"finetuned_model_{utc_now()}.pt"
    latest_path = Path(args.output_dir) / "latest_finetuned_model.pt"

    torch.save(new_ckpt, out_path)
    torch.save(new_ckpt, latest_path)
    save_json(new_ckpt["finetune_info"], Path(args.output_dir) / "latest_finetune_info.json")

    print(json.dumps({
        "status": "finetuned",
        "new_ckpt": str(out_path),
        "latest_ckpt": str(latest_path),
        "new_samples": new_count
    }, ensure_ascii=False, indent=2))

    return latest_path, total_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", default="outputs/best_model.pt")
    ap.add_argument("--raw_dir", default="realtime/raw")
    ap.add_argument("--dataset_path", default="realtime/dataset/new_samples.jsonl")
    ap.add_argument("--converter_state", default="realtime/converter_state.json")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--interval_seconds", type=int, default=600)
    ap.add_argument("--min_new_samples", type=int, default=128)
    ap.add_argument("--finetune_epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--force_label", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--run_once", action="store_true")
    args = ap.parse_args()

    device = get_device(args.device)
    current_ckpt = Path(args.base_ckpt)
    last_seen = count_jsonl(Path(args.dataset_path))

    while True:
        if run_converter(args):
            current_ckpt, last_seen = finetune_once(args, current_ckpt, device, last_seen)

        if args.run_once:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
