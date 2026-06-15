import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import AuditDataset
from src.metrics import compute_metrics, format_multiclass_report
from src.model import CNNBiLSTMAttention
from src.plotting import plot_confusion_matrix, plot_roc_curves
from src.utils import get_device, ensure_dir, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/best_model.pt")
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--plot_dir", default="plots")
    args = ap.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("config", {})
    class_names = ckpt.get("class_names", cfg.get("class_names", [0, 1, 2, 3, 4]))
    num_classes = len(class_names)
    input_dim = ckpt.get("input_dim", cfg.get("data", {}).get("input_dim", 19))
    seq_len = ckpt.get("seq_len", cfg.get("data", {}).get("seq_len", 20))
    model_cfg = cfg.get("model", {})

    ds = AuditDataset(args.data, seq_len=seq_len, input_dim=input_dim)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = CNNBiLSTMAttention(
        input_dim=input_dim,
        embed_dim=model_cfg.get("embed_dim", 64),
        conv_channels=model_cfg.get("conv_channels", 96),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_classes=num_classes,
        dropout=model_cfg.get("dropout", 0.3),
    ).to(device)

    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()

    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device).float()
            logits = model(x)
            prob = torch.softmax(logits, dim=1)
            y_true.extend(y.numpy().tolist())
            y_pred.extend(torch.argmax(prob, dim=1).cpu().numpy().tolist())
            y_prob.extend(prob.cpu().numpy().tolist())

    metrics = compute_metrics(y_true, y_pred, np.asarray(y_prob), class_names)

    ensure_dir(args.output_dir)
    ensure_dir(args.plot_dir)
    save_json(metrics, Path(args.output_dir) / "test_metrics.json")
    save_json(metrics.get("per_class", {}), Path(args.output_dir) / "per_class_metrics.json")
    save_json(metrics.get("prediction_distribution", {}), Path(args.output_dir) / "prediction_distribution.json")
    save_json(metrics.get("ground_truth_distribution", {}), Path(args.output_dir) / "ground_truth_distribution.json")

    plot_confusion_matrix(metrics["confusion_matrix"], class_names, Path(args.plot_dir) / "confusion_matrix.png")
    plot_roc_curves(metrics["roc"], Path(args.plot_dir) / "roc_curve.png")

    report_text = format_multiclass_report(metrics, class_names)
    print(report_text)
    with open(Path(args.output_dir) / "multiclass_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text + "\n")


if __name__ == "__main__":
    main()
