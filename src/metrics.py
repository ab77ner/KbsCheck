import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report,
)
from sklearn.preprocessing import label_binarize


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def compute_metrics(y_true, y_pred, y_prob, class_names):
    """
    Multi-class evaluation.

    Returns:
      - aggregate metrics: accuracy, weighted/macro/micro precision/recall/f1, auc
      - per_class: per-class precision/recall/f1/support
      - confusion_matrix: fixed-label confusion matrix
      - prediction_distribution: predicted count and ratio per class
      - ground_truth_distribution: true count and ratio per class
      - roc: one-vs-rest ROC curves per class
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    num_classes = len(class_names)
    labels = list(range(num_classes))
    class_names_str = [str(x) for x in class_names]

    out = {}

    # Aggregate metrics
    out["accuracy"] = _safe_float(accuracy_score(y_true, y_pred))
    out["precision_weighted"] = _safe_float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    out["recall_weighted"] = _safe_float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    out["f1_weighted"] = _safe_float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))

    # Backward-compatible aliases
    out["precision"] = out["precision_weighted"]
    out["recall"] = out["recall_weighted"]
    out["f1"] = out["f1_weighted"]

    out["precision_macro"] = _safe_float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    out["recall_macro"] = _safe_float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    out["f1_macro"] = _safe_float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    out["precision_micro"] = _safe_float(precision_score(y_true, y_pred, labels=labels, average="micro", zero_division=0))
    out["recall_micro"] = _safe_float(recall_score(y_true, y_pred, labels=labels, average="micro", zero_division=0))
    out["f1_micro"] = _safe_float(f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out["confusion_matrix"] = cm.tolist()

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names_str,
        zero_division=0,
        output_dict=True,
    )
    out["classification_report"] = report

    # Explicit per-class metrics
    per_class = {}
    for idx, name in enumerate(class_names_str):
        r = report.get(name, {})
        per_class[name] = {
            "label_id": idx,
            "precision": _safe_float(r.get("precision", 0.0)),
            "recall": _safe_float(r.get("recall", 0.0)),
            "f1": _safe_float(r.get("f1-score", 0.0)),
            "support": int(r.get("support", 0)),
            "true_count": int(np.sum(y_true == idx)),
            "pred_count": int(np.sum(y_pred == idx)),
        }
    out["per_class"] = per_class

    # Distributions
    total_true = max(1, len(y_true))
    total_pred = max(1, len(y_pred))
    out["ground_truth_distribution"] = {
        class_names_str[i]: {
            "count": int(np.sum(y_true == i)),
            "ratio": _safe_float(np.sum(y_true == i) / total_true),
        }
        for i in labels
    }
    out["prediction_distribution"] = {
        class_names_str[i]: {
            "count": int(np.sum(y_pred == i)),
            "ratio": _safe_float(np.sum(y_pred == i) / total_pred),
        }
        for i in labels
    }

    # AUC and ROC, one-vs-rest
    roc_data = {}
    auc = 0.5
    try:
        if len(np.unique(y_true)) > 1 and y_prob.ndim == 2 and y_prob.shape[1] == num_classes:
            y_bin = label_binarize(y_true, classes=labels)

            # For binary label_binarize returns [n, 1]; keep multi-class logic robust
            if num_classes == 2 and y_bin.shape[1] == 1:
                auc = roc_auc_score(y_true, y_prob[:, 1])
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                roc_data[class_names_str[1]] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            else:
                auc = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
                for i, name in enumerate(class_names_str):
                    # ROC undefined if a class has no positives in this split
                    if np.sum(y_bin[:, i]) == 0:
                        continue
                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                    roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    except Exception as e:
        roc_data["warning"] = str(e)
        auc = 0.5

    out["auc"] = _safe_float(auc)
    out["roc"] = roc_data
    return out


def format_multiclass_report(metrics, class_names=None):
    """
    Create a human-readable multi-class report string for terminal output.
    """
    if class_names is None:
        class_names = list(metrics.get("per_class", {}).keys())

    lines = []
    lines.append("========== Overall Metrics ==========")
    lines.append(f"Accuracy          : {metrics.get('accuracy', 0):.6f}")
    lines.append(f"Precision(weight) : {metrics.get('precision_weighted', metrics.get('precision', 0)):.6f}")
    lines.append(f"Recall(weight)    : {metrics.get('recall_weighted', metrics.get('recall', 0)):.6f}")
    lines.append(f"F1(weight)        : {metrics.get('f1_weighted', metrics.get('f1', 0)):.6f}")
    lines.append(f"Precision(macro)  : {metrics.get('precision_macro', 0):.6f}")
    lines.append(f"Recall(macro)     : {metrics.get('recall_macro', 0):.6f}")
    lines.append(f"F1(macro)         : {metrics.get('f1_macro', 0):.6f}")
    lines.append(f"AUC(macro OVR)    : {metrics.get('auc', 0.5):.6f}")
    lines.append("")
    lines.append("========== Per-Class Metrics ==========")
    lines.append(f"{'Class':<28} {'ID':>3} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>9} {'Pred':>9}")
    lines.append("-" * 86)

    per_class = metrics.get("per_class", {})
    for name, row in per_class.items():
        lines.append(
            f"{name:<28} {int(row.get('label_id', 0)):>3} "
            f"{row.get('precision', 0):>10.4f} {row.get('recall', 0):>10.4f} "
            f"{row.get('f1', 0):>10.4f} {int(row.get('support', 0)):>9} {int(row.get('pred_count', 0)):>9}"
        )

    lines.append("")
    lines.append("========== Confusion Matrix ==========")
    cm = metrics.get("confusion_matrix", [])
    if cm:
        header = "true\\pred".ljust(12) + "".join([str(i).rjust(8) for i in range(len(cm))])
        lines.append(header)
        for i, row in enumerate(cm):
            lines.append(str(i).ljust(12) + "".join([str(int(v)).rjust(8) for v in row]))

    return "\n".join(lines)
