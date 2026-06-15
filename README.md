注意：data/train.zip在本地需要解压缩到本地后使用，因为文件太大无法上传做了压缩处理

## 数据来源

- `data/raw_legacy/train_raw.jsonl`：训练样本原始格式
- `data/raw_legacy/test_raw.jsonl`：测试样本原始格式
- `data/attack_chain_library.json`：攻击链模式库
- `data/train.jsonl` / `data/val.jsonl` / `data/test.jsonl`：已转换为模型输入格式的特征序列

## 标签

8 类标签：

1. Normal
2. InitialAccess
3. PrivilegeEscalation
4. Persistence
5. LateralMovement
6. ContainerEscape
7. DataExfiltration
8. Misconfiguration

## 安装

```bash
pip install -r requirements.txt
```

## 训练

```bash
python train.py --config configs/default_config.yaml
```

可覆盖参数：

```bash
python train.py --epochs 1500 --batch_size 128 --learning_rate 5e-4
```

## 测试

```bash
python evaluate.py --ckpt outputs/best_model.pt --data data/test.jsonl
```

输出：

- `outputs/test_metrics.json`
- `plots/confusion_matrix.png`
- `plots/roc_curve.png`

## 处理原始 Kubernetes audit.log

```bash
python preprocess_raw_json.py --input audit.log --output data/custom.jsonl --seq_len 20
```

## 如果需要重新从旧格式数据转换

```bash
python convert_legacy_dataset.py   --train_raw data/raw_legacy/train_raw.jsonl   --test_raw data/raw_legacy/test_raw.jsonl   --out_dir data   --seq_len 20
```

## 实时推理

```bash
python stream_infer.py --ckpt outputs/best_model.pt --audit_log /var/log/kubernetes/audit.log
```

## 实时微调

```bash
python stream_finetune.py --ckpt outputs/best_model.pt --audit_log /var/log/kubernetes/audit.log
```


## 多分类结果输出

运行：

```bash
python evaluate.py --ckpt outputs/best_model.pt --data data/test.jsonl
```

会在终端显示：

- Overall Metrics：Accuracy、Precision/Recall/F1 的 weighted、macro、micro 汇总
- Per-Class Metrics：每个类别的 Precision、Recall、F1、Support、Pred Count
- Confusion Matrix：多分类混淆矩阵

同时生成：

```text
outputs/test_metrics.json
outputs/per_class_metrics.json
outputs/prediction_distribution.json
outputs/ground_truth_distribution.json
outputs/multiclass_report.txt
plots/confusion_matrix.png
plots/roc_curve.png
```

## 命令完整运行顺序

```bash
# 1. 离线训练与测试
pip install -r requirements.txt
python train.py --config configs/default_config.yaml
python evaluate.py --ckpt outputs/best_model.pt --data data/test.jsonl

# 2. 实时保存原始 Kubernetes 审计日志
python collect_k8s_audit_logs.py \
  --source /var/log/kubernetes/audit.log \
  --out_dir realtime/raw \
  --state_file realtime/collector_state.json

# 3. 将新增原始日志转换为数据集输入格式
python convert_new_audit_to_dataset.py \
  --input_dir realtime/raw \
  --output realtime/dataset/new_samples.jsonl \
  --state_file realtime/converter_state.json \
  --seq_len 20 \
  --stride 1

# 4. 周期微调并生成新权重
python scheduled_finetune.py \
  --base_ckpt outputs/best_model.pt \
  --raw_dir realtime/raw \
  --dataset_path realtime/dataset/new_samples.jsonl \
  --converter_state realtime/converter_state.json \
  --interval_seconds 600 \
  --min_new_samples 128 \
  --finetune_epochs 3

# 5. 使用新权重实时推理
python stream_infer.py \
  --ckpt outputs/latest_finetuned_model.pt \
  --audit_log /var/log/kubernetes/audit.log
```
