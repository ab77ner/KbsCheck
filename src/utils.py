import json
import random
from pathlib import Path
import numpy as np
import torch
try:
    import yaml
except Exception:
    yaml = None


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str = "auto"):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_config(path: str):
    if yaml is None:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_jsonl(path):
    rows=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            try: rows.append(json.loads(line))
            except Exception: continue
    return rows
