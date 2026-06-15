import json
import numpy as np
import torch
from torch.utils.data import Dataset


class AuditDataset(Dataset):
    def __init__(self, path: str, seq_len: int = 20, input_dim: int = 19, strict: bool = False):
        self.path = path
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                    arr = np.asarray(item['features'], dtype=np.float32)
                    if arr.ndim != 2:
                        raise ValueError('features must be 2D')
                    y = int(item.get('label', 0))
                    arr = self._fix_shape(arr)
                    self.data.append((arr, y))
                except Exception as e:
                    if strict:
                        raise ValueError(f'Bad sample at {path}:{line_no}: {e}') from e
                    continue
        if not self.data:
            raise ValueError(f'No valid samples found in {path}')

    def _fix_shape(self, arr):
        if arr.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - arr.shape[0], arr.shape[1]), dtype=np.float32)
            arr = np.vstack([arr, pad])
        elif arr.shape[0] > self.seq_len:
            arr = arr[:self.seq_len]
        if arr.shape[1] < self.input_dim:
            pad = np.zeros((arr.shape[0], self.input_dim - arr.shape[1]), dtype=np.float32)
            arr = np.hstack([arr, pad])
        elif arr.shape[1] > self.input_dim:
            arr = arr[:, :self.input_dim]
        return arr.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def compute_class_weights(dataset: AuditDataset, num_classes: int):
    counts = np.zeros(num_classes, dtype=np.float32)
    for _, y in dataset.data:
        if 0 <= y < num_classes:
            counts[y] += 1
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)
