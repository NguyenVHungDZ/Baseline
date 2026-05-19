import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple


def load_data(csv_path: str, drop_list: Optional[list] = None) -> pd.DataFrame:
    if drop_list is None:
        drop_list = ["id", "pop"]
    data = pd.read_csv(csv_path)
    return data.drop(columns=[c for c in drop_list if c in data.columns])

def df_to_tensors(df: pd.DataFrame, target_col: str) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(df.drop(columns=[target_col]).values, dtype=torch.float32)
    y = torch.tensor(df[target_col].values, dtype=torch.float32).view(-1, 1)
    return x, y


def signed_log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))

def log_output(y: torch.Tensor) -> torch.Tensor:
    return torch.log1p(y)

def scale_input(x_train: torch.Tensor, x_val: torch.Tensor):
    scaler = RobustScaler()

    x_train_scaled = torch.from_numpy(scaler.fit_transform(x_train.numpy())).float()
    x_val_scaled = torch.from_numpy(scaler.transform(x_val.numpy())).float()
    return x_train_scaled, x_val_scaled, scaler


def oversample_high_values(x: torch.Tensor, y: torch.Tensor, threshold: float, repeat_factor: int):
    mask = (y > threshold).squeeze()
    if mask.any() and repeat_factor > 1:
        x_high, y_high = x[mask], y[mask]
        num_repeats = repeat_factor - 1

        x_new = torch.cat([x, x_high.repeat(num_repeats, 1)], dim=0)
        y_new = torch.cat([y, y_high.repeat(num_repeats, 1)], dim=0)

        idx = torch.randperm(len(x_new))
        return x_new[idx], y_new[idx]
    return x, y


def split_tensors(x: torch.Tensor, y: torch.Tensor, test_size: float = 0.2, seed: int = 18):
    n = len(x)
    n_test = int(n * test_size)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]

def get_kfold_splits(x: torch.Tensor, y: torch.Tensor, n_splits: int = 5):
    kf = KFold(n_splits=n_splits)
    for ti, vi in kf.split(x):
        yield x[ti], y[ti], x[vi], y[vi]

def create_dataloaders(x: torch.Tensor, 
                       y: torch.Tensor, 
                       batch_size: int = 64, 
                       shuffle: bool = False,
                       drop_last: bool = False) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)