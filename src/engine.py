import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_lr_finder import LRFinder

from torch.utils.data import DataLoader, TensorDataset

from src.data_utils import (
    df_to_tensors, split_tensors, get_kfold_splits,
    oversample_high_values, scale_input, log_output, signed_log1p, create_dataloaders
)
from src.visualization import plot_predicted_vs_observed
from src.testing import evaluate_regression, observed_predicted
from src.model import build_model

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_lr(
    model_builder,      
    train_loader,
    start_lr: float = 1e-4,
    end_lr: float = 1.0,
    num_iter: int = 200,
    diverge: float = 5.0,
    loss_func = None,
    seed: Optional[int] = None,
):
    if seed is not None:
        set_seed(seed)

    finder_model = model_builder().to('cpu')

    loss_func = loss_func or nn.SmoothL1Loss()
    optimizer = optim.Adam(finder_model.parameters(), lr=start_lr)

    lr_finder = LRFinder(finder_model, optimizer, loss_func, device='cpu')
    lr_finder.range_test(train_loader, end_lr=end_lr, num_iter=num_iter, diverge_th=diverge) # type: ignore

    history_len = len(lr_finder.history["loss"])
    print(f"iteration: {history_len}")
    print(f"Search range: {start_lr:.1e} -> {end_lr:.1e}")

    best_lr = None
    if history_len > 0:
        lrs = lr_finder.history["lr"]
        losses = lr_finder.history["loss"]
        default_skip_start = 10
        
        lrs_for_suggestion = lrs[default_skip_start:]
        losses_for_suggestion = losses[default_skip_start:]
        
        min_grad_idx = np.gradient(np.asarray(losses_for_suggestion)).argmin()
        best_lr = float(lrs_for_suggestion[min_grad_idx])
        print(f"Suggested LR (bằng tính toán): {best_lr:.2E}")

        plot_result = lr_finder.plot(suggest_lr=True)
        
        if isinstance(plot_result, tuple) and len(plot_result) == 2:
            _, best_lr_from_plot = plot_result
            best_lr = best_lr_from_plot 

    lr_finder.reset()
    
    return best_lr


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr: float = 1e-3,
    loss_func=None,
    patience=None,
    restore_best_weights: bool = True,
    verbose: bool = True
):
    model = model.to(device)

    loss_func = loss_func or nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    history = {
        "train_loss": [], "val_loss": [], "best_epoch": None,
        "best_val_loss": None, "stopped_epoch": None,
    }
    
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_func(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += loss_func(model(x), y).item() * x.size(0)

        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch + 1
            history["best_val_loss"] = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

        if patience is not None and patience > 0 and patience_counter >= patience:
            history["stopped_epoch"] = epoch + 1
            
            if verbose:
                print(
                    f"Early stopping at epoch {epoch + 1}/{epochs} | "
                    f"Best Epoch: {history['best_epoch']} | "
                    f"Best Val Loss: {best_val_loss:.4f}"
                )
            break

    if history["stopped_epoch"] is None:
        history["stopped_epoch"] = len(history["train_loss"])

    if restore_best_weights:
        model.load_state_dict(best_state)

    return model, history

def cross_validate_model(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    model_builder=build_model,
    n_splits: int = 5,
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 64,
    patience: int = 20,
    loss_func=None,
    # oversample_threshold: float = 100.0,
    # oversample_factor: int = 10,
    verbose: bool = True,
):
    splits = get_kfold_splits(x_train, y_train, n_splits=n_splits)
    fold_metrics = []

    for fold_idx, (x_fold_train, y_fold_train, x_fold_val, y_fold_val) in enumerate(splits):
        if verbose:
            print(f"Fold {fold_idx + 1}/{n_splits}")

        x_fold_train = signed_log1p(x_fold_train)
        x_fold_val   = signed_log1p(x_fold_val)

        # x_fold_train, y_fold_train = oversample_high_values(
        #     x_fold_train, y_fold_train,
        #     threshold=oversample_threshold,
        #     repeat_factor=oversample_factor
        # )

        x_fold_train_scaled, x_fold_val_scaled, _ = scale_input(x_fold_train, x_fold_val)

        y_fold_train_log = log_output(y_fold_train)
        y_fold_val_log   = log_output(y_fold_val)

        train_loader = create_dataloaders(x_fold_train_scaled, y_fold_train_log, batch_size=batch_size, shuffle=False, drop_last=True)
        val_loader   = create_dataloaders(x_fold_val_scaled,   y_fold_val_log,   batch_size=batch_size)
        
        model = build_model()

        model, _ = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device="mps",
            loss_func=loss_func,
            patience=patience,
            verbose=False, 
            restore_best_weights=True, 
        )

        observed, predicted = observed_predicted(model, val_loader)
        
        metrics = evaluate_regression(observed, predicted, verbose=verbose) 
        fold_metrics.append(metrics)

    avg_metrics = np.nanmean(np.array(fold_metrics), axis=0)
    
    if verbose:\
        print(f"%RMSE trung bình: {avg_metrics[2]:.2f}%")

    return fold_metrics