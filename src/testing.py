import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def _to_numpy_1d(values):
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()

    return np.ravel(values)

def observed_predicted(model, data_loader, device="mps"):
    model.eval()

    observed_list = []
    predicted_list = []

    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)

            output = model(data)

            label = torch.expm1(label)
            output = torch.expm1(output)

            observed_list.append(label.cpu().numpy().ravel())
            predicted_list.append(output.cpu().numpy().ravel())

    observed = np.concatenate(observed_list)
    predicted = np.concatenate(predicted_list)

    return observed, predicted


def evaluate_regression(y_true, y_pred, verbose: bool = True):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # MAE = abs(predicted - observed) / size
    mae = mean_absolute_error(y_true, y_pred)
    
    # RMSE = sqrt((observed - predicted)^2 / size)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # %RMSE = (RMSE / mean(observed)) * 100
    mean_true = np.mean(y_true)
    pct_rmse = (rmse / mean_true) * 100 if mean_true > 0 else np.nan

    # R^2 
    r2 = r2_score(y_true, y_pred)

    if verbose:
        print(f"RMSE    : {rmse:.4f}")
        print(f"MAE     : {mae:.4f}")
        print(f"%RMSE   : {pct_rmse:.2f}%")
        print(f"R^2     : {r2:.4f}")
        
    return rmse, mae, pct_rmse