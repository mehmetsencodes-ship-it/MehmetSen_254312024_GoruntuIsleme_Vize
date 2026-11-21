import numpy as np


def count_metrics(pred, ref, times=None):
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    err = pred - ref
    out = {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "Bias": float(np.mean(err)),
        "ExactAcc": float(np.mean(err == 0)),
    }
    if times is not None:
        out["mean_time_s"] = float(np.mean(times))
    return out
