import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from functions import (
    filtering, windowing, normalize,
    timeFeatures, freqFeatures,
    interChannelFeatures, hjorthFeatures,
    temporalFeatures
)

# ------------------
# Load config
# ------------------
with open("config.json") as f:
    cfg = json.load(f)

fs = cfg["fs"]
window_size = cfg["window_size"]
stride = cfg["stride"]
gesture_map = cfg["gesture_map"]

# ------------------
# Load model + scaler
# ------------------
model = joblib.load("histgb_model.joblib")
scaler = joblib.load("scaler.joblib")


def infer(csv_path):
    """
    Runs full inference on a single trial CSV file.
    Returns final gesture label (string).
    """

    # 1) Load raw EMG
    raw = pd.read_csv(csv_path).values

    # 2) Preprocessing
    filtered = filtering(fs, raw)
    windows_raw = windowing(filtered, window_size, stride)
    windows_norm = np.array([normalize(np.abs(w)) for w in windows_raw])

    n_windows = windows_norm.shape[0]

    # 3) Window-level features
    X_time = timeFeatures(windows_norm)
    X_freq = freqFeatures(windows_raw)
    X_inter = interChannelFeatures(windows_norm)
    X_hjorth = hjorthFeatures(windows_norm)

    # 4) Temporal (trial-level) features
    # computed ONCE, then broadcast
    x_temp = temporalFeatures(windows_norm)          # shape (3,)
    X_temp = np.tile(x_temp, (n_windows, 1))         # shape (n_windows, 3)

    # 5) Concatenate in SAME ORDER as training
    X = np.hstack([X_time, X_freq, X_inter, X_hjorth, X_temp])

    # 6) Scale
    X = scaler.transform(X)

    # 7) Window-level prediction
    y_pred = model.predict(X)

    # 8) Trial-level majority vote
    final_label = pd.Series(y_pred).value_counts().idxmax()
    final_gesture = gesture_map[str(final_label)]

    return final_gesture

def infer_path(path):
    path = Path(path)

    if path.is_file():
        pred = infer(path)
        print(f"{path.name} -> {pred}")

    elif path.is_dir():
        for csv in sorted(path.glob("*.csv")):
            pred = infer(csv)
            print(f"{csv.name} -> {pred}")

    else:
        raise ValueError("Invalid input path")

# Optional CLI usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gesture inference")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to CSV file OR folder containing CSVs"
    )
    args = parser.parse_args()

    infer_path(args.input)
