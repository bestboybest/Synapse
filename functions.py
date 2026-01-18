import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import welch
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

titleFont = {'weight':'bold', 'color':'orangered', 'size':20, 'name':'Comic Sans MS'}
normalFont = {'color':'maroon', 'size':16}

#for plotting signals
def plot(time, x):
    plt.figure(figsize=(12, 4))
    plt.plot(time, x)
    plt.xlabel("Time (s)", fontdict = normalFont)
    plt.ylabel("Amplitude", fontdict = normalFont)
    plt.show()  

def visualizeFilter(time, raw, filtered):
    diff = raw - filtered

    plt.figure(figsize=(12, 3))
    plt.plot(time, diff)
    plt.title("Removed Component (Raw - Filtered)", fontdict = titleFont)
    plt.xlabel("Time (s)", fontdict = normalFont)
    plt.ylabel("Amplitude", fontdict = normalFont)
    plt.show()

#For bandpass filtering
def filtering(fs, x):
    #Defining nyquist frequency
    nyq = fs/2
    #We will take our range as 20-250 hz (standard for EMG preprocessing)
    low = 20/nyq
    high = 250/nyq

    #We apply butterworth bandpass filter and get filtering coefficients (we use order = 4 as is standard)
    b, a = signal.butter(N = 4, Wn = [low, high], btype = "bandpass")

    #We use filtfilt() so theres no time shift in data, for each channel independently
    filtered = np.zeros_like(x)
    for i in range(x.shape[1]):
        filtered[:, i] = signal.filtfilt(b, a, x[:, i])

    return filtered

def normalize(x):
    #We will do RMS normalization
    #RMS represents the muscle activation strength, which will differ from subject to subject
    #So we do normalization with respect to this RMS strength, to remove subject bias
    #We will perform this for each trial individually only (there is no data leakage)

    normalized = np.zeros_like(x)
    for i in range(x.shape[1]):
        rms = np.sqrt(np.mean((x[:, i])**2))
        
        if rms < 1e-6:
            #To guard against dead channels
            normalized[:, i] = 0.0
        else:
            normalized[:, i] = x[:, i] / rms

    return normalized

def windowing(x, windowSize, stride):
    windows = []
    start = 0
    #Drops trailing 20ms (if we use windowSize = 200ms and overlap = 50%) but thats okay
    while start + windowSize <= x.shape[0]:
        windows.append(x[start : start + windowSize, :])
        start += stride
    return np.array(windows)

def timeFeatures(windows_norm):
    X = []
    for w in windows_norm:
        row = []
        for ch in range(w.shape[1]):
            sig = w[:, ch]
            rms = np.sqrt(np.mean(sig**2))
            mav = np.mean(np.abs(sig))
            wl  = np.sum(np.abs(np.diff(sig)))
            ssc = np.sum(
                (sig[1:-1] - sig[:-2]) *
                (sig[1:-1] - sig[2:]) > 0
            )
            row.extend([rms, mav, wl, ssc])
        X.append(row)
    return np.array(X, dtype=np.float32)

def freqFeatures(windows_raw, fs=512):
    X = []
    for w in windows_raw:
        row = []
        for ch in range(w.shape[1]):
            sig = w[:, ch]
            rms = np.sqrt(np.mean(sig**2))
            zc = np.sum(np.diff(np.sign(sig)) != 0)

            f, pxx = welch(sig, fs=fs, nperseg=len(sig))
            mnf = np.sum(f * pxx) / np.sum(pxx)
            cdf = np.cumsum(pxx)
            mdf = f[np.where(cdf >= np.sum(pxx)/2)[0][0]]

            row.extend([rms, zc, mnf, mdf])
        X.append(row)
    return np.array(X, dtype=np.float32)

def interChannelFeatures(windows_norm):
    X = []
    for w in windows_norm:
        rmsVals = np.sqrt(np.mean(w**2, axis=0))
        rmsVals = np.array(rmsVals) + 1e-8

        q75, q25 = np.percentile(rmsVals, [75 ,25])
        iqr_spread = (q75 - q25) / np.median(rmsVals)

        dominance = np.percentile(rmsVals, 90) / np.median(rmsVals)

        p = rmsVals / (np.sum(rmsVals) + 1e-8)
        entropy = -np.sum(p * np.log(p + 1e-8))

        corr_vals = []
        for i in range(w.shape[1]):
            for j in range(i+1, w.shape[1]):
                c = np.corrcoef(w[:, i], w[:, j])[0, 1]
                if not np.isnan(c):
                    corr_vals.append(c)

        corr_mean = np.mean(corr_vals)
        corr_std  = np.std(corr_vals)
        corr_max  = np.max(np.abs(corr_vals))

        X.append([iqr_spread, dominance, entropy, corr_mean, corr_std, corr_max])
    return np.array(X, dtype=np.float32)

def hjorthFeatures(windows_norm):
    #Calculating hjorth mobility and complexity
    X = []

    for w in windows_norm:
        row = []
        for ch in range(w.shape[1]):
            sig = w[:, ch]

            #First and second derivatives
            d1 = np.diff(sig)
            d2 = np.diff(d1)

            var_sig = np.var(sig)
            var_d1  = np.var(d1)
            var_d2  = np.var(d2)

            if var_sig < 1e-8 or var_d1 < 1e-8:
                mobility = 0.0
                complexity = 0.0
            else:
                mobility = np.sqrt(var_d1 / var_sig)
                complexity = np.sqrt(var_d2 / var_d1) / mobility

            row.extend([mobility, complexity])

        X.append(row)

    return np.array(X, dtype=np.float32)

def rms_sequence(windows_norm):
    return np.sqrt(np.mean(windows_norm**2, axis=(1, 2)))

def temporalFeatures(windows_norm):
    rms_seq = rms_sequence(windows_norm)

    t = np.arange(len(rms_seq))
    slope = np.polyfit(t, rms_seq, 1)[0]

    volatility = np.std(rms_seq) / (np.mean(rms_seq) + 1e-8)
    delta = rms_seq[-1] - rms_seq[0]

    return np.array([slope, volatility, delta], dtype=np.float32)

def createData(window_size, stride, fs=512, out_path="./data/window_cache.npz"):
    windows_raw_all = []
    windows_norm_all = []
    y_all = []
    meta_all = []
    groups_all = []

    dataset_dir = Path("./Synapse_Dataset")

    for session_dir in dataset_dir.iterdir():
        if not session_dir.is_dir():
            continue

        for subject_dir in session_dir.iterdir():
            subject_id = int(subject_dir.name.split("_")[2])

            for csv_file in subject_dir.glob("*.csv"):
                raw = pd.read_csv(csv_file).values

                filtered = filtering(fs, raw)

                win_raw = windowing(filtered, window_size, stride)

                #Rectified + normalized
                win_norm = np.array([normalize(np.abs(w)) for w in win_raw])

                gesture = int(csv_file.stem.split("_")[0].replace("gesture", ""))
                trial = int(csv_file.stem.split("_")[1].replace("trial", ""))

                n = win_raw.shape[0]

                windows_raw_all.append(win_raw)
                windows_norm_all.append(win_norm)

                y_all.extend([gesture]*n)
                groups_all.extend([subject_id]*n)

                meta_all.extend([
                    {
                        "subject": subject_id,
                        "session": int(session_dir.name.replace("Session", "")),
                        "trial": trial
                    }
                ] * n)

    np.savez(out_path, windows_raw=np.vstack(windows_raw_all), windows_norm=np.vstack(windows_norm_all), y=np.array(y_all), meta=np.array(meta_all, dtype=object), groups=np.array(groups_all))

#For soft voting
def soft_vote(x):
    return x.value_counts().idxmax()


