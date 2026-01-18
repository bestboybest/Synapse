import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from collections import Counter

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

def rectify(x):
    #Take absolute value to recitfy the signal
    x = np.array(x)
    rectified = np.abs(x)
    return rectified

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

def featureEngineering(windows):
    Totalfeatures = []
    for window in windows:
        features = []
        for i in range(window.shape[1]):
            #Calculating RMS of window (root mean square) (represents signal energy) 
            rms = np.sqrt(np.mean((window[:, i]) ** 2))
            features.append(rms)

            #Calculating MAV of window (mean absolute value) (represents average activation)
            mav = np.mean(window[:, i])
            features.append(mav)

            #Calculating WL of window (window length) (represents signal complexity)
            wl = np.sum(np.abs(np.diff(window[:, i])))
            features.append(wl)

        Totalfeatures.append(features)
    return np.array(Totalfeatures, dtype=np.float32)

def zc(windows, alpha):
    Totalfeatures = []
    for window in windows:
        features = []
        for i in range(window.shape[1]):
            #Calculating RMS of window (root mean square) (represents signal energy) (RMS same regardless of rectification)
            rms = np.sqrt(np.mean((window[:, i]) ** 2))

            #Need a threshold to get rid of noisy false positive zero crossings
            threshold = alpha * rms
            #Let's calculate zero crossings now
            zcCount = 0
            x = window[:, i]
            for j in range(len(x) - 1):
                if ((x[j] > 0 and x[j+1] < 0) or (x[j] < 0 and x[j+1] > 0)):
                    if abs(x[j] - x[j+1]) >= threshold:
                        zcCount += 1
            features.append(zcCount)
        Totalfeatures.append(features)
    return np.array(Totalfeatures, dtype = np.float32)

def merge(features1, features2):
    merged = []

    for i in range(features1.shape[0]):
        row = []
        for ch in range(8):
            row.append(features1[i, 3*ch])     # RMS
            row.append(features1[i, 3*ch + 1]) # MAV
            row.append(features1[i, 3*ch+2])   # WL
            row.append(features2[i, ch])       # ZC
        merged.append(row)
    return np.array(merged, dtype=np.float32)

def features(fs, raw, windowSize, stride, alpha):
    filtered = filtering(fs, raw)

    zcCopy = filtered.copy()
    zcWindows = windowing(zcCopy, windowSize, stride)
    features1 = zc(zcWindows, alpha)

    rectified = rectify(filtered)
    normalized = normalize(rectified)
    windows = windowing(normalized, windowSize, stride)
    features2 = featureEngineering(windows)

    features = merge(features2, features1)
    return features

def createData(windowSize, stride, alpha):
    xAll = []
    yAll = []
    metaAll = []
    fs = 512 # Already given

    datasetDir = Path("./Synapse_Dataset")

    for sessionDir in datasetDir.iterdir():
        if not sessionDir.is_dir():
            continue

        for subjectDir in sessionDir.iterdir():
            for csvFile in subjectDir.glob("*.csv"):

                raw = pd.read_csv(csvFile).values
                xCsv = features(fs, raw, windowSize, stride, alpha)

                gesture_id = int(csvFile.stem.split("_")[0].replace("gesture", ""))
                trial_no = int(csvFile.stem.split("_")[1].replace("trial", ""))
                session_no = int(subjectDir.name.split("_")[0].replace("session", ""))
                subject_no = int(subjectDir.name.split("_")[2])

                yCsv = np.full(xCsv.shape[0], gesture_id)

                metaCsv = [{"subject": subject_no, "session": session_no, "trial": trial_no} for _ in range(xCsv.shape[0])]

                xAll.append(xCsv)
                yAll.append(yCsv)
                metaAll.extend(metaCsv)
    
    x = np.vstack(xAll)
    y = np.concatenate(yAll)
    meta = metaAll

    return x, y, meta

def majorityVote(labels):
    return Counter(labels).most_common(1)[0][0]

