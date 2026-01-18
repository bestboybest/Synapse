# sEMG Gesture Recognition – Synapse PS

This repository implements an end-to-end pipeline for hand gesture recognition using surface EMG (sEMG) signals, developed for the Synapse Problem Statement.

The system performs window-level classification followed by trial-level aggregation to output a single gesture prediction per trial.

---

## Gestures

The model predicts the following 5 gestures:

Label | Gesture
----- | -------
0 | Open Hand
1 | Closed Hand
2 | Lateral Pinch
3 | Signalling Sign
4 | Rock Sign

---

## Repository Structure

Only code and trained artifacts are included.  
The dataset is NOT committed due to size.

.
├── infer.py                 
├── functions.py             
├── histgb_model.joblib      
├── scaler.joblib            
├── config.json            
├── README.md                
├── model.ipynb             
└── data/                    
    ├── xtime.npy
    ├── xfreq.npy
    ├── xinter.npy
    ├── xhjorth.npy
    ├── xtemp.npy
    └── window_cache.npz

## Dataset Assumptions

- The testing dataset has the **same folder structure and file naming convention** as the training dataset.
- Each CSV file corresponds to **one trial**.
- Each CSV contains multi-channel raw sEMG samples.

Example (dataset not included in repo):

Synapse_Dataset/
└── Session1/
    └── session1_subject_1/
        ├── gesture00_trial01.csv
        ├── gesture00_trial02.csv
        └── ...

---

## Method Summary

1. Signal preprocessing  
   - Bandpass filtering  
   - Rectification  
   - RMS-based normalization  

2. Windowing  
   - Fixed-length overlapping windows  

3. Feature extraction (per window)  
   - Time-domain features  
   - Frequency-domain features  
   - Hjorth parameters  
   - Inter-channel correlation features  

4. Temporal (trial-level) features  
   - RMS slope  
   - RMS volatility  
   - RMS first–last difference  

5. Classification  
   - HistGradientBoostingClassifier  
   - Window-level predictions  

6. Trial-level aggregation  
   - Majority voting over window predictions  
   - Produces one final label per trial  

---

## Evaluation

- Subject-wise GroupKFold cross-validation
- Metric: Trial-level Macro F1 score
- Performance consistently observed in the range **0.71–0.72 Macro F1**
- Subject-wise splitting prevents data leakage and reflects real-world generalization

---

## How to Run Inference

### Requirements

Python 3.9+

Install dependencies:
pip install numpy pandas scikit-learn scipy joblib

---

### Single Trial Inference

Run inference on one CSV file:

python infer.py --input path/to/gestureXX_trialYY.csv

Example:
python infer.py --input Synapse_Dataset/Session1/session1_subject_1/gesture00_trial01.csv

Output:
Predicted Gesture: Open Hand

---

### Batch Inference (Folder)

Run inference on all trials in a folder:

python infer.py --input Synapse_Dataset/Session1/session1_subject_1/

Output:
gesture00_trial01.csv -> Open Hand  
gesture00_trial02.csv -> Closed Hand  
gesture00_trial03.csv -> Open Hand  

The script automatically processes all CSV files in the folder.

---

## Inference Pipeline

Raw CSV  
→ Filtering  
→ Windowing  
→ Feature Extraction  
→ Scaling  
→ Window-level Prediction  
→ Trial-level Majority Vote  
→ Final Gesture  

The inference pipeline mirrors the training pipeline to ensure consistency.

---

## Notes

- Temporal smoothing and ensemble methods were evaluated but excluded, as they did not improve subject-wise performance.
- Explicit feature pruning was avoided to preserve cross-subject generalization.
- The focus was on robustness, interpretability, and correct evaluation.

---

## Submission Contents

- Trained model and scaler
- End-to-end inference script
- Feature extraction and preprocessing code
- Clear instructions for running inference

The dataset is intentionally excluded.

---

## Final Remarks

This submission provides a complete, reproducible, and interpretable sEMG gesture recognition pipeline suitable for evaluation on the provided Synapse test dataset.
