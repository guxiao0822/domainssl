import os
import fnmatch
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal
import neurokit2 as nk
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import pickle
from tqdm import tqdm

from ECG_feature import extract_feature

# -------------------------------
# Configuration
# -------------------------------
DATA_ROOT = "/mnt/Data/engs2588/data/PhysioNet2017/" # replace to your own data root where PhysioNet2017 is downaloded
REFERENCE_CSV = os.path.join(DATA_ROOT, "REFERENCE.csv")
ECG_OUT_PATH = os.path.join('../data', "ecg.pkl")
LABEL_OUT_PATH = os.path.join('../data', "label.pkl")
GROUP_OUT_PATH = os.path.join('../data', "fold.pkl")
FEAT_OUT_PATH = os.path.join('../data', "feat.pkl")

ORIGINAL_SR = 300  # original sampling rate
TARGET_SR = 125    # target sampling rate
DURATION_SEC = 10  # duration to clip
N_SPLITS = 5       # for StratifiedKFold

# -------------------------------
# 1. Collect .hea file paths and derive .mat filenames
# -------------------------------
hea_paths = []
mat_keys = []

for root, _, files in os.walk(DATA_ROOT):
    for file in files:
        if fnmatch.fnmatch(file, '*.hea'):
            hea_paths.append(os.path.join(root, file))
            # Extract relative file name without extension
            rel_path = os.path.relpath(os.path.join(root, file), DATA_ROOT)
            mat_keys.append(rel_path[:-4])  # remove ".hea"

hea_paths.sort()
mat_keys.sort()

# -------------------------------
# 2. Load and preprocess ECG signals
# -------------------------------
ecg_all = []
ecg_feature_all = []

for i, file_key in tqdm(enumerate(mat_keys), total=len(mat_keys)):

    # if i != 1254:
    #     continue

    mat_path = os.path.join(DATA_ROOT, file_key + '.mat')
    ecg = sio.loadmat(mat_path)['val'][0]  # shape (length,)

    # Clip to 10 seconds
    ecg = ecg[:ORIGINAL_SR * DURATION_SEC]
    ecg = nk.ecg_clean(ecg, sampling_rate=ORIGINAL_SR)

    # Pad if too short
    if len(ecg) < ORIGINAL_SR * DURATION_SEC:
        ecg = np.pad(ecg, (0, ORIGINAL_SR * DURATION_SEC - len(ecg)), mode='constant')

    # Resample to target sampling rate
    resampled_ecg = scipy.signal.resample(ecg, num=int(len(ecg) * TARGET_SR / ORIGINAL_SR))
    ecg_all.append(resampled_ecg)

    # calculate the features per samples
    # (please be noted that this is different from the features reported from the main paper. It was originally based on matlab, we found it might be easier to wrap everything in python standalone)
    ecg_feature = extract_feature(resampled_ecg, fs=TARGET_SR)
    ecg_feature_all.append(ecg_feature)

ecg_all = np.stack(ecg_all)  # shape (N, L)
ecg_feature_all = np.stack(ecg_feature_all)

# -------------------------------
# 3. Load labels and map to integers
# -------------------------------
df = pd.read_csv(REFERENCE_CSV, header=None)
label_names = df[1].values

class_mapping = {'N': 0, 'A': 1, 'O': 2, '~': 3}
labels = np.array([class_mapping[label] for label in label_names], dtype=int)

# -------------------------------
# 4. Stratified K-Fold Split
# -------------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
fold_indices = np.empty(len(labels), dtype=int)

for fold_id, (_, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    fold_indices[test_idx] = fold_id

# -------------------------------
# 5. Show label distribution
# -------------------------------
print("Overall class distribution:")
for k, v in sorted(Counter(labels).items()):
    label_name = list(class_mapping.keys())[list(class_mapping.values()).index(k)]
    print(f"  Class {k} ({label_name}): {v}")

print("\n Per-fold class distribution:")
for fold_id in range(N_SPLITS):
    fold_labels = labels[fold_indices == fold_id]
    dist = Counter(fold_labels)
    print(f"  Fold {fold_id}: ", end="")
    for k in sorted(class_mapping.values()):
        count = dist.get(k, 0)
        label_name = list(class_mapping.keys())[list(class_mapping.values()).index(k)]
        print(f"{label_name}: {count} ", end="")
    print()

# -------------------------------
# 6. Save processed data
# -------------------------------
with open(ECG_OUT_PATH, 'wb') as f:
    pickle.dump(ecg_all, f)

with open(LABEL_OUT_PATH, 'wb') as f:
    pickle.dump(labels, f)

with open(GROUP_OUT_PATH, 'wb') as f:
    pickle.dump(fold_indices, f)

with open(FEAT_OUT_PATH, 'wb') as f:
    pickle.dump(ecg_feature_all, f)

print("\n ECG and label data saved successfully.")