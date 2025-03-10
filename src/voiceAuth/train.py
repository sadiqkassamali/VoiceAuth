from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
import logging
import os

import librosa


def frozen_oo():
    """Check if code is frozen with optimization=2"""
    import sys
    if frozen_oo.__doc__ is None and hasattr(sys, "frozen"):
        from ctypes import c_int, pythonapi
        c_int.in_dll(pythonapi, "Py_OptimizeFlag").value = 2


frozen_oo()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

data_dir = "dataset"


def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type="kaiser_fast")
        mfccs = np.mean(
            librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0
        )
        return mfccs
    except Exception as e:
        logger.error(
            f"Error encountered while parsing file: {file_path}. Error: {e}")
        return None


def load_data(data_dir):
    fake_files = [
        os.path.join(data_dir, "fake", f)
        for f in os.listdir(os.path.join(data_dir, "fake"))
        if f.endswith(".wav")
    ]
    real_files = [
        os.path.join(data_dir, "real", f)
        for f in os.listdir(os.path.join(data_dir, "real"))
        if f.endswith(".wav")
    ]

    fake_labels = [0] * len(fake_files)
    real_labels = [1] * len(real_files)

    files = fake_files + real_files
    labels = fake_labels + real_labels

    logger.info(
        f"Loaded {len(fake_files)} fake files and {len(real_files)} real files."
    )
    return files, labels



files, labels = load_data(data_dir)


X_train, X_test, y_train, y_test = train_test_split(
    files, labels, test_size=0.2, random_state=42
)


X_train = [extract_features(file) for file in X_train]
X_test = [extract_features(file) for file in X_test]


X_train = [x for x in X_train if x is not None]
X_test = [x for x in X_test if x is not None]

logger.info(
    f"Extracted features for {len(X_train)} training samples and {len(X_test)} testing samples."
)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
logger.info("Model training completed.")


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")


model_filename = "dataset/deepfakevoice.joblib"
joblib.dump(model, model_filename)
logger.info(f"Model saved as {model_filename}")
