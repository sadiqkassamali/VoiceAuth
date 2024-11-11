import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import logging

# Set up logging
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
        logger.error(f"Error encountered while parsing file: {file_path}. Error: {e}")
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


# Load the data
files, labels = load_data(data_dir)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    files, labels, test_size=0.2, random_state=42
)

# Convert audio files to feature matrices
X_train = [extract_features(file) for file in X_train]
X_test = [extract_features(file) for file in X_test]

# Filter out None values
X_train = [x for x in X_train if x is not None]
X_test = [x for x in X_test if x is not None]

logger.info(
    f"Extracted features for {len(X_train)} training samples and {len(X_test)} testing samples."
)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
logger.info("Model training completed.")

# Evaluate the model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model
model_filename = "dataset/deepfakevoice.joblib"
joblib.dump(model, model_filename)
logger.info(f"Model saved as {model_filename}")
