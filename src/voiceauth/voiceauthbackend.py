import platform
import subprocess
from multiprocessing import freeze_support
import librosa.display
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import librosa
import threading
import os
from voiceauthCore.core import predict_vggish

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

freeze_support()
matplotlib.use("Agg")

def typewriter_effect(text_widget, text, typing_speed=0.009):
    if hasattr(text_widget, "delete") and hasattr(text_widget, "insert"):

        for i in range(len(text) + 1):
            text_widget.delete("1.0", "end")

            text_widget.insert("end", text[:i])
            text_widget.yview("end")
            text_widget.update()
            threading.Event().wait(
                typing_speed
            )
    else:
        pass


def get_score_label(confidence):
    if confidence is None or not isinstance(confidence, (int, float)):
        return "Invalid confidence value"

    if confidence > 0.90:
        return "Almost certainly real"
    elif confidence > 0.80:
        return "Probably real but with slight doubt"
    elif confidence > 0.65:
        return "High likelihood of being fake, use caution"
    else:
        return "Considered fake: quality of audio does matter, do check for false positive just in case.."


def visualize_mfcc(temp_file_path):
    """Function to visualize MFCC features."""

    audio_data, sr = librosa.load(temp_file_path, sr=None)

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)

    plt.figure(figsize=(10, 4))
    plt.imshow(mfccs, aspect="auto", origin="lower", cmap="coolwarm")
    plt.title("MFCC Features")
    plt.ylabel("MFCC Coefficients")
    plt.xlabel("Time Frames")
    plt.colorbar(format="%+2.0f dB")

    plt.tight_layout()
    plt_file_path = os.path.join(
        os.path.dirname(temp_file_path),
        "mfccfeatures.png")
    plt.savefig(plt_file_path)

    if platform.system() == "Windows":
        os.startfile(plt_file_path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", plt_file_path], check=True)
    else:  # Linux/Unix
        subprocess.run(["xdg-open", plt_file_path], check=True)


def create_mel_spectrogram(temp_file_path):
    audio_file = os.path.join(temp_file_path)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    y, sr = librosa.load(audio_file)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    librosa.display.specshow(
        log_mel_spectrogram, sr=sr, x_axis="time", y_axis="mel", cmap="inferno"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.savefig("melspectrogram.png")
    mel_file_path = os.path.join(
        os.path.dirname(temp_file_path),
        "melspectrogram.png")
    plt.savefig(mel_file_path)
    if platform.system() == "Windows":
        os.startfile(mel_file_path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", mel_file_path], check=True)
    else:  # Linux/Unix
        subprocess.run(["xdg-open", mel_file_path], check=True)


def visualize_embeddings_tsne(file_path, output_path="tsne_visualization.png"):
    embeddings = predict_vggish(file_path)

    n_samples = embeddings.shape[0]

    if n_samples <= 1:
        print(
            f"t-SNE cannot be performed with only {n_samples} sample(s). Skipping visualization."
        )

        plt.figure(figsize=(10, 6))
        plt.text(
            0.5,
            0.5,
            "Not enough samples for t-SNE",
            fontsize=12,
            ha="center")
        plt.title("t-SNE Visualization of Audio Embeddings")
        plt.savefig(output_path)
        plt.close()
        os.startfile(output_path)
        return

    perplexity = min(30, n_samples - 1)

    perplexity = max(5.0, perplexity)

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c="blue",
        alpha=0.7,
        edgecolors="k",
    )
    plt.title("t-SNE Visualization of Audio Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()

    if platform.system() == "Windows":
        os.startfile(output_path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", output_path], check=True)
    else:  # Linux/Unix
        subprocess.run(["xdg-open", output_path], check=True)
