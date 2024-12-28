import platform
import subprocess

from transformers import pipeline



print("Loading Hugging Face model...")
pipe = pipeline("audio-classification",
                model="MelodyMachine/Deepfake-audio-detection-V2")
print("model-melody model loaded successfully.")
pipe.save_pretrained("/dataset/")


print("Loading Hugging Face model...")
pipe2 = pipeline("audio-classification",
                 model="HyperMoon/wav2vec2-base-960h-finetuned-deepfake")
print("960h model loaded successfully.")
pipe2.save_pretrained("/dataset/")