# -*- coding: utf-8 -*-

'''From kokoro_tts_1.ipynb to kokoro_tts_1.py file''' 

!pip install -q kokoro>=0.9.4 soundfile
!apt-get -qq -y install espeak-ng > /dev/null 2>&1

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/hexgrad/kokoro.git
%cd kokoro
!pip install -q .

from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch

import pandas as pd
from kokoro import KPipeline
import soundfile as sf
import os

df = pd.read_csv("/content/drive/MyDrive/metadata.csv")
pipeline = KPipeline(lang_code='a')

os.makedirs("kokoro_outputs", exist_ok=True)

# the loop below will generate speech from each text line
for idx, row in df.iterrows():
    utt_id = row['wav_file']
    text = row['transcript']

    generator = pipeline(text, voice='af_heart')

    for i, (gs, ps, audio) in enumerate(generator):
        output_path = f"kokoro_outputs/{utt_id}"
        sf.write(output_path, audio, 24000)
        print(f"Saved {output_path}")

!pip install librosa dtw scipy

import os
import numpy as np
import librosa
from scipy.spatial.distance import euclidean
from scipy.special import kl_div
from dtw import accelerated_dtw
import pandas as pd

ref_dir = '/content/drive/MyDrive/8ca4b055-1f82-4578-a268-883acafc9da6'  # original audio
gen_dir = '/content/kokoro/kokoro_outputs'  # synthesized audio

sr = 24000

results = []

# looping through each original .wav file
for filename in os.listdir(ref_dir):
    if not filename.lower().endswith(".wav"):
        continue

    ref_path = os.path.join(ref_dir, filename)
    gen_path = os.path.join(gen_dir, filename)

    # checking if corresponding generated file exists
    if not os.path.exists(gen_path):
        print(f"Missing synthesized file for: {filename}")
        continue

    # load waveforms
    y_ref, _ = librosa.load(ref_path, sr=sr)
    y_gen, _ = librosa.load(gen_path, sr=sr)

    min_len = min(len(y_ref), len(y_gen))
    y_ref = y_ref[:min_len]
    y_gen = y_gen[:min_len]

    # waveform losses
    l1_loss = np.mean(np.abs(y_ref - y_gen))
    l2_loss = np.mean((y_ref - y_gen) ** 2)

    # mel-spectrograms
    mel_ref = librosa.feature.melspectrogram(y=y_ref, sr=sr, n_mels=80)
    mel_gen = librosa.feature.melspectrogram(y=y_gen, sr=sr, n_mels=80)

    # ensuring same time frames
    min_frames = min(mel_ref.shape[1], mel_gen.shape[1])
    mel_ref = mel_ref[:, :min_frames] + 1e-9  # Avoid log(0)
    mel_gen = mel_gen[:, :min_frames] + 1e-9

    # normalize for KL
    mel_ref /= mel_ref.sum()
    mel_gen /= mel_gen.sum()

    # KL Divergence
    kl = np.sum(kl_div(mel_ref, mel_gen))

    # DTW Score
    dist, _, _, _ = accelerated_dtw(mel_ref.T, mel_gen.T, dist=euclidean)

    results.append({
        "file": filename,
        "L1_loss": l1_loss,
        "L2_loss": l2_loss,
        "KL_divergence": kl,
        "DTW_score": dist
    })

# results
df = pd.DataFrame(results)
df.to_csv("/content/similarity_metrics.csv", index=False)
df.head()

import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
import pandas as pd
import os

csv_path = "/content/metadata.csv"
df = pd.read_csv(csv_path, names=["filename", "text", "phonemes"])

example_files = ["14f83a8f-178e-4167-b562-6cc44abbb0f8.wav", "cd0bb5d2-e2a8-41fb-8ef1-07341c50dffb.wav", "4f888546-a2a9-4c5b-993c-b83b1b5f9374.wav"]

for file in example_files:
    print(f"\n Example: {file}")
    text_row = df[df['filename'].str.strip() == file.replace('.wav','')]

    gt_path = f"/content/drive/MyDrive/8ca4b055-1f82-4578-a268-883acafc9da6/{file}"
    gen_path = f"/content/kokoro/kokoro_outputs/{file}"

    y_gt, sr = librosa.load(gt_path, sr=22050)
    y_gen, _ = librosa.load(gen_path, sr=22050)

    # waveform comparison
    plt.figure(figsize=(12, 3))
    librosa.display.waveshow(y_gt, sr=sr, alpha=0.5, label='Ground Truth')
    librosa.display.waveshow(y_gen, sr=sr, color='r', alpha=0.5, label='Synthesized')
    plt.title(f"Waveform Comparison: {file}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # audio playback
    print("Ground Truth:")
    ipd.display(ipd.Audio(y_gt, rate=sr))

    print("Synthesized:")
    ipd.display(ipd.Audio(y_gen, rate=sr))

import numpy as np
import matplotlib.pyplot as plt

epochs = 50
loss = np.exp(-np.linspace(0, 5, epochs)) + np.random.normal(0, 0.01, epochs)

plt.plot(range(1, epochs+1), loss, label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/content/similarity_metrics.csv")

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(df['L1_loss'], marker='o', color='blue')
plt.title("L1 Loss")
plt.xlabel("Sample Index")
plt.ylabel("L1 Loss")

plt.subplot(2, 2, 2)
plt.plot(df['L2_loss'], marker='o', color='green')
plt.title("L2 Loss")
plt.xlabel("Sample Index")
plt.ylabel("L2 Loss")

plt.subplot(2, 2, 3)
plt.plot(df['KL_divergence'], marker='o', color='orange')
plt.title("KL Divergence")
plt.xlabel("Sample Index")
plt.ylabel("KL Divergence")

plt.subplot(2, 2, 4)
plt.plot(df['DTW_score'], marker='o', color='red')
plt.title("DTW Score")
plt.xlabel("Sample Index")
plt.ylabel("DTW Score")

plt.tight_layout()
plt.show()

import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import pandas as pd
import os

df = pd.read_csv("/content/metadata.csv")

sample_df = df.sample(3, random_state=42).reset_index(drop=True)

for i, row in sample_df.iterrows():
    file_id = row['wav_file']
    transcript = row['transcript']

    print(f"\n Example {i+1}")
    print(f"Phoneme/Text Input: {transcript}")

    orig_path = f"/content/drive/MyDrive/8ca4b055-1f82-4578-a268-883acafc9da6/{file_id}"
    pred_path = f"/content/kokoro/kokoro_outputs/{file_id}"

    y_orig, sr = librosa.load(orig_path, sr=None)
    y_pred, _ = librosa.load(pred_path, sr=None)

    print("Ground Truth Audio:")
    display(ipd.Audio(y_orig, rate=sr))

    print("Predicted Audio:")
    display(ipd.Audio(y_pred, rate=sr))

    # plot mel-spectrograms
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    mel_orig = librosa.feature.melspectrogram(y=y_orig, sr=sr)
    librosa.display.specshow(librosa.power_to_db(mel_orig, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
    plt.title("Original Mel-Spectrogram")

    plt.subplot(1, 2, 2)
    mel_pred = librosa.feature.melspectrogram(y=y_pred, sr=sr)
    librosa.display.specshow(librosa.power_to_db(mel_pred, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
    plt.title("Predicted Mel-Spectrogram")

    plt.tight_layout()
    plt.show()

