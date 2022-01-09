import numpy as np
import pandas as pd
from main import SpeechClassifierOutput, Wav2Vec2ForSpeechClassification
from datasets import load_dataset
from transformers import AutoConfig, Wav2Vec2Processor
import torchaudio
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

sns.set_theme(style="darkgrid", palette="pastel")

def demo_speech_file_to_array_fn(path):
    speech_array, _sampling_rate = torchaudio.load(path, normalize=True)
    resampler = torchaudio.transforms.Resample(_sampling_rate, 16_000)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def demo_predict(df_row):
    path, emotion = df_row["path"], df_row["emotion"]
    speech = demo_speech_file_to_array_fn(path)
    features = processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Emotion": config.id2label[i], "Score": round(score * 100, 3)} for i, score in enumerate(scores)]
    return outputs

def cache_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name_or_path = 'm3hrdadfi/wav2vec2-xlsr-greek-speech-emotion-recognition'
    config = AutoConfig.from_pretrained(model_name_or_path)
    processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
    model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)
    return config, processor, model, device

@st.cache
def load_data():
    return pd.read_csv('data/test.csv', delimiter = '\t')

def bar_plot(df):
    fig = plt.figure(figsize=(8, 6))
    plt.title("Prediction Scores")
    plt.xticks(fontsize=12)
    sns.barplot(x="Score", y="Emotion", data=df)
    st.pyplot(fig)

if __name__ == '__main__':
    test = load_data()

    config, processor, model, device = cache_model()
    print('Model loaded')

    st.title("Emotion Classifier for Greek Speech Audio Demo")
    if st.button("Classify Random Audio"):
        # Load demo file
        idx = np.random.randint(0, len(test))
        sample = test.iloc[idx]

        audio_file = open(sample['path'], 'rb')
        audio_bytes = audio_file.read()

        st.success(f'Label: {sample["emotion"]}')
        st.audio(audio_bytes, format='audio/ogg')

        outputs = demo_predict(sample)
        r = pd.DataFrame(outputs)
        # st.dataframe(r)
        bar_plot(r)