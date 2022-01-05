import pandas as pd
import numpy as np
import random

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torchaudio
from torch.utils.data import Dataset, DataLoader

from transformers import Wav2Vec2Processor, Wav2Vec2Model

'''
RNG seed
'''

def seed_everything(seed=86):
    _ = torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

'''
Dataset Class
'''
class greekEmotionDataset(Dataset):
    def __init__(self, annotations, processor, label_list):
        self.annotations = annotations
        self.processor = processor
        self.label_list
        self.target_sampling_rate = self.processor.feature_extractor.target_sampling_rate

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        data_row = self.annotations.iloc[index]

        emotion = data_row['emotion']
        label = self.__label_to_id(emotion)
        signal, sr = torchaudio.load(data_row['path'], normalize=True)

        signal = self.__preprocess(signal, sr)

        # Run through the processor to tokenize
        inputs = self.processor(signal, sampling_rate = self.target_sampling_rate, return_tensors = 'pt')

        return dict(
            inputsinput_ids = inputs['input_ids'].flatten(),
            attention_mask = inputs['attention_mask'].flatten(),
            emotion = emotion,
            label = torch.tensor(label, dtype=torch.float32)
        )

    def __label_to_id(self, label):
        return self.abel_list.index(label) if label in self.label_list else -1

    def __preprocess(self, signal, sr):
        # Resample
        resampler = torchaudio.transforms.Resample(sr, self.target_sampling_rate)
        signal = resampler(signal).squeeze().numpy()

        return signal

'''
Create Data loader
'''
def create_aesdd_dataloader(file_path, processor, label_list, batch_size, shuffle = False, sample = None):

    df = pd.read_csv(file_path)

    if sample:
        df = df.sample(sample)

    ds = greekEmotionDataset(df, processor, label_list)
    return DataLoader(ds, batch_size = batch_size, shuffle = shuffle)

'''
Create Classifier Model
'''
class ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
class EmotionClassifier(nn.Module):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(config)
        self.classifier = ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor.freeze_feature_extractor()

    def merged_strategy(self, hidden_states):
        return torch.mean(hidden_states, dim=1)

    def forward(self, input_values, attention_mask = None, labels = None):
        outputs = self.wav2vec2(
            input_values,
            attention_mask = attention_mask
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            self.config.problem_type = "multi_label_classification"
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return logits, loss


