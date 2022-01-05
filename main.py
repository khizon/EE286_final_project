import pandas as pd
import numpy as np
import torchaudio

from datasets import load_dataset, load_metric
from transformers import AutoConfig, Wav2Vec2Processor

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.file_utils import ModelOutput

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def label_to_id(label, label_list):

    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label

def preprocess_function(examples):
    speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
    target_list = [label_to_id(label, label_list) for label in examples[output_column]]

    result = processor(speech_list, sampling_rate=target_sampling_rate)
    result["labels"] = list(target_list)

    return result

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


if __name__ == '__main__':
    data_files = {
        "train": "data/train.csv", 
        "validation": "data/test.csv",
    }

    dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    print(train_dataset)
    print(eval_dataset)
    
    # We need to specify the input and output column
    input_column = "path"
    output_column = "emotion"
    # we need to distinguish the unique labels in our SER dataset
    label_list = train_dataset.unique(output_column)
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)
    print(f"A classification problem with {num_labels} classes: {label_list}")

    model_name_or_path = "lighteternal/wav2vec2-large-xlsr-53-greek"
    pooling_mode = "mean"

    # config
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf",
    )
    setattr(config, 'pooling_mode', pooling_mode)
    
    processor = Wav2Vec2Processor.from_pretrained(model_name_or_path,)
    target_sampling_rate = processor.feature_extractor.sampling_rate
    print(f"The target sampling rate: {target_sampling_rate}")
    
    train_dataset = train_dataset.map(
        preprocess_function,
        batch_size=100,
        batched=True,
        num_proc=4
    )
    
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batch_size=100,
        batched=True,
        num_proc=4
    )