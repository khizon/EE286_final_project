from main import *

from sklearn.metrics import classification_report

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array
    resampler = torchaudio.transforms.Resample(sampling_rate, 16_000)
    speech_array = resampler(speech_array).squeeze().numpy()

    batch["speech"] = speech_array
    return batch


def predict(batch):
    features = processor(batch["speech"], sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits 

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    batch["predicted"] = pred_ids
    return batch

if __name__ == '__main__':
    
    data_files = {
        "test" : 'data/test.csv'
    }
    test_dataset = load_dataset('csv', data_files = data_files, delimiter = "\t")["test"]
    print(test_dataset)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # model_name_or_path = "m3hrdadfi/wav2vec2-xlsr-greek-speech-emotion-recognition"
    model_name_or_path2 = "lighteternal/wav2vec2-large-xlsr-53-greek"
    # model_name_or_path = "data/wav2vec2-xlsr-greek-speech-emotion-recognition/checkpoint-180"
    model_name_or_path = 'artifacts/aesdd_classifier:v0'
    config = AutoConfig.from_pretrained(model_name_or_path)
    processor = Wav2Vec2Processor.from_pretrained(model_name_or_path2)
    model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)
    
    test_dataset = test_dataset.map(speech_file_to_array_fn)
    
    result = test_dataset.map(predict, batched=True, batch_size=8)
    
    label_names = [config.id2label[i] for i in range(config.num_labels)]
    
    print(f'Labels: {label_names}')
          
    y_true = [config.label2id[name] for name in result["emotion"]]
    y_pred = result["predicted"]

    print(y_true[:5])
    print(y_pred[:5])
          
    print(classification_report(y_true, y_pred, target_names=label_names))