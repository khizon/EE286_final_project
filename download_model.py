import wandb
from main import *

def cache_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generic_greek_model = 'lighteternal/wav2vec2-large-xlsr-53-greek'
    local_model = 'artifacts/aesdd_classifier-v0'
    config = AutoConfig.from_pretrained(local_model)
    processor = Wav2Vec2Processor.from_pretrained(generic_greek_model)
    model = Wav2Vec2ForSpeechClassification.from_pretrained(local_model).to(device)
    return config, processor, model, device

if __name__ == '__main__':
    with wandb.init() as run:
        artifact = run.use_artifact('khizon/EE286_final_project/aesdd_classifier:v0', type='model')
        artifact_dir = artifact.download()
    config, processor, model, device = cache_model()

    model.push_to_hub("greek-emotion-classifier-demo")