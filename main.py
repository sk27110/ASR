from src.dataset.librispeech_dataset import LibrispeechDataset

dataset = LibrispeechDataset("train-clean-100")

print(dataset.collate_fn())