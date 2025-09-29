import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
            self,
            index: list[dict],
            text_encoder=None,
            target_sr: int = 16000,
            n_mels: int = 80,
            win_length: int = 400,
            hop_length: int = 160,
            n_fft: int = 512,
            f_min: float = 0.0,
            f_max: float = None
    ):

        self._index: list[dict] = index
        self.text_encoder = text_encoder
        self.target_sr = target_sr

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self._index)

    def __getitem__(self, index: int) -> dict:
        data_dict = self._index[index]

        audio_path = data_dict["path"]
        text = data_dict.get("text", "")

        audio = self.load_audio(audio_path)

        spectrogram = self.get_spectrogram(audio)


        text_encoded = self.encode_text(text)

        instance_data = {
            "audio": audio, 
            "spectrogram": spectrogram,  
            "text": text, 
            "text_encoded": text_encoded, 
            "audio_path": audio_path,
        }


        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def load_audio(self, path: str) -> torch.Tensor:
        audio_tensor, sr = torchaudio.load(path)

        audio_tensor = audio_tensor[0:1, :]

        if sr != self.target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, self.target_sr)

        return audio_tensor

    def get_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel_transform(audio)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        return mel_spec_db.squeeze(0)

    def encode_text(self, text: str) -> torch.Tensor | None:
        if self.text_encoder is None:
            return None
        if hasattr(self.text_encoder, "encode"):
            return torch.tensor(self.text_encoder.encode(text), dtype=torch.long)
        else:
            return torch.tensor(self.text_encoder(text), dtype=torch.long)

    def preprocess_data(self, instance: dict) -> dict:
        spec = instance.get("spectrogram")
        if spec is not None:
            spec = (spec - spec.mean()) / (spec.std() + 1e-9)
            instance["spectrogram"] = spec

        return instance

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        audio_lengths = [b["audio"].shape[-1] for b in batch]
        max_audio_len = max(audio_lengths)

        padded_audios = torch.zeros(len(batch), 1, max_audio_len)
        for i, b in enumerate(batch):
            padded_audios[i, 0, :audio_lengths[i]] = b["audio"]

        spec_lengths = [b["spectrogram"].shape[-1] for b in batch]
        max_spec_len = max(spec_lengths)
        n_mels = batch[0]["spectrogram"].shape[0]

        padded_specs = torch.zeros(len(batch), n_mels, max_spec_len)
        for i, b in enumerate(batch):
            padded_specs[i, :, :spec_lengths[i]] = b["spectrogram"]

        texts = [b["text"] for b in batch]
        text_encoded = [b["text_encoded"] for b in batch if b["text_encoded"] is not None]

        return {
            "audio": padded_audios,
            "audio_lengths": torch.tensor(audio_lengths),
            "spectrogram": padded_specs,
            "spectrogram_lengths": torch.tensor(spec_lengths),
            "text": texts,
            "text_encoded": text_encoded,
        }
