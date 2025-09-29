import torch
from torch import nn


class DeepSpeechLike(nn.Module):

    def __init__(
        self,
        n_feats: int,
        n_tokens: int,
        hidden_size: int = 512,
        rnn_layers: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32,
                kernel_size=11, stride=2, padding=5
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32, out_channels=32,
                kernel_size=11, stride=2, padding=5
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        rnn_input_size = (n_feats // 4) * 32 

        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_tokens),
        )


    def forward(self, spectrogram: torch.Tensor, spectrogram_length: torch.Tensor, **batch):
        x = spectrogram.unsqueeze(1)

        x = self.cnn(x)           
        B, C, T, F = x.shape

      
        x = x.permute(0, 2, 1, 3).contiguous()  
        x = x.view(B, T, C * F)                 

        x, _ = self.rnn(x)
        x = self.fc(x)

        log_probs = nn.functional.log_softmax(x, dim=-1)

        log_probs_length = self.transform_input_lengths(spectrogram_length)

        return {"log_probs": log_probs, "log_probs_length": log_probs_length}


    def transform_input_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        return (input_lengths // 2) // 2

    def __str__(self):
        all_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        info = super().__str__()
        info += f"\nAll parameters: {all_params}"
        info += f"\nTrainable parameters: {trainable}"
        return info
