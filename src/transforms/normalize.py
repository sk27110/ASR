import torch
import torch.nn as nn


class SpectrogramNormalize(nn.Module):
    
    def __init__(self, mean: float = -20.0, std: float = 5.0):
        super().__init__()
        self.mean = mean
        self.std = std
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std
        
    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"