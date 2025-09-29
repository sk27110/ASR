from typing import List, Optional
import numpy as np
import torch
from torch import Tensor

from .base_metric import BaseMetric
from .utils import cer
from .beam_decoder import BeamCTCDecoder


class ArgmaxCER(BaseMetric):
    def __init__(self, text_encoder, blank: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.text_encoder = text_encoder
        self.blank = blank

    def __call__(self, log_probs: Tensor, lengths: Tensor, text: List[str], **kwargs) -> float:
        if not torch.is_floating_point(log_probs):
            logp = torch.log_softmax(log_probs.float(), dim=-1)
        else:
            logp = torch.log_softmax(log_probs, dim=-1)

        preds = torch.argmax(logp.cpu(), dim=-1).numpy()  
        lengths_np = lengths.detach().cpu().numpy()
        scores = []
        for pred_seq, L, tgt in zip(preds, lengths_np, text):
            tgt_n = self.text_encoder.normalize_text(tgt)
            pred_labels = list(map(int, pred_seq[:L]))
            pred_text = self.text_encoder.ctc_decode(pred_labels)
            scores.append(cer(tgt_n, pred_text))
        return float(np.mean(scores)) if scores else 0.0


class BeamCER(BaseMetric):
    def __init__(self, text_encoder, beam_width: int = 8, blank: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.text_encoder = text_encoder
        self.decoder = BeamCTCDecoder(blank=blank, beam_width=beam_width)

    def __call__(self, log_probs: Tensor, lengths: Tensor, text: List[str], **kwargs) -> float:
        if not torch.is_floating_point(log_probs):
            logp = torch.log_softmax(log_probs.float(), dim=-1)
        else:
            logp = torch.log_softmax(log_probs, dim=-1)

        logp = logp.detach().cpu().numpy() 
        lengths_np = lengths.detach().cpu().numpy()
        scores = []
        N = logp.shape[0]
        for i in range(N):
            L = int(lengths_np[i])
            per_t = logp[i, :L, :]  
            best_labels = self.decoder.decode(per_t)
            pred_text = self.text_encoder.ctc_decode(best_labels)
            tgt_n = self.text_encoder.normalize_text(text[i])
            scores.append(cer(tgt_n, pred_text))
        return float(np.mean(scores)) if scores else 0.0
