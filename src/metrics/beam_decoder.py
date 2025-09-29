from typing import List, Tuple
import math
import numpy as np
import torch


class BeamCTCDecoder:
    def __init__(self, blank: int = 0, beam_width: int = 8):
        self.blank = blank
        self.beam_width = beam_width

    @staticmethod
    def _topk_log_probs(logp_t: np.ndarray, k: int) -> List[Tuple[int, float]]:
        idx = np.argsort(logp_t)[-k:][::-1]
        return [(int(i), float(logp_t[i])) for i in idx]

    def decode(self, log_probs_t: np.ndarray) -> List[int]:

        T, C = log_probs_t.shape
        Beam = {(): 0.0} 
        for t in range(T):
            next_beam = {}
            topk = max(1, min(self.beam_width, C))
            candidates = self._topk_log_probs(log_probs_t[t], topk)
            for prefix, prefix_score in Beam.items():
                for symbol, logp in candidates:
                    new_prefix = prefix + (symbol,)
                    new_score = prefix_score + logp
                    if new_prefix not in next_beam or next_beam[new_prefix] < new_score:
                        next_beam[new_prefix] = new_score
            sorted_beam = sorted(next_beam.items(), key=lambda x: x[1], reverse=True)[: self.beam_width]
            Beam = dict(sorted_beam)
        best_prefix = max(Beam.items(), key=lambda x: x[1])[0]
        collapsed: List[int] = []
        prev = None
        for p in best_prefix:
            if p == prev:
                prev = p
                continue
            if p != self.blank:
                collapsed.append(int(p))
            prev = p
        return collapsed
