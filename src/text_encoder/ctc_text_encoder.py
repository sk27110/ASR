import re
from typing import List, Optional
import torch


class CTCTextEncoder:
    def __init__(self, alphabet: Optional[List[str]] = None):
        self.blank_token = "" 
        
        if alphabet is None:
            alphabet = list("abcdefghijklmnopqrstuvwxyz ")
        
        self.alphabet = alphabet
        self.vocab = [self.blank_token] + self.alphabet
        
        self.index_to_char = dict(enumerate(self.vocab))
        self.char_to_index = {char: idx for idx, char in self.index_to_char.items()}

    def __len__(self) -> int:
        return len(self.vocab)

    def __getitem__(self, index: int) -> str:
        if index not in self.index_to_char:
            raise KeyError(f"Index {index} not in vocabulary")
        return self.index_to_char[index]

    @property
    def blank_index(self) -> int:
        return self.char_to_index[self.blank_token]

    def encode(self, text: str) -> torch.Tensor:

        text = self.normalize_text(text)
        
        indices = []
        for char in text:
            if char not in self.char_to_index:
                unknown_chars = [c for c in text if c not in self.char_to_index]
                raise ValueError(
                    f"Text '{text}' contains unknown characters: {unknown_chars}. "
                    f"Vocabulary: {self.alphabet}"
                )
            indices.append(self.char_to_index[char])
        
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    def decode(self, indices: List[int]) -> str:
        return "".join([self.index_to_char[int(idx)] for idx in indices]).strip()

    def ctc_decode(self, indices: List[int]) -> str:
        decoded_chars = []
        previous_index = self.blank_index
        
        for current_index in indices:
            current_index = int(current_index)
            
            if current_index == previous_index:
                continue
            
            if current_index != self.blank_index:
                decoded_chars.append(self.index_to_char[current_index])
            
            previous_index = current_index
        
        return "".join(decoded_chars)

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def get_valid_characters(self) -> List[str]:
        return self.alphabet.copy()