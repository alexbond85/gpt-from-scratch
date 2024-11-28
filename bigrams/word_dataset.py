from typing import List, Tuple
import torch
from dataclasses import dataclass

from bigrams.tokenizer import TokenizerCharacter


@dataclass
class WordDataset:
    """Handles word data and training pair generation"""
    words: List[str]
    tokenizer: TokenizerCharacter

    def training_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for word in self.words:
            indices = self.tokenizer.encode_word(word)
            for idx1, idx2 in zip(indices, indices[1:]):
                xs.append(idx1)
                ys.append(idx2)
        return torch.tensor(xs), torch.tensor(ys)

    def __getitem__(self, idx) -> str:
        return self.words[idx]

    def __len__(self) -> int:
        return len(self.words)