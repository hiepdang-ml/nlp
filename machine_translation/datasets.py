from typing import Callable, List, Tuple, Dict, Literal, TextIO, Optional
from collections import Counter

import torch
from torch.utils.data import Dataset

from .utils import Vocabularies


class EnglishFrenchDataset(Dataset):

    def __init__(
        self, 
        txt_path: str, 
        max_samples: Optional[int] = None,
        num_steps: int = 10,
        device: torch.device = torch.device('cpu'),
    ) -> None:
        
        self.txt_path: str = txt_path
        self.max_samples: Optional[int] = max_samples
        self.num_steps: int = num_steps
        self.device: torch.device = device
        file: TextIO

        with open(file=txt_path, encoding='utf-8') as file:
            raw_text: str = file.read()
        
        self.source: List[List[str]]
        self.target: List[List[str]]
        self.source, self.target = self.__tokenize(text=self.__preprocess(text=raw_text))
        
        self.source_vocab: Vocabularies
        self.source_array: torch.Tensor
        self.source_valid_len: torch.Tensor
        self.source_vocab, self.source_array, self.source_valid_len = self.__build_array(docs_of_tokens=self.source)
        
        self.target_vocab: Vocabularies
        self.target_array: torch.Tensor
        self.target_valid_len: torch.Tensor
        self.target_vocab, self.target_array, self.target_valid_len = self.__build_array(docs_of_tokens=self.target)
        
    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, index: int) -> Dict[Literal['source', 'target'], Tuple[torch.Tensor, int]]:
        return {
            'source': (self.source_array[index], self.source_valid_len[index]),
            'target': (self.target_array[index], self.target_valid_len[index]),
        }

    @staticmethod
    def __preprocess(text: str) -> str:
        # Replace non-breaking space with space
        text: str = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
        # Insert space between words and punctuation marks
        no_space: Callable[[str, str], bool] = lambda char, prev_char: char in ',.!?' and prev_char != ' '
        out: List[str] = [
            ' ' + text[i]
            if i > 0 and no_space(char=text[i], prev_char=text[i - 1]) 
            else text[i]
            for i in range(len(text))
        ]
        return ''.join(out)

    def __tokenize(self, text: str) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Tokenize by word
        """
        sources: List[List[str]] = []
        targets: List[List[str]] = []
        for line in text.split('\n')[:self.max_samples]:
            parts: List[str] = line.split('\t')
            if len(parts) == 2:
                sources.append([t for t in f'<bos> {parts[0]} <eos>'.split()])
                targets.append([t for t in f'<bos> {parts[1]} <eos>'.split()])
        
        return sources, targets
    
    def __build_array(self, docs_of_tokens: List[List[str]]) -> Tuple[Vocabularies, torch.Tensor, torch.Tensor]:
        pad_or_truncate: Callable[[List[str], int], List[str]] = lambda tokens: (
            tokens[:self.num_steps] 
            if len(tokens) > self.num_steps
            else tokens + ['<pad>'] * (self.num_steps - len(tokens))
        )
        docs_of_tokens: List[List[str]] = [pad_or_truncate(doc) for doc in docs_of_tokens]
        vocab = Vocabularies(tokens=[token for doc in docs_of_tokens for token in doc], min_freq=2)
        array: torch.Tensor = torch.tensor(
            [vocab.find_ids(tokens=doc) for doc in docs_of_tokens],
            device=self.device,
            dtype=torch.int64,
        )
        valid_len: torch.Tensor = (array != vocab.find_ids(tokens=['<pad>'])[0]).type(torch.int32).sum(dim=1)
        return vocab, array, valid_len


if __name__ == '__main__':
    self = EnglishFrenchDataset(txt_path="../data/fra-eng/fra.txt", num_steps=10, device=torch.device('cpu'))
