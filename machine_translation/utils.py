from functools import cached_property
from typing import List, Tuple, Dict
from collections import Counter


class Vocabularies:

    def __init__(self, tokens: List[str], min_freq: int, reserved_tokens: List[str] = []):
        self.reserved_tokens: List[str] = reserved_tokens
        # Count token frequencies
        counter: Counter = Counter(tokens)
        self.token_freqs: List[Tuple[str, int]] = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # The list of unique tokens
        self.ids_to_tokens: Dict[int, str] = dict(
            enumerate(set(['<unk>'] + reserved_tokens + [token for token, freq in self.token_freqs if freq >= min_freq]))
        )
        self.tokens_to_ids: Dict[str, int] = dict((v, k) for k, v in self.ids_to_tokens.items())

    def find_ids(self, tokens: List[str]) -> List[int]:
        return [self.tokens_to_ids.get(token, self.unk) for token in tokens]

    def find_tokens(self, ids: List[int]) -> List[str]:
        return [self.ids_to_tokens[idx] for idx in ids]

    def __len__(self):
        return len(self.ids_to_tokens)

    @cached_property
    def unk(self) -> int:
        return self.tokens_to_ids['<unk>']
    



