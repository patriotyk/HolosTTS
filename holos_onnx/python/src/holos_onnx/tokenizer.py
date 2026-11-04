from __future__ import annotations

from typing import Iterable, List, Sequence, Union

import numpy as np


class Tokenizer:
    """Character tokenizer compatible with the training-time Holos tokenizer."""

    def __init__(self, vocab: Sequence[str]):
        self.vocab = list(vocab)
        self._token_to_id = {
            symbol: index for index, symbol in enumerate(self.vocab)
        }

    def encode(self, text: str) -> List[int]:
        return [
            self._token_to_id[symbol]
            for symbol in text
            if symbol in self._token_to_id
        ]

    __call__ = encode

    def encode_with_boundaries(self, text: str) -> np.ndarray:
        return np.asarray([0, *self.encode(text), 0], dtype=np.int64)

    def decode(
        self,
        tokens: Union[int, np.integer, Iterable[int]],
        as_list: bool = False,
    ):
        if isinstance(tokens, (int, np.integer)):
            token_values = [int(tokens)]
        else:
            token_values = [int(token) for token in tokens]

        decoded = []
        for token in token_values:
            if token == 0:
                decoded.append("$")
            elif 0 <= token < len(self.vocab):
                decoded.append(self.vocab[token])
            else:
                decoded.append("$")
        return decoded if as_list else "".join(decoded)

