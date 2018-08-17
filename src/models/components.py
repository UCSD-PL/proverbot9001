#!/usr/bin/env python3

from typing import Dict, Any, List

class SimpleEmbedding:
    def __init__(self) -> None:
        self.tokens_to_indices = {} #type: Dict[str, int]
        self.indices_to_tokens = {} #type: Dict[int, str]
    def encode_token(self, token : str) -> int :
        if token in self.tokens_to_indices:
            return self.tokens_to_indices[token]
        else:
            new_idx = len(self.tokens_to_indices)
            self.tokens_to_indices[token] = new_idx
            self.indices_to_tokens[new_idx] = token
            return new_idx

    def decode_token(self, idx : int) -> str:
        return self.indices_to_tokens[idx]
    def num_tokens(self) -> int:
        return len(self.indices_to_tokens)
    def has_token(self, token : str) -> bool :
        return token in self.tokens_to_indices
