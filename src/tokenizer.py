from typing import List, Optional

import MeCab

from pointer_generator_network.src.vocab import PAD_TOKEN


class Tokenizer():
    def __init__(self):
        """ constructor """
        self.mc = MeCab.Tagger('-Owakati')

    def __call__(self,
                 text: str,
                 sos_token: bool = None,
                 eos_token: bool = None,
                 padding: bool = False,
                 max_length: Optional[int] = None,
                 truncation: bool = False) -> List[str]:
        tokens = self.tokenize(text)
        if sos_token is not None:
            tokens = [sos_token] + tokens
        if eos_token is not None:
            tokens = tokens + [eos_token]
        if padding and max_length is not None:
            padding_length = max_length - len(tokens)
            tokens += [PAD_TOKEN] * padding_length
        if truncation and max_length is not None:
            # if the length of tokens is over max_length, the eos token is truncated
            tokens = tokens[:max_length]
        return tokens

    def tokenize(self, text: str) -> List[str]:
        """ simple tokenization """
        return self.mc.parse(text).strip().split()