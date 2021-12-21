""" vocabulary module """
from typing import List

SEP_TOKEN = '。'
PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
SOS_TOKEN = '[SOS]'
EOS_TOKEN = '[EOS]'
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


class Vocab():
    def __init__(self, vocab_path: str):
        """ constructor """
        self.word2idx = {word: idx for idx, word in enumerate(SPECIAL_TOKENS)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.build_vocab(vocab_path)

    def read_vocab_file(self, vocab_path: str) -> List[str]:
        """ 
        read vocabulary file 
        a format of one word per line a is expected
        """
        vocab = list()
        with open(vocab_path) as vocab_file:
            for word in vocab_file:
                vocab.append(word.strip())
        return vocab

    def build_vocab(self, vocab_path: str) -> None:
        """ build vocabulary """
        vocab = self.read_vocab_file(vocab_path)
        for word in vocab:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def size(self) -> int:
        """ get size of vocabulary """
        return len(self.word2idx)

    def get_oovs(self, tokens: List[str]) -> List[str]:
        """ get list of out-of-vocabulary tokens """
        return list(filter(lambda token: token not in self.word2idx, tokens))

    def get_mask(self, tokens: List[str]) -> List[int]:
        """ mask 1 for padding tokens """
        return [1 if token != PAD_TOKEN else 0 for token in tokens]

    def encode(self, tokens: List[str]) -> List[int]:
        """ encode tokens """
        return [self.word2idx.get(token, self.word2idx[UNK_TOKEN]) for token in tokens]

    def decode(self, ids: List[int], oov_words: List[str] = None) -> List[str]:
        """ decode indices """
        if oov_words is not None:
            vocab_size = self.size()
            extended_idx2word = {
                **self.idx2word,
                **{i + vocab_size: word
                   for i, word in enumerate(oov_words)}
            }
            return [extended_idx2word[idx] for idx in ids]
        else:
            return [self.idx2word[idx] for idx in ids]
