from typing import List
from typing import Optional


class BaseExample():
    def __init__(self, input_ids: List[int], mask: List[int]):
        self.input_ids = input_ids
        self.mask = mask


class EncExample(BaseExample):
    def __init__(self, input_ids: List[int], extended_ids: List[int], mask: List[int],
                 oov_words: List[str]):
        super(EncExample, self).__init__(input_ids, mask)
        self.extended_ids = extended_ids
        self.oov_words = oov_words


class DecExample(BaseExample):
    def __init__(self, input_ids: List[int], target_ids: List[int], mask: List[int]):
        super(DecExample, self).__init__(input_ids, mask)
        self.target_ids = target_ids


class SummExample():
    def __init__(self, enc_example: EncExample, dec_example: Optional[DecExample] = None):
        self.enc = enc_example
        self.dec = dec_example
