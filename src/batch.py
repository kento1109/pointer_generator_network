from typing import Optional
import torch


class BaseBatch():
    def __init__(self, input_ids: torch.Tensor, mask: torch.Tensor):
        self.input_ids = input_ids
        self.mask = mask


class EncBatch(BaseBatch):
    def __init__(self, input_ids: torch.Tensor, extended_ids: torch.Tensor, mask: torch.Tensor,
                 max_n_oovs: int):
        super(EncBatch, self).__init__(input_ids, mask)
        self.input_ids = input_ids
        self.extended_ids = extended_ids
        self.mask = mask
        self.max_n_oovs = max_n_oovs


class DecBatch(BaseBatch):
    def __init__(self, input_ids: torch.Tensor, target_ids: torch.Tensor, mask: torch.Tensor):
        super(DecBatch, self).__init__(input_ids, mask)
        self.input_ids = input_ids
        self.target_ids = target_ids
        self.mask = mask


class SummBatch():
    def __init__(self, enc_batch: EncBatch, dec_batch: Optional[DecBatch] = None):
        self.enc = enc_batch
        self.dec = dec_batch
