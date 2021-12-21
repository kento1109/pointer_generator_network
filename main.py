import hydra
from omegaconf import OmegaConf
import random

import numpy as np
import torch

from pointer_generator_network.src.trainer import Trainer
from pointer_generator_network.src.vocab import Vocab
from pointer_generator_network.src.vocab import SOS_TOKEN
from pointer_generator_network.src.tokenizer import Tokenizer
from pointer_generator_network.src.rnn import Summarizer
from pointer_generator_network.src.writer import MlflowWriter


def set_seed(seed: int) -> None:
    """ set random seed """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(config_name="config")
def main(cfg):

    src_vocab = Vocab(cfg.source_vocab_path)
    tgt_vocab = Vocab(cfg.target_vocab_path)
    tokenizer = Tokenizer()
    model = Summarizer(cfg, tgt_vocab.word2idx[SOS_TOKEN])
    trainer = Trainer(model, tokenizer, src_vocab, tgt_vocab)

    if cfg.do_train:
        article_dicts = trainer.load_from_jsons(cfg.train_data_path)
        examples = list(map(trainer.convert_article_to_example, article_dicts))
        data_loader = trainer.build_data_loader(examples, shuffle=True)
        set_seed(1)
        trainer.train(data_loader)

    if cfg.do_evaluate:
        article_dicts = trainer.load_from_jsons(cfg.test_data_path)
        trainer.evaluate(article_dicts)


if __name__ == "__main__":
    main()
