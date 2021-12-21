import copy
import glob
import json
import os
from typing import Dict, List, Union

import numpy as np
from logzero import logger
from omegaconf import OmegaConf
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.optim import Adagrad
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from rouge_score import rouge_scorer

import pointer_generator_network
from pointer_generator_network.src.vocab import Vocab
from pointer_generator_network.src.example import EncExample
from pointer_generator_network.src.example import DecExample
from pointer_generator_network.src.example import SummExample
from pointer_generator_network.src.batch import EncBatch
from pointer_generator_network.src.batch import DecBatch
from pointer_generator_network.src.batch import SummBatch
from pointer_generator_network.src.vocab import SOS_TOKEN
from pointer_generator_network.src.vocab import EOS_TOKEN
from pointer_generator_network.src.vocab import UNK_TOKEN
from pointer_generator_network.src.vocab import SEP_TOKEN
from pointer_generator_network.src.writer import MlflowWriter

BASE_PATH = pointer_generator_network.__path__[0]

ROUGE_PATTERNS = ['rouge1', 'rouge2', 'rougeL']


class Trainer():
    """ a manager class for training """
    def __init__(self, model, tokenizer, src_vocab, tgt_vocab):
        """ constructor """
        self.model = model
        self.tokenizer = tokenizer
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device('cpu')
        # self.model.to(self.device)
        self.config = OmegaConf.load(os.path.join(BASE_PATH, 'config.yaml'))
        self.optimizer = None

        self.writer = None

        if self.config.write_hydra:
            self.writer = MlflowWriter('pointer_generator_network')
            self.writer.set_tag("tag", self.config.tag)

        self.model.to(self.device)

        if self.config.load_model_state:
            model_path = os.path.join(self.config.model_dir, self.config.load_model_name)
            # self.model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
            self.model.load_state_dict(torch.load(model_path), strict=False)

        if self.config.do_evaluate:
            self.scorer = rouge_scorer.RougeScorer(ROUGE_PATTERNS)

    def _create_optimizer(self) -> None:
        """ set optimizer """
        params = list(self.model.encoder.parameters()) + list(
            self.model.decoder.parameters()) + list(self.model.reduce_state.parameters())
        initial_lr = self.config.dec_params.lr_coverage if self.config.dec_params.is_coverage else self.config.comm_params.lr
        self.optimizer = Adagrad(params,
                                 lr=initial_lr,
                                 initial_accumulator_value=self.config.comm_params.adagrad_init_acc)
        # self.optimizer = Adam(params, lr=initial_lr)

    def _prepare_enc_example(self, tokens: List[str], vocab: Vocab) -> EncExample:
        def _prepare_extended_ids_and_oov_words():
            extend_ids = list()
            oov_words = list()
            vocab_size = vocab.size()
            unk_idx = vocab.word2idx[UNK_TOKEN]
            for token, input_id in zip(tokens, input_ids):
                if input_id == unk_idx:
                    if token not in oov_words:
                        oov_words.append(token)
                    n_oovs = oov_words.index(token)
                    extend_ids.append(vocab_size + n_oovs)
                else:
                    extend_ids.append(input_id)
            return extend_ids, oov_words

        input_ids = vocab.encode(tokens)
        extended_ids, oov_words = _prepare_extended_ids_and_oov_words()
        mask = vocab.get_mask(tokens)
        return EncExample(input_ids, extended_ids, mask, oov_words)

    def _prepare_dec_example(self, input_tokens: List[str], target_tokens: List[str], vocab: Vocab,
                             oov_words: List[str]) -> DecExample:
        def _replace_oovs():
            vocab_size = vocab.size()
            unk_idx = vocab.word2idx[UNK_TOKEN]
            for i, (target_token, target_id) in enumerate(zip(target_tokens, target_ids)):
                if (target_id == unk_idx) and (target_token in oov_words):
                    target_ids[i] = vocab_size + oov_words.index(target_token)

        input_ids = vocab.encode(input_tokens)
        target_ids = vocab.encode(target_tokens)
        _replace_oovs()
        mask = vocab.get_mask(input_tokens)
        return DecExample(input_ids, target_ids, mask)

    def convert_article_to_example(self, article_dict: Dict[str, str]) -> SummExample:
        """ convert article dicts to example """
        src_text = article_dict['body']
        enc_input_tokens = self.tokenizer(src_text,
                                          padding=True,
                                          max_length=self.config.enc_params.max_seq_length,
                                          truncation=True)
        # store data as an example instance
        enc_example = self._prepare_enc_example(enc_input_tokens, self.src_vocab)
        example = SummExample(enc_example, None)

        if "summary" in article_dict:
            tgt_text = self.concat_summary(article_dict['summary'])

            dec_input_tokens = self.tokenizer(tgt_text,
                                              sos_token=SOS_TOKEN,
                                              padding=True,
                                              max_length=self.config.dec_params.max_seq_length,
                                              truncation=True)
            dec_target_tokens = self.tokenizer(tgt_text,
                                               eos_token=EOS_TOKEN,
                                               padding=True,
                                               max_length=self.config.dec_params.max_seq_length,
                                               truncation=True)
            dec_example = self._prepare_dec_example(dec_input_tokens, dec_target_tokens,
                                                    self.tgt_vocab, enc_example.oov_words)
            example = SummExample(enc_example, dec_example)

        return example

    @staticmethod
    def load_from_jsons(data_path_pattern: str) -> List[Dict[str, str]]:
        """ load json files """
        logger.info("load json files ..")
        json_paths = glob.glob(os.path.join(data_path_pattern))
        article_dicts = list()
        for json_path in json_paths:
            with open(json_path) as json_file:
                article_dicts.extend(json.load(json_file))
        return article_dicts

    @staticmethod
    def concat_summary(summaries: List[str]) -> str:
        """ concat summaries to a chunk """
        return SEP_TOKEN.join(summaries)

    def build_data_loader(self, examples: List[SummExample], shuffle: bool = False) -> DataLoader:
        """ build data loader from examples"""
        return DataLoader(examples,
                          batch_size=self.config.comm_params.batch_size,
                          shuffle=shuffle,
                          collate_fn=self._collate_examples)

    def _collate_examples(self, examples: List[SummExample]) -> SummBatch:
        """ collate lists of samples into batch """
        enc_examples = list(map(lambda example: example.enc, examples))
        enc_batch = self.enc_examples_to_batch(enc_examples)
        batch = SummBatch(enc_batch, None)
        if examples[0].dec is not None:
            dec_examples = list(map(lambda example: example.dec, examples))
            dec_batch = self.dec_examples_to_batch(dec_examples)
            batch = SummBatch(enc_batch, dec_batch)
        return batch

    @staticmethod
    def enc_examples_to_batch(examples: List[EncExample]) -> EncBatch:
        """ convert encoder examples to tensor """
        enc_dict = dict()
        enc_dict['input_ids'] = torch.tensor([example.input_ids for example in examples],
                                             dtype=torch.long)
        enc_dict['extended_ids'] = torch.tensor([example.extended_ids for example in examples],
                                                dtype=torch.long)
        enc_dict['mask'] = torch.tensor([example.mask for example in examples], dtype=torch.long)
        # determine the max number of in-article OOVs in this batch
        enc_dict['max_n_oovs'] = max(list(map(lambda example: len(example.oov_words), examples)))
        return EncBatch(**enc_dict)

    @staticmethod
    def dec_examples_to_batch(examples: List[DecExample]) -> DecBatch:
        """ convert decoder examples to tensor """
        dec_dict = dict()
        dec_dict['input_ids'] = torch.tensor([example.input_ids for example in examples],
                                             dtype=torch.long)
        dec_dict['target_ids'] = torch.tensor([example.target_ids for example in examples],
                                              dtype=torch.long)
        dec_dict['mask'] = torch.tensor([example.mask for example in examples], dtype=torch.long)
        return DecBatch(**dec_dict)

    def _to_device_batch(self, batch: Union[SummBatch, EncBatch]) -> None:
        """ transfer data to device"""
        def _to_device(batch):
            for key, value in vars(batch).items():
                if isinstance(value, torch.Tensor):
                    setattr(batch, key, value.to(self.device))

        if isinstance(batch, EncBatch):
            _to_device(batch)
        else:
            for batch_ in [batch.enc, batch.dec]:
                if batch_ is not None:
                    _to_device(batch_)

    def train(self, data_loader):
        """ runs optimization """

        self._create_optimizer()

        self.model.train()

        n_iterations = 0

        prev_model = copy.deepcopy(self.model)
        prev_optimizer = copy.deepcopy(self.optimizer)

        for epoch in tqdm(range(self.config.comm_params.num_epochs)):

            epoch_loss = 0.0

            for batch in data_loader:

                self.optimizer.zero_grad()

                self._to_device_batch(batch)
                result = self.model(batch, self.device)
                # epoch_loss += loss

                if torch.isnan(result["loss"]):
                    # ignore this section
                    logger.warning('loss is Nan. we load previous state')
                    self.model = prev_model
                    self._create_optimizer()
                    self.optimizer.load_state_dict(prev_optimizer.state_dict())
                else:
                    prev_model = copy.deepcopy(self.model)
                    prev_optimizer = copy.deepcopy(self.optimizer)

                clip_grad_norm_(self.model.encoder.parameters(),
                                self.config.comm_params.max_grad_norm)
                clip_grad_norm_(self.model.decoder.parameters(),
                                self.config.comm_params.max_grad_norm)
                clip_grad_norm_(self.model.reduce_state.parameters(),
                                self.config.comm_params.max_grad_norm)

                self.optimizer.step()

                logger.info(f'loss {result["loss"].item():.4f} [{n_iterations + 1} iterations]')

                if self.writer is not None:
                    self.writer.log_metric("loss", result["loss"].item(), n_iterations)

                n_iterations += 1

            epoch_loss /= len(data_loader)
            # logger.info(f'{epoch + 1} epoch iteration is finished with loss {epoch_loss:.4f}')

        if self.config.save_model:
            self.save_model()

    def save_model(self):
        """ save model state """
        model_path = os.path.join(self.config.model_dir, self.writer.run_id + ".pt")
        state = self.model.state_dict()
        torch.save(state, model_path)

    def evaluate(self, article_dicts) -> Dict[str, float]:
        """ evaluate summarised result by ROUGE """
        rouge_score_dict = defaultdict(list)
        for i, article_dict in tqdm(enumerate(article_dicts)):
            example = self.convert_article_to_example(article_dict)
            reference = self.concat_summary(article_dict['summary'])
            decoded_tokens = self.decode(example)
            rouge_score = self.scorer.score(' '.join(self.tokenizer(reference)),
                                            ' '.join(decoded_tokens))
            for key, item in rouge_score.items():
                rouge_score_dict[key].append(item.fmeasure)
            # if rouge_score_dict['rouge2'][-1] < 0.1:
            # print(i, article_dict)
        mean_scores = {key: np.mean(scores) for key, scores in rouge_score_dict.items()}
        logger.info(mean_scores)
        return mean_scores

    def sort_beams(self, beams):
        """ sort beam results """
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def decode(self, example):
        """ beam search decoding """
        enc_batch = self.enc_examples_to_batch([example.enc] * self.config.beam_size)
        batch = SummBatch(enc_batch=enc_batch)

        self.model.eval()

        # Run beam search to get best Hypothesis
        with torch.no_grad():
            best_summary = self.beam_search(batch)

        decoded_tokens = self.tgt_vocab.decode(best_summary.tokens[1:], example.enc.oov_words)

        # Remove the [STOP] token from decoded_words, if necessary
        try:
            fst_stop_idx = decoded_tokens.index(EOS_TOKEN)
            decoded_tokens = decoded_tokens[:fst_stop_idx]
        except ValueError:
            decoded_tokens = decoded_tokens

        return decoded_tokens

    def beam_search(self, batch):

        self._to_device_batch(batch)

        batch_size = batch.enc.input_ids.size(0)
        enc_outputs, enc_feature, enc_hidden = self.model.encoder(batch.enc.input_ids)
        s_t_0 = self.model.reduce_state(enc_hidden)
        c_t_0 = torch.zeros((batch_size, 2 * self.config.dec_params.hidden_dim)).to(self.device)
        extra_zeros = torch.zeros((batch_size, batch.enc.max_n_oovs)).to(self.device)
        coverage_t_0 = torch.zeros(batch.enc.input_ids.size()).to(self.device)

        dec_h, dec_c = s_t_0  # 1 x 2*hidden_size

        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        #decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [
            Beam(tokens=[self.tgt_vocab.word2idx[SOS_TOKEN]],
                 log_probs=[0.0],
                 state=(dec_h[0], dec_c[0]),
                 context=c_t_0[0],
                 coverage=(coverage_t_0[0] if self.config.dec_params.is_coverage else None))
            for _ in range(self.config.beam_size)
        ]
        results = []
        steps = 0
        while steps < self.config.dec_params.max_seq_length and len(
                results) < self.config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [
                t if t < self.tgt_vocab.size() else self.tgt_vocab.word2idx[UNK_TOKEN]
                for t in latest_tokens
            ]

            y_t_1 = torch.LongTensor(latest_tokens).to(self.device)

            all_state_h = []
            all_state_c = []

            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c,
                                                                           0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if self.config.dec_params.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, _, _, coverage_t = self.model.decoder(
                y_t_1, s_t_1, enc_outputs, enc_feature, batch.enc.mask, c_t_1, extra_zeros,
                batch.enc.extended_ids, coverage_t_1, steps)

            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, self.config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if self.config.dec_params.is_coverage else None)

                for j in range(self.config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                        log_prob=topk_log_probs[i, j].item(),
                                        state=state_i,
                                        context=context_i,
                                        coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.tgt_vocab.word2idx[EOS_TOKEN]:
                    if steps >= self.config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == self.config.beam_size or len(results) == self.config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]


class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage

    def extend(self, token, log_prob, state, context, coverage):
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    state=state,
                    context=context,
                    coverage=coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)