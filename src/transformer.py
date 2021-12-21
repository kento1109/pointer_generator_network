import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerDecoder
from torch.nn import TransformerDecoderLayer
from torch.nn import LayerNorm


class PositionalEncoding(nn.Module):
    """
    ref. https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_encoder_layers, heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = TransformerEncoderLayer(d_model, heads)
        self.encoder_norm = LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers,
                                                      self.encoder_norm)

    def forward(self, src, mask):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        return self.transformer_encoder(src, mask)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_encoder_layers, heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = TransformerDecoderLayer(d_model, heads)
        self.encoder_norm = LayerNorm(d_model)
        self.transformer_decoder = TransformerDecoder(encoder_layers, num_encoder_layers,
                                                      self.encoder_norm)
        # if pointer_gen:
        #     self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        # #p_vocab
        # self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        # self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        return self.transformer_decoder(tgt,
                                        memory,
                                        tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_key_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask)

    # def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
    #             extra_zeros, enc_batch_extend_vocab, coverage, step):

    #     if not self.training and step == 0:
    #         h_decoder, c_decoder = s_t_1
    #         s_t_hat = torch.cat((h_decoder.view(
    #             -1, self.config.hidden_dim), c_decoder.view(-1, self.config.hidden_dim)),
    #                             1)  # B x 2*hidden_dim
    #         c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs,
    #                                                        encoder_feature, enc_padding_mask,
    #                                                        coverage)
    #         coverage = coverage_next

    #     y_t_1_embd = self.embedding(y_t_1)

    #     # print(self.embedding.weight)
    #     # if torch.isnan(y_t_1_embd).any().item():
    #     #     raise (ValueError('Nan Occured ..'))

    #     x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))

    #     self.lstm.flatten_parameters()
    #     lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

    #     # if torch.isnan(lstm_out[0, 0, 0]):
    #     # raise (ValueError)

    #     h_decoder, c_decoder = s_t
    #     s_t_hat = torch.cat((h_decoder.view(
    #         -1, self.config.hidden_dim), c_decoder.view(-1, self.config.hidden_dim)),
    #                         1)  # B x 2*hidden_dim
    #     c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs,
    #                                                            encoder_feature, enc_padding_mask,
    #                                                            coverage)

    #     if self.training or step > 0:
    #         coverage = coverage_next

    #     p_gen = None
    #     if self.config.pointer_gen:
    #         p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
    #         p_gen = self.p_gen_linear(p_gen_input)
    #         p_gen = torch.sigmoid(p_gen)

    #     output = torch.cat((lstm_out.view(-1, self.config.hidden_dim), c_t),
    #                        1)  # B x hidden_dim * 3

    #     output = self.out1(output)  # B x hidden_dim

    #     #output = F.relu(output)

    #     output = self.out2(output)  # B x vocab_size
    #     vocab_dist = F.softmax(output, dim=1)

    #     if self.config.pointer_gen:
    #         vocab_dist_ = p_gen * vocab_dist
    #         attn_dist_ = (1 - p_gen) * attn_dist

    #         if extra_zeros is not None:
    #             vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

    #         final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
    #     else:
    #         final_dist = vocab_dist

    #     return final_dist, s_t, c_t, attn_dist, p_gen, coverage


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_encoder_layers, heads):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_encoder_layers, heads)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_encoder_layers, heads)
        self.out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_key_padding_mask, tgt_mask, tgt_key_padding_mask):
        e_outputs = self.encoder(src, src_key_padding_mask)
        d_output = self.decoder(tgt=tgt,
                                memory=e_outputs,
                                tgt_mask=tgt_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=src_key_padding_mask)
        output = self.out(d_output)
        return output

    # def forward(self, batch, device):

    #     step_losses = list()
    #     outputs = list()
    #     output = None

    #     batch_size = batch.enc.input_ids.size(0)
    #     enc_outputs, enc_feature, enc_hidden = self.encoder(batch.enc.input_ids)
    #     s_t_1 = self.reduce_state(enc_hidden)
    #     c_t_1 = torch.zeros((batch_size, 2 * self.config.dec_params.hidden_dim)).to(device)
    #     extra_zeros = torch.zeros((batch_size, batch.enc.max_n_oovs)).to(device)
    #     coverage = torch.zeros(batch.enc.input_ids.size()).to(device)

    #     for di in range(self.config.dec_params.max_seq_length):
    #         if batch.dec is not None:
    #             y_t_1 = batch.dec.input_ids[:, di]  # Teacher forcing
    #         else:
    #             if di == 0:
    #                 y_t_1 = (torch.ones(batch_size, dtype=torch.long) + self.sos_tensor).to(device)
    #             else:
    #                 # Setting the next input (extended token is replaced to UNK)
    #                 y_t_1 = output.masked_fill(output >= 50000, 0)

    #         final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.decoder(
    #             y_t_1, s_t_1, enc_outputs, enc_feature, batch.enc.mask, c_t_1, extra_zeros,
    #             batch.enc.extended_ids, coverage, di)
    #         output = final_dist.max(1)[1]
    #         outputs.append(output.unsqueeze(0))

    #         if batch.dec is not None:

    #             target = batch.dec.target_ids[:, di]
    #             gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()

    #             step_loss = -torch.log(gold_probs + 1e-12)

    #             step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
    #             # step_loss += step_coverage_loss
    #             step_loss = step_loss + self.config.dec_params.cov_loss_wt * step_coverage_loss

    #             step_mask = batch.dec.mask[:, di]
    #             step_loss = step_loss * step_mask
    #             step_losses.append(step_loss)

    #         coverage = next_coverage

    #     outputs = torch.cat(outputs)

    #     if batch.dec is not None:
    #         sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
    #         batch_avg_loss = sum_losses / self.config.dec_params.max_seq_length
    #         loss = torch.mean(batch_avg_loss)

    #         loss.backward()

    #         return {"outputs": outputs, "loss": loss}
    #     else:
    #         return {"outputs": outputs, "loss": None}
