import torch
import torch.nn as nn
import torch.nn.functional as F

from pointer_generator_network.src.rnn import Encoder
from pointer_generator_network.src.rnn import Encoder
from pointer_generator_network.src.rnn import Encoder


class Summarizer(nn.Module):
    def __init__(self, config, sos_idx=None):
        super().__init__()

        self.config = config

        self.encoder = Encoder(config.enc_params)
        self.reduce_state = ReduceState(config.enc_params)
        self.decoder = Decoder(config.dec_params)

        # shared the embedding between encoder and decoder
        self.decoder.embedding.weight = self.encoder.embedding.weight

        self.sos_tensor = torch.tensor([sos_idx], dtype=torch.long)

    def forward(self, batch, device):

        step_losses = list()
        outputs = list()
        output = None

        batch_size = batch.enc.input_ids.size(0)
        enc_outputs, enc_feature, enc_hidden = self.encoder(batch.enc.input_ids)
        s_t_1 = self.reduce_state(enc_hidden)
        c_t_1 = torch.zeros((batch_size, 2 * self.config.dec_params.hidden_dim)).to(device)
        extra_zeros = torch.zeros((batch_size, batch.enc.max_n_oovs)).to(device)
        coverage = torch.zeros(batch.enc.input_ids.size()).to(device)

        for di in range(self.config.dec_params.max_seq_length):
            if batch.dec is not None:
                y_t_1 = batch.dec.input_ids[:, di]  # Teacher forcing
            else:
                if di == 0:
                    y_t_1 = (torch.ones(batch_size, dtype=torch.long) + self.sos_tensor).to(device)
                else:
                    # Setting the next input (extended token is replaced to UNK)
                    y_t_1 = output.masked_fill(output >= 50000, 0)

            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.decoder(
                y_t_1, s_t_1, enc_outputs, enc_feature, batch.enc.mask, c_t_1, extra_zeros,
                batch.enc.extended_ids, coverage, di)
            output = final_dist.max(1)[1]
            outputs.append(output.unsqueeze(0))

            if batch.dec is not None:

                target = batch.dec.target_ids[:, di]
                gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()

                step_loss = -torch.log(gold_probs + 1e-12)

                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                # step_loss += step_coverage_loss
                step_loss = step_loss + self.config.dec_params.cov_loss_wt * step_coverage_loss

                step_mask = batch.dec.mask[:, di]
                step_loss = step_loss * step_mask
                step_losses.append(step_loss)

            coverage = next_coverage

        outputs = torch.cat(outputs)

        if batch.dec is not None:
            sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
            batch_avg_loss = sum_losses / self.config.dec_params.max_seq_length
            loss = torch.mean(batch_avg_loss)

            loss.backward()

            return {"outputs": outputs, "loss": loss}
        else:
            return {"outputs": outputs, "loss": None}
