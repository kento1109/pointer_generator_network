import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim, padding_idx=0)

        self.lstm = nn.LSTM(config.emb_dim,
                            config.hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)

        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(embedded)
        outputs = outputs.contiguous()
        feature = outputs.view(-1, 2 * self.config.hidden_dim)  # B * t_k x 2*hidden_dim
        feature = self.W_h(feature)

        return outputs, feature, hidden


class ReduceState(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

    def forward(self, hidden):
        h, c = hidden  # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, self.config.hidden_dim * 2)
        hidden_reduced_h = torch.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, self.config.hidden_dim * 2)
        hidden_reduced_c = torch.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)
                )  # h, c dim = 1 x b x hidden_dim


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # attention
        self.config = config
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())
        dec_fea = self.decode_proj(s_t_hat)  # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k,
                                                       n).contiguous()  # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded  # B * t_k x 2*hidden_dim
        if self.config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features)  # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask  # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, self.config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if self.config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.attention_network = Attention(config)
        # decoder
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim, padding_idx=0)

        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim,
                            config.hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        #p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                extra_zeros, enc_batch_extend_vocab, coverage, step):

        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(
                -1, self.config.hidden_dim), c_decoder.view(-1, self.config.hidden_dim)),
                                1)  # B x 2*hidden_dim
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs,
                                                           encoder_feature, enc_padding_mask,
                                                           coverage)
            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)

        # print(self.embedding.weight)
        # if torch.isnan(y_t_1_embd).any().item():
        #     raise (ValueError('Nan Occured ..'))

        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))

        self.lstm.flatten_parameters()
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        # if torch.isnan(lstm_out[0, 0, 0]):
        # raise (ValueError)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(
            -1, self.config.hidden_dim), c_decoder.view(-1, self.config.hidden_dim)),
                            1)  # B x 2*hidden_dim
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs,
                                                               encoder_feature, enc_padding_mask,
                                                               coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if self.config.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, self.config.hidden_dim), c_t),
                           1)  # B x hidden_dim * 3

        output = self.out1(output)  # B x hidden_dim

        #output = F.relu(output)

        output = self.out2(output)  # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if self.config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage


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
