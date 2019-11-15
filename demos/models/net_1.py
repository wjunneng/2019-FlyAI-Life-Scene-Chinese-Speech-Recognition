# -*- coding: utf-8 -*
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Util(object):
    @staticmethod
    def pad_list(xs, pad_value):
        """
        :param xs:
        :param pad_value:
        :return:
        """
        # From: espnet/src/nets/e2e_asr_th.py: pad_list()
        n_batch = len(xs)
        max_len = max(x.size(0) for x in xs)
        pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
        for i in range(n_batch):
            pad[i, :xs[i].size(0)] = xs[i]
        return pad

    @staticmethod
    def get_attn_key_pad_mask(seq_k, seq_q, pad_idx):
        """
            For masking out the padding part of key sequence.
        """
        # Expand to fit the shape of key query attention matrix.
        len_q = seq_q.size(1)
        padding_mask = seq_k.eq(pad_idx)
        padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

        return padding_mask

    @staticmethod
    def get_subsequent_mask(seq):
        """
            For masking out the subsequent info.
        """

        sz_b, len_s = seq.size()
        subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

        return subsequent_mask

    @staticmethod
    def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
        """
            padding position is set to 0, either use input_lengths or pad_idx
        """
        assert input_lengths is not None or pad_idx is not None
        if input_lengths is not None:
            # padded_input: N x T x ..
            N = padded_input.size(0)
            non_pad_mask = padded_input.new_ones(padded_input.size()[:-1])  # N x T
            for i in range(N):
                non_pad_mask[i, input_lengths[i]:] = 0
        if pad_idx is not None:
            # padded_input: N x T
            assert padded_input.dim() == 2
            non_pad_mask = padded_input.ne(pad_idx).float()
        # unsqueeze(-1) for broadcast
        return non_pad_mask.unsqueeze(-1)

    @staticmethod
    def get_attn_pad_mask(padded_input, input_lengths, expand_length):
        """
            mask position is set to 1
        """
        # N x Ti x 1
        non_pad_mask = Util.get_non_pad_mask(padded_input, input_lengths=input_lengths)
        # N x Ti, lt(1) like not operation
        pad_mask = non_pad_mask.squeeze(-1).lt(1)
        attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
        return attn_mask

    @staticmethod
    def cal_performance(pad_id, pred, gold, smoothing=0.0):
        """Calculate cross entropy loss, apply label smoothing if needed.
        Args:
            pred: N x T x C, score before softmax
            gold: N x T
        """

        pred = pred.view(-1, pred.size(2))
        gold = gold.contiguous().view(-1)

        loss = Util.cal_loss(pad_id, pred, gold, smoothing)

        pred = pred.max(1)[1]
        non_pad_mask = gold.ne(pad_id)
        n_correct = pred.eq(gold)
        n_correct = n_correct.masked_select(non_pad_mask).sum().item()

        return loss, n_correct

    @staticmethod
    def cal_loss(pad_id, pred, gold, smoothing):
        """
            Calculate cross entropy loss, apply label smoothing if needed.
        """

        if smoothing > 0.0:
            eps = smoothing
            n_class = pred.size(1)

            # Generate one-hot matrix: N x C.
            # Only label position is 1 and all other positions are 0
            # gold include -1 value (IGNORE_ID) and this will lead to assert error
            gold_for_scatter = gold.ne(pad_id).long() * gold
            one_hot = torch.zeros_like(pred).scatter(1, gold_for_scatter.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / n_class
            log_prb = F.log_softmax(pred, dim=1)

            non_pad_mask = gold.ne(pad_id)
            n_word = non_pad_mask.sum().item()
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum() / n_word
        else:
            loss = F.cross_entropy(pred, gold, ignore_index=pad_id, reduction='elementwise_mean')

        return loss


class TransformerOptimizer(object):
    """
        A simple wrapper class for learning rate scheduling
    """

    def __init__(self, optimizer, warmup_steps=4000, k=0.2):
        self.optimizer = optimizer
        self.k = k
        self.warmup_steps = warmup_steps
        d_model = 512
        self.init_lr = d_model ** (-0.5)
        self.lr = self.init_lr
        self.warmup_steps = warmup_steps
        self.k = k
        self.step_num = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._update_lr()
        self.optimizer.step()

    def _update_lr(self):
        self.step_num += 1
        self.lr = self.k * self.init_lr * min(self.step_num ** (-0.5),
                                              self.step_num * (self.warmup_steps ** (-1.5)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr


class MultiHeadAttention(nn.Module):
    """
        Multi-Head Attention module
    """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5),
                                                   attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    """
        Scaled Dot-Product Attention
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask.bool(), -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class Decoder(nn.Module):
    """
        A decoder model with self attention mechanism.
    """

    def __init__(self, pad_id, sos_id, eos_id, n_tgt_vocab=4335, d_word_vec=512, n_layers=6, n_head=8, d_k=64, d_v=64,
                 d_model=512, d_inner=2048, dropout=0.1, tgt_emb_prj_weight_sharing=True, pe_maxlen=5000):
        super(Decoder, self).__init__()
        # parameters
        self.pad_id = pad_id
        # Start of Sentence
        self.sos_id = sos_id
        # End of Sentence
        self.eos_id = eos_id
        self.n_tgt_vocab = n_tgt_vocab
        self.d_word_vec = d_word_vec
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.tgt_emb_prj_weight_sharing = tgt_emb_prj_weight_sharing
        self.pe_maxlen = pe_maxlen

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList(
            [DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def preprocess(self, padded_input):
        """Generate decoder input and output label from padded_input
        Add <sos> to decoder input, and add <eos> to decoder output label
        """
        ys = [y[y != self.pad_id] for y in padded_input]  # parse padded ys
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos_id])
        sos = ys[0].new([self.sos_id])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = Util.pad_list(ys_in, self.eos_id)
        ys_out_pad = Util.pad_list(ys_out, self.pad_id)
        assert ys_in_pad.size() == ys_out_pad.size()
        return ys_in_pad, ys_out_pad

    def forward(self, padded_input, encoder_padded_outputs,
                encoder_input_lengths, return_attns=False):
        """
        Args:
            padded_input: N x To
            encoder_padded_outputs: N x Ti x H
        Returns:
        """
        dec_slf_attn_list, dec_enc_attn_list = [], []

        # Get Deocder Input and Output
        ys_in_pad, ys_out_pad = self.preprocess(padded_input)

        # Prepare masks
        non_pad_mask = Util.get_non_pad_mask(ys_in_pad, pad_idx=self.eos_id)

        slf_attn_mask_keypad = Util.get_attn_key_pad_mask(seq_k=ys_in_pad,
                                                          seq_q=ys_in_pad,
                                                          pad_idx=self.eos_id)
        slf_attn_mask_subseq = Util.get_subsequent_mask(ys_in_pad).type_as(slf_attn_mask_keypad)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        output_length = ys_in_pad.size(1)
        dec_enc_attn_mask = Util.get_attn_pad_mask(encoder_padded_outputs,
                                                   encoder_input_lengths,
                                                   output_length)

        # Forward
        dec_output = self.dropout(self.tgt_word_emb(ys_in_pad) * self.x_logit_scale +
                                  self.positional_encoding(ys_in_pad))

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, encoder_padded_outputs,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        # before softmax
        seq_logit = self.tgt_word_prj(dec_output)

        # Return
        pred, gold = seq_logit, ys_out_pad

        if return_attns:
            return pred, gold, dec_slf_attn_list, dec_enc_attn_list
        return pred, gold

    def recognize_beam(self, encoder_outputs, char_list, args):
        """Beam search, decode one utterence now.
        Args:
            encoder_outputs: T x H
            char_list: list of character
            args: args.beam
        Returns:
            nbest_hyps:
        """
        # search params
        beam = args.beam_size
        nbest = args.nbest
        if args.decode_max_len == 0:
            maxlen = encoder_outputs.size(0)
        else:
            maxlen = args.decode_max_len

        encoder_outputs = encoder_outputs.unsqueeze(0)

        # prepare sos
        ys = torch.ones(1, 1).fill_(self.sos_id).type_as(encoder_outputs).long()

        # yseq: 1xT
        hyp = {'score': 0.0, 'yseq': ys}
        hyps = [hyp]
        ended_hyps = []

        for i in range(maxlen):
            hyps_best_kept = []
            for hyp in hyps:
                ys = hyp['yseq']  # 1 x i
                # last_id = ys.cpu().numpy()[0][-1]
                # freq = bigram_freq[last_id]
                # freq = torch.log(torch.from_numpy(freq))
                # # print(freq.dtype)
                # freq = freq.type(torch.float).to(device)
                # print(freq.dtype)
                # print('freq.size(): ' + str(freq.size()))
                # print('freq: ' + str(freq))
                # -- Prepare masks
                non_pad_mask = torch.ones_like(ys).float().unsqueeze(-1)  # 1xix1
                slf_attn_mask = Util.get_subsequent_mask(ys)

                # -- Forward
                dec_output = self.dropout(
                    self.tgt_word_emb(ys) * self.x_logit_scale +
                    self.positional_encoding(ys))

                for dec_layer in self.layer_stack:
                    dec_output, _, _ = dec_layer(
                        dec_output, encoder_outputs,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask,
                        dec_enc_attn_mask=None)

                seq_logit = self.tgt_word_prj(dec_output[:, -1])
                # local_scores = F.log_softmax(seq_logit, dim=1)
                local_scores = F.log_softmax(seq_logit, dim=1)
                # print('local_scores.size(): ' + str(local_scores.size()))
                # local_scores += freq
                # print('local_scores: ' + str(local_scores))

                # topk scores
                local_best_scores, local_best_ids = torch.topk(
                    local_scores, beam, dim=1)

                for j in range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = torch.ones(1, (1 + ys.size(1))).type_as(encoder_outputs).long()
                    new_hyp['yseq'][:, :ys.size(1)] = hyp['yseq']
                    new_hyp['yseq'][:, ys.size(1)] = int(local_best_ids[0, j])
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(hyps_best_kept,
                                        key=lambda x: x['score'],
                                        reverse=True)[:beam]
            # end for hyp in hyps
            hyps = hyps_best_kept

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                for hyp in hyps:
                    hyp['yseq'] = torch.cat([hyp['yseq'],
                                             torch.ones(1, 1).fill_(self.eos_id).type_as(encoder_outputs).long()],
                                            dim=1)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][0, -1] == self.eos_id:
                    ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            hyps = remained_hyps
            # if len(hyps) > 0:
            #     print('remeined hypothes: ' + str(len(hyps)))
            # else:
            #     print('no hypothesis. Finish decoding.')
            #     break
            #
            # for hyp in hyps:
            #     print('hypo: ' + ''.join([char_list[int(x)]
            #                               for x in hyp['yseq'][0, 1:]]))
        # end for i in range(maxlen)
        nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[
                     :min(len(ended_hyps), nbest)]
        # compitable with LAS implementation
        for hyp in nbest_hyps:
            hyp['yseq'] = hyp['yseq'][0].cpu().numpy().tolist()
        return nbest_hyps


class DecoderLayer(nn.Module):
    """
        Compose with three layers
    """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn


class Encoder(nn.Module):
    """
        Encoder of Transformer including self-attention and feed forward.
    """

    def __init__(self, d_input=320, n_layers=6, n_head=8, d_k=64, d_v=64, d_model=512, d_inner=2048, dropout=0.1,
                 pe_maxlen=5000):
        super(Encoder, self).__init__()
        # parameters
        self.d_input = d_input
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout_rate = dropout
        self.pe_maxlen = pe_maxlen

        # use linear transformation with layer norm to replace input embedding
        self.linear_in = nn.Linear(d_input, d_model)
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, padded_input, input_lengths, return_attns=False):
        """
        Args:
            padded_input: N x T x D
            input_lengths: N
        Returns:
            enc_output: N x T x H
        """
        enc_slf_attn_list = []

        # Prepare masks
        non_pad_mask = Util.get_non_pad_mask(padded_input, input_lengths=input_lengths)
        length = padded_input.size(1)
        slf_attn_mask = Util.get_attn_pad_mask(padded_input, input_lengths, length)

        # Forward
        enc_output = self.dropout(
            self.layer_norm_in(self.linear_in(padded_input)) +
            self.positional_encoding(padded_input))

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class EncoderLayer(nn.Module):
    """
        Compose with two sub-layers.
        1. A multi-head self-attention mechanism
        2. A simple, position-wise fully connected feed-forward network.
    """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class PositionalEncoding(nn.Module):
    """
        Implement the positional encoding (PE) function.
        PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
        PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input):
        """
        Args:
            input: N x T x D
        """
        length = input.size(1)
        return self.pe[:, :length]


class PositionwiseFeedForward(nn.Module):
    """
        Implements position-wise feedforward sublayer.
        FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class PositionwiseFeedForwardUseConv(nn.Module):
    """
        # Another implementation
        A two-feed-forward-layer module
    """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForwardUseConv, self).__init__()
        # position-wise
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class Transformer(nn.Module):
    """
        An encoder-decoder framework only includes attention.
    """

    def __init__(self, encoder=None, decoder=None):
        super(Transformer, self).__init__()

        if encoder is not None and decoder is not None:
            self.encoder = encoder
            self.decoder = decoder

            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            self.encoder = Encoder()
            self.decoder = Decoder()

    def forward(self, padded_input, input_lengths, padded_target):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        encoder_padded_outputs, *_ = self.encoder(padded_input, input_lengths)
        # pred is score before softmax
        pred, gold, *_ = self.decoder(padded_target, encoder_padded_outputs,
                                      input_lengths)
        return pred, gold

    def recognize(self, input, input_length, char_list, args):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam
        Returns:
            nbest_hyps:
        """
        encoder_outputs, *_ = self.encoder(input.unsqueeze(0), input_length)
        nbest_hyps = self.decoder.recognize_beam(encoder_outputs[0],
                                                 char_list,
                                                 args)
        return nbest_hyps
