# -*- coding:utf-8 -*-
from collections import OrderedDict
import torch
import torch.nn.functional as F

from Seq2Seq.args import IGNORE_ID


class SortedByCountsDict(object):
    """
    构建具备自动排序的字典类
    """

    def __init__(self):
        self.vocab = OrderedDict()
        self.i_vocab = OrderedDict()

    def append_token(self, token: str):
        if token not in self.vocab:
            self.vocab[token] = 1
        else:
            self.vocab[token] += 1

    def append_tokens(self, tokens: list):
        for token in tokens:
            self.append_token(token)

    def get_vocab(self):
        self.vocab = OrderedDict(sorted(self.vocab.items(), key=lambda item: item[1], reverse=True))

        return self.vocab

    def get_i_vocab(self):
        self.i_vocab = dict(zip((value, key) for (key, value) in self.vocab.items()))

        return self.i_vocab


class Util(object):
    """
    工具类
    """

    @staticmethod
    def pad_list(xs, pad_value):
        # From: espnet/src/nets/e2e_asr_th.py: pad_list()
        n_batch = len(xs)
        max_len = max(x.size(0) for x in xs)
        pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
        for i in range(n_batch):
            pad[i, :xs[i].size(0)] = xs[i]

        return pad

    @staticmethod
    def process_dict(dict_path):
        with open(dict_path, 'rb') as f:
            dictionary = f.readlines()
        char_list = [entry.decode('utf-8').split(' ')[0]
                     for entry in dictionary]
        sos_id = char_list.index('<sos>')
        eos_id = char_list.index('<eos>')

        return char_list, sos_id, eos_id

    # * ------------------ recognition related ------------------ *
    @staticmethod
    def parse_hypothesis(hyp, char_list):
        """Function to parse hypothesis
        :param list hyp: recognition hypothesis
        :param list char_list: list of characters
        :return: recognition text strinig
        :return: recognition token strinig
        :return: recognition tokenid string
        """
        # remove sos and get results
        tokenid_as_list = list(map(int, hyp['yseq'][1:]))
        token_as_list = [char_list[idx] for idx in tokenid_as_list]
        score = float(hyp['score'])

        # convert to string
        tokenid = " ".join([str(idx) for idx in tokenid_as_list])
        token = " ".join(token_as_list)
        text = "".join(token_as_list).replace('<space>', ' ')

        return text, token, tokenid, score

    @staticmethod
    def add_results_to_json(js, nbest_hyps, char_list):
        """Function to add N-best results to json
        :param dict js: groundtruth utterance dict
        :param list nbest_hyps: list of hypothesis
        :param list char_list: list of characters
        :return: N-best results added utterance dict
        """
        # copy old json info
        new_js = dict()
        new_js['utt2spk'] = js['utt2spk']
        new_js['output'] = []

        for n, hyp in enumerate(nbest_hyps, 1):
            # parse hypothesis
            rec_text, rec_token, rec_tokenid, score = Util.parse_hypothesis(
                hyp, char_list)

            # copy ground-truth
            out_dic = dict(js['output'][0].items())

            # update name
            out_dic['name'] += '[%d]' % n

            # add recognition results
            out_dic['rec_text'] = rec_text
            out_dic['rec_token'] = rec_token
            out_dic['rec_tokenid'] = rec_tokenid
            out_dic['score'] = score

            # add to list of N-best result dicts
            new_js['output'].append(out_dic)

            # show 1-best result
            if n == 1:
                print('groundtruth: %s' % out_dic['text'])
                print('prediction : %s' % out_dic['rec_text'])

        return new_js

    # -- Transformer Related --
    @staticmethod
    def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
        """
        padding position is set to 0, either use input_lengths or pad_idx
        """
        assert input_lengths is not None or pad_idx is not None
        non_pad_mask = None
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
    def get_subsequent_mask(seq):
        """
        For masking out the subsequent info.
        """

        sz_b, len_s = seq.size()
        subsequent_mask = torch.triu(
            torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

        return subsequent_mask

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

    # -- Transformer Loss --
    @staticmethod
    def cal_performance(pred, gold, smoothing=0.0):
        """Calculate cross entropy loss, apply label smoothing if needed.
        Args:
            pred: N x T x C, score before softmax
            gold: N x T
        """

        pred = pred.view(-1, pred.size(2))
        gold = gold.contiguous().view(-1)

        loss = Util.cal_loss(pred, gold, smoothing)

        pred = pred.max(1)[1]
        non_pad_mask = gold.ne(IGNORE_ID)
        n_correct = pred.eq(gold)
        n_correct = n_correct.masked_select(non_pad_mask).sum().item()

        return loss, n_correct

    @staticmethod
    def cal_loss(pred, gold, smoothing=0.0):
        """Calculate cross entropy loss, apply label smoothing if needed.
        """

        if smoothing > 0.0:
            eps = smoothing
            n_class = pred.size(1)

            # Generate one-hot matrix: N x C.
            # Only label position is 1 and all other positions are 0
            # gold include -1 value (IGNORE_ID) and this will lead to assert error
            gold_for_scatter = gold.ne(IGNORE_ID).long() * gold
            one_hot = torch.zeros_like(pred).scatter(1, gold_for_scatter.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / n_class
            log_prb = F.log_softmax(pred, dim=1)

            non_pad_mask = gold.ne(IGNORE_ID)
            n_word = non_pad_mask.sum().item()
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum() / n_word
        else:
            loss = F.cross_entropy(pred, gold, ignore_index=IGNORE_ID, reduction='elementwise_mean')

        return loss


class TransformerOptimizer(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, warmup_steps=4000, k=0.2):
        self.optimizer = optimizer
        self.k = k
        self.warmup_steps = warmup_steps
        self.init_lr = 512 ** (-0.5)
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
