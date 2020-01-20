# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

import torch
import pickle
import librosa
import random
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from Seq2Seq.args import IGNORE_ID


class SortedByCountsDict(object):
    """
    构建具备自动排序的字典类
    """

    def __init__(self, dump_dir, type='predict'):
        # dump dir
        self.dump_dir = dump_dir
        # 字：次数
        self.s_vocab = OrderedDict()
        # 字：索引
        self.vocab = {}
        # 索引：字
        self.i_vocab = {}

        if os.path.exists(dump_dir) and type == 'predict':
            self.vocab = SortedByCountsDict.load_pkl(load_dir=self.dump_dir)

    def append_token(self, token: str):
        if token not in self.s_vocab:
            self.s_vocab[token] = 1
        else:
            self.s_vocab[token] += 1

    def append_tokens(self, tokens: list):
        for token in tokens:
            self.append_token(token)

    def get_vocab(self):
        if len(self.vocab) != 0:
            return self.vocab

        before = {'<sos>': 0, '<eos>': 1}
        after = dict(sorted(self.s_vocab.items(), key=lambda item: item[1], reverse=True))
        after = dict(zip(after.keys(), [i + 2 for i in range(len(after))]))

        self.vocab.update(before)
        self.vocab.update(after)

        return self.vocab

    def get_i_vocab(self):
        self.vocab = self.get_vocab()
        self.i_vocab = {value: key for (key, value) in self.vocab.items()}

        return self.i_vocab

    def dump_pkl(self):
        self.vocab = self.get_vocab()
        with open(self.dump_dir, mode='wb') as file:
            pickle.dump(file=file, obj=self.vocab)

    @staticmethod
    def load_pkl(load_dir):
        with open(load_dir, mode='rb') as file:
            vocab = pickle.load(file=file)

        return vocab


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

    @staticmethod
    def normalize(yt):
        yt_max = np.max(yt)
        yt_min = np.min(yt)
        a = 1.0 / (yt_max - yt_min)
        b = -(yt_max + yt_min) / (2 * (yt_max - yt_min))

        yt = yt * a + b
        return yt

    @staticmethod
    def extract_feature(input_file, feature='fbank', dim=80, cmvn=True, delta=False, delta_delta=False,
                        window_size=25, stride=10, save_feature=None):
        y, sr = librosa.load(input_file, sr=16000)
        yt, _ = librosa.effects.trim(y, top_db=20)
        yt = Util.normalize(yt)
        ws = int(sr * 0.001 * window_size)
        st = int(sr * 0.001 * stride)
        if feature == 'fbank':  # log-scaled
            feat = librosa.feature.melspectrogram(y=yt, sr=sr, n_mels=dim, n_fft=ws, hop_length=st)
            feat = np.log(feat + 1e-6)
        elif feature == 'mfcc':
            feat = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=dim, n_mels=dim, n_fft=ws, hop_length=st)
            feat[0] = librosa.feature.rms(yt, hop_length=st, frame_length=ws)

        else:
            raise ValueError('Unsupported Acoustic Feature: ' + feature)

        feat = [feat]
        if delta:
            feat.append(librosa.feature.delta(feat[0]))

        if delta_delta:
            feat.append(librosa.feature.delta(feat[0], order=2))
        feat = np.concatenate(feat, axis=0)
        if cmvn:
            feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]
        if save_feature is not None:
            tmp = np.swapaxes(feat, 0, 1).astype('float32')
            np.save(save_feature, tmp)
            return len(tmp)
        else:
            return np.swapaxes(feat, 0, 1).astype('float32')

    @staticmethod
    def spec_augment(spec: np.ndarray, num_mask=2, freq_masking=0.15, time_masking=0.20, value=0):
        spec = spec.copy()
        num_mask = random.randint(1, num_mask)
        for i in range(num_mask):
            all_freqs_num, all_frames_num = spec.shape
            freq_percentage = random.uniform(0.0, freq_masking)

            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[f0:f0 + num_freqs_to_mask, :] = value

            time_percentage = random.uniform(0.0, time_masking)

            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[:, t0:t0 + num_frames_to_mask] = value

        return spec

    @staticmethod
    def build_LFR_features(inputs, m, n):
        """
        Actually, this implements stacking frames and skipping frames.
        if m = 1 and n = 1, just return the origin features.
        if m = 1 and n > 1, it works like skipping.
        if m > 1 and n = 1, it works like stacking but only support right frames.
        if m > 1 and n > 1, it works like LFR.
        Args:
            inputs_batch: inputs is T x D np.ndarray
            m: number of frames to stack
            n: number of frames to skip
        """
        # LFR_inputs_batch = []
        # for inputs in inputs_batch:
        LFR_inputs = []
        T = inputs.shape[0]
        T_lfr = int(np.ceil(T / n))
        for i in range(T_lfr):
            if m <= T - i * n:
                LFR_inputs.append(np.hstack(inputs[i * n:i * n + m]))
            else:  # process last LFR frame
                num_padding = m - (T - i * n)
                frame = np.hstack(inputs[i * n:])
                for _ in range(num_padding):
                    frame = np.hstack((frame, inputs[-1]))
                LFR_inputs.append(frame)

        return np.vstack(LFR_inputs)

    @staticmethod
    def pad_collate(batch):
        max_input_len = float('-inf')
        max_target_len = float('-inf')

        for elem in batch:
            feature, trn = elem
            max_input_len = max_input_len if max_input_len > feature.shape[0] else feature.shape[0]
            max_target_len = max_target_len if max_target_len > len(trn) else len(trn)

        for i, elem in enumerate(batch):
            feature, trn = elem
            input_length = feature.shape[0]
            input_dim = feature.shape[1]
            padded_input = np.zeros((max_input_len, input_dim), dtype=np.float32)
            padded_input[:input_length, :] = feature
            padded_target = np.pad(trn, (0, max_target_len - len(trn)), 'constant', constant_values=IGNORE_ID)
            batch[i] = (padded_input, padded_target, input_length)

        # sort it by input lengths (long to short)
        batch.sort(key=lambda x: x[2], reverse=True)

        return default_collate(batch)

    @staticmethod
    def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, is_best, output_dir):
        state = {'epoch': epoch,
                 'epochs_since_improvement': epochs_since_improvement,
                 'loss': loss,
                 'model': model,
                 'optimizer': optimizer}

        torch.save(state, os.path.join(output_dir, 'checkpoint.tar'))
        # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
        if is_best:
            torch.save(state, os.path.join(output_dir, 'BEST_checkpoint.tar'))

    @staticmethod
    def load_checkpoint(output_dir):
        state = torch.load(f=output_dir)

        return state['model']


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


class AiShellDataset(Dataset):
    def __init__(self, args, samples, vocab):
        self.args = args
        self.samples = samples
        self.vocab = vocab

    def __getitem__(self, i):
        sample = self.samples[i]
        wave = os.path.join(self.args.wav_dir, sample['wav'])

        feature = Util.extract_feature(input_file=wave, feature=self.args.feature_type, dim=self.args.d_input,
                                       cmvn=True)
        # zero mean and unit variance
        feature = (feature - feature.mean()) / feature.std()
        feature = Util.spec_augment(feature)
        feature = Util.build_LFR_features(feature, m=self.args.LFR_m, n=self.args.LFR_n)

        if 'trn' in sample.keys():
            trn = [self.vocab[i] if i in self.vocab.keys() else -1 for i in sample['trn']]
            return feature, trn

        return feature

    def __len__(self):
        return len(self.samples)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
