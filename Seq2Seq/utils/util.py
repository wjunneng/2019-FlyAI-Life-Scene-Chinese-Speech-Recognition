# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

import torch
import pickle
import librosa
import random
import time
import numpy as np
import torch.nn.functional as F
import soundfile as sf
from collections import OrderedDict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from python_speech_features import logfbank
from sklearn import preprocessing

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

        before = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
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

    @staticmethod
    def compute_fbank_from_file(file, feature_dim=80):
        signal, sample_rate = sf.read(file)
        feature = Util.compute_fbank_from_api(signal, sample_rate, nfilt=feature_dim)

        return feature

    @staticmethod
    def compute_fbank_from_api(signal, sample_rate, nfilt):
        """
        Fbank特征提取, 结果进行零均值归一化操作
        :param wav_file: 文件路径
        :return: feature向量
        """
        feature = logfbank(signal, sample_rate, nfilt=nfilt, nfft=2048)
        feature = preprocessing.scale(feature)
        return feature

    @staticmethod
    def han2id(line, vocab, PAD_FLAG, PAD, SOS_FLAG, SOS, EOS_FLAG, EOS):
        """
        文字转向量 one-hot embedding，没有成功在vocab中找到索引抛出异常，交给上层处理
        :param line:
        :param vocab:
        :return:
        """
        try:
            res = list([])
            for han in line:
                if han == PAD_FLAG:
                    res.append(PAD)
                elif han == SOS_FLAG:
                    res.append(SOS)
                elif han == EOS_FLAG:
                    res.append(EOS)
                else:
                    res.append(vocab[han])
            return res
        except ValueError:
            raise ValueError

    @staticmethod
    def wav_padding(wav_data_lst):
        feature_dim = wav_data_lst[0].shape[1]
        # len(data)实际上就是求语谱图的第一维的长度，也就是n_frames
        wav_lens = np.array([len(data) for data in wav_data_lst])
        # 取一个batch中的最长
        wav_max_len = max(wav_lens)
        new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, feature_dim), dtype=np.float32)
        for i in range(len(wav_data_lst)):
            new_wav_data_lst[i, :wav_data_lst[i].shape[0], :] = wav_data_lst[i]
        return new_wav_data_lst, wav_lens

    @staticmethod
    def label_padding(label_data_lst, pad_idx):
        label_lens = np.array([len(label) for label in label_data_lst])
        max_label_len = max(label_lens)
        new_label_data_lst = np.zeros((len(label_data_lst), max_label_len), dtype=np.int32)
        new_label_data_lst += pad_idx
        for i in range(len(label_data_lst)):
            new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
        return new_label_data_lst, label_lens


class TransformerOptimizer(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, k, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.k = k
        self.init_lr = d_model ** (-0.5)
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.visdom_lr = None

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._update_lr()
        self._visdom()
        self.optimizer.step()

    def _update_lr(self):
        self.step_num += 1
        lr = self.k * self.init_lr * min(self.step_num ** (-0.5),
                                         self.step_num * (self.warmup_steps ** (-1.5)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def set_k(self, k):
        self.k = k

    def set_visdom(self, visdom_lr, vis):
        self.visdom_lr = visdom_lr  # Turn on/off visdom of learning rate
        self.vis = vis  # visdom enviroment
        self.vis_opts = dict(title='Learning Rate',
                             ylabel='Leanring Rate', xlabel='step')
        self.vis_window = None
        self.x_axis = torch.LongTensor()
        self.y_axis = torch.FloatTensor()

    def _visdom(self):
        if self.visdom_lr is not None:
            self.x_axis = torch.cat(
                [self.x_axis, torch.LongTensor([self.step_num])])
            self.y_axis = torch.cat(
                [self.y_axis, torch.FloatTensor([self.optimizer.param_groups[0]['lr']])])
            if self.vis_window is None:
                self.vis_window = self.vis.line(X=self.x_axis, Y=self.y_axis,
                                                opts=self.vis_opts)
            else:
                self.vis.line(X=self.x_axis, Y=self.y_axis, win=self.vis_window,
                              update='replace')


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


class AudioDataset(Dataset):
    def __init__(self, args, samples):
        super(AudioDataset, self).__init__()
        self.args = args
        self.path_count = len(samples)
        self.BATCH = args.BATCH

        # 随机选取BATCH个wav数据组成一个batch_wav_data
        batch_nums = self.path_count // self.BATCH
        rest = self.path_count % self.BATCH
        index_list = list(range(0, self.path_count))
        batch_list = []

        # 多加一个表示最后的一个个数不足的batch
        if rest != 0:
            batch_nums += 1
        for i in range(batch_nums):
            begin = i * self.args.BATCH
            end = min(self.path_count, begin + self.args.BATCH)
            batch_index = index_list[begin: end]
            dict = {'batch_list': batch_index}
            batch_list.append(dict)

        self.minibatch = batch_list
        print('True')

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class AudioDataLoader(DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, feature_dim, char_list, path_list, label_list, arguments, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = LFRCollate(feature_dim=feature_dim, char_list=char_list, path_list=path_list,
                                     label_list=label_list, args=arguments)


class LFRCollate(object):
    """Build this wrapper to pass arguments(LFR_m, LFR_n) to _collate_fn"""

    def __init__(self, feature_dim, char_list, path_list, label_list, args):
        self.path_list = path_list
        self.label_list = label_list
        self.feature_dim = feature_dim
        self.char_list = char_list
        self.args = args

    def __call__(self, batch):
        return LFRCollate._collate_fn(batch=batch, feature_dim=self.feature_dim, char_list=self.char_list, path_list=self.path_list,
                                      label_list=self.label_list, args=self.args)

    @staticmethod
    def _collate_fn(batch, feature_dim, char_list, path_list, label_list, args):
        sub_list = batch[0]['batch_list']
        label_lst, input_lst, error_count = list([]), list([]), list([])
        random.shuffle(sub_list)
        for i in sub_list:
            try:
                # get_fbank_and_hanzi_data(i, feature_dim, char_list, path_list, label_list)
                feature = Util.compute_fbank_from_file(file=os.path.join(args.wav_dir, path_list[i]),
                                                       feature_dim=feature_dim)
                label = Util.han2id(line=label_list[i], vocab=char_list, PAD_FLAG=args.PAD_FLAG, PAD=args.PAD,
                                    SOS_FLAG=args.SOS_FLAG, SOS=args.SOS, EOS_FLAG=args.EOS_FLAG, EOS=args.EOS)
                # 长度大于1600帧，过长，跳过
                if len(feature) > 1600:
                    continue

                input_data = Util.build_LFR_features(inputs=feature, m=args.LFR_m, n=args.LFR_n)
                label_lst.append(label)
                input_lst.append(input_data)
            except ValueError:
                error_count.append(i)
                continue

        # 删除异常语音信息
        if error_count != list([]):
            input_lst = np.delete(input_lst, error_count, axis=0)
            label_lst = np.delete(label_lst, error_count, axis=0)
        pad_wav_data, pad_lengths = Util.wav_padding(input_lst)
        pad_target_data, _ = Util.label_padding(label_lst, args.IGNORE_ID)
        padded_input = torch.from_numpy(pad_wav_data).float()
        input_lengths = torch.from_numpy(pad_lengths)
        padded_target = torch.from_numpy(pad_target_data).long()

        return padded_input, padded_target, input_lengths


class Solver(object):
    """
    训练+验证
    """

    def __init__(self, tr_loader, cv_loader, model, optimizer, args):
        self.tr_loader = tr_loader
        self.cv_loader = cv_loader

        self.model = model
        self.optimizer = optimizer

        # Low frame rate feature
        self.LFR_m = args.LFR_m
        self.LFR_n = args.LFR_n

        # Training config
        self.EPOCHS = args.EPOCHS
        self.label_smoothing = args.label_smoothing
        # save and load model
        self.output_dir = args.output_dir
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.model_path = args.model_path
        # logging
        self.print_freq = args.print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.EPOCHS)
        self.cv_loss = torch.Tensor(self.EPOCHS)
        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.continue_from)
            self.model.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.output_dir, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False

    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.EPOCHS):
            # Train one epoch
            print("Training...")
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | Train Loss {2:.3f}'.format(epoch + 1,
                                                                                                 time.time() - start,
                                                                                                 tr_avg_loss))
            print('-' * 85)

            # Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            val_loss = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | Valid Loss {2:.3f}'.format(epoch + 1,
                                                                                                 time.time() - start,
                                                                                                 val_loss))
            print('-' * 85)

            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(self.output_dir, 'epoch%d_%.3f.pth.tar' % (epoch + 1, val_loss))
                torch.save(self.model.serialize(self.model, self.optimizer, epoch + 1, self.LFR_m, self.LFR_n,
                                                tr_loss=self.tr_loss, cv_loss=self.cv_loss), file_path)
                print('Saving checkpoint model to %s' % file_path)

            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.output_dir, self.model_path)
                torch.save(self.model.serialize(self.model, self.optimizer, epoch + 1, self.LFR_m, self.LFR_n,
                                                tr_loss=self.tr_loss, cv_loss=self.cv_loss), file_path)
                print("Find better validated model, saving to %s" % file_path)

    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0

        loader = self.tr_loader if not cross_valid else self.cv_loader
        # batch_nums = self.train_batch_nums if not cross_valid else self.verify_batch_nums

        for i, data in enumerate(loader):
            padded_input, padded_target, input_lengths = data
            padded_input = padded_input.cuda()
            input_lengths = input_lengths.cuda()
            padded_target = padded_target.cuda()

            pred, gold = self.model(padded_input, input_lengths, padded_target)
            loss, n_correct = Util.cal_performance(pred, gold, smoothing=self.label_smoothing)
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            non_pad_mask = gold.ne(IGNORE_ID)
            n_word = non_pad_mask.sum().item()

            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                    epoch + 1, i + 1, total_loss / (i + 1),
                    loss.item(), 1000 * (time.time() - start) / (i + 1)),
                    flush=True)

        return total_loss / (i + 1)
