# -*- coding=utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

import torch
import copy
import pandas as pd
import torchaudio as ta
import numpy as np
from torch.utils.data import Dataset
from Seq2seq_Transformer import args


class Util(object):
    @staticmethod
    # 定义优化器以及学习率更新函数
    def get_learning_rate(step):
        return args.lr_factor * args.model_size ** (-0.5) * min(step ** (-0.5), step * args.warmup_steps ** (-1.5))

    @staticmethod
    def collate_fn(batch):
        """
        收集函数，将同一批内的特征填充到相同的长度，并在文本中加上起始和结束标记
        :param batch:
        :return:
        """
        if len(batch) == 1:
            features_length = [data.shape[0] for data in batch]
        else:
            features_length = [data[0].shape[0] for data in batch]

        max_feat_length = max(features_length)
        padded_features = []

        if len(batch[0]) == 2:
            targets_length = [len(data[1]) for data in batch]
            max_text_length = max(targets_length)
            padded_targets = []

        for parts in batch:
            if len(batch) == 1:
                feat = parts
            else:
                feat = parts[0]

            feat_len = feat.shape[0]
            padded_features.append(
                np.pad(feat, ((0, max_feat_length - feat_len), (0, 0)), mode='constant', constant_values=0.0))

            if len(batch[0]) == 2:
                target = parts[1]
                text_len = len(target)
                padded_targets.append([args.vocab['<BOS>']] + target + [args.vocab['<EOS>']] + [args.vocab['<PAD>']] * (
                        max_text_length - text_len))

        if len(batch[0]) == 2:
            return torch.FloatTensor(padded_features), torch.LongTensor(padded_targets)
        else:
            return torch.FloatTensor(padded_features)

    @staticmethod
    def get_seq_mask(targets):
        """
        遮掉未来的文本信息
        :param targets:
        :return:
        """
        batch_size, steps = targets.size()
        seq_mask = torch.ones([batch_size, steps, steps], device=targets.device)
        seq_mask = torch.tril(seq_mask).bool()

        return seq_mask


class DataUtil(object):
    def __init__(self):
        dev_csv_path = args.dev_csv_path
        vocab_txt_path = args.vocab_txt_path

        DataUtil.generate_vocab_table(dev_csv_path=dev_csv_path, vocab_txt_path=vocab_txt_path)

    # 词表生成
    @staticmethod
    def generate_vocab_table(dev_csv_path, vocab_txt_path):
        """
        根据训练集文本生成词表，并加入起始标记<BOS>,结束标记<EOS>,填充标记<PAD>,以及未识别词标记<UNK>

        :return: 返回模型词表大小
        """
        vocab_dict = {}

        dev_data = pd.read_csv(filepath_or_buffer=dev_csv_path, encoding='utf-8')

        for text in dev_data['label']:
            for char in text:
                if char == ' ':
                    char = '#'
                if char not in vocab_dict:
                    vocab_dict[char] = 1
                else:
                    vocab_dict[char] += 1

        vocab_list = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
        vocab = copy.deepcopy(args.vocab)
        for index, item in enumerate(vocab_list):
            vocab[item[0]] = index + 4

        print('There are {} units in Vocabulary!'.format(len(vocab)))

        with open(vocab_txt_path, mode='w', encoding='utf-8') as file:
            for key, value in vocab.items():
                file.write(key + ' ' + str(value) + '\n')

        return len(vocab)


class AudioDataset(Dataset):
    def __init__(self, audios_list, labels_list=None, unit2idx=None):
        self.audios_list = audios_list
        self.unit2idx = unit2idx

        if labels_list is not None:
            self.targets_list = []
            for line in labels_list:
                label = []
                for c in line:
                    if c == ' ':
                        label.append(self.unit2idx['#'])
                    elif c in self.unit2idx:
                        label.append(self.unit2idx[c])
                    else:
                        label.append(self.unit2idx['<UNK>'])
                self.targets_list.append(label)
        else:
            self.targets_list = None

        self.lengths = len(self.audios_list)

    def __getitem__(self, index):
        path = os.path.join(args.input_dir, self.audios_list[index])
        # 加载wav文件
        wavform, _ = ta.load_wav(path)
        feature = ta.compliance.kaldi.fbank(wavform, num_mel_bins=args.input_size)
        # 特征归一化
        mean = torch.mean(feature)
        std = torch.std(feature)
        feature = (feature - mean) / std

        if self.targets_list is not None:
            targets = self.targets_list[index]
            return feature, targets
        else:
            return feature

    def __len__(self):
        return self.lengths

    @property
    def idx2char(self):
        return {i: c for (c, i) in self.unit2idx.items()}


if __name__ == '__main__':
    DataUtil()
