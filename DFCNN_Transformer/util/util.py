# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict
import tensorflow as tf
from keras import backend

from DFCNN_Transformer import args


class SortedByCountsDict(object):
    """
    构建具备自动排序的字典类
    """

    def __init__(self, dump_dir):
        # dump dir
        self.dump_dir = dump_dir
        # 字：次数
        self.s_vocab = OrderedDict()
        # 字：索引
        self.vocab = {}
        # 索引：字
        self.i_vocab = {}

        if os.path.exists(dump_dir):
            self.vocab = self.load_pkl(load_dir=self.dump_dir)

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
        with open(self.dump_dir, mode='wb') as file:
            pickle.dump(file=file, obj=self.vocab)

    def load_pkl(self, load_dir):
        with open(load_dir, mode='rb') as file:
            self.vocab = pickle.load(file=file)

        return self.vocab


class Util(object):
    def __init__(self):
        pass

    @staticmethod
    def get_acoustic_vocab_list():
        text = pd.read_table(args.mixdict_dir, header=None)
        symbol_list = text.iloc[:, 0].tolist()
        symbol_list.append('_')

        return len(symbol_list), symbol_list

    @staticmethod
    def decode_ctc(num_result, input_length):
        """
        定义解码器
        :param num_result:
        :param input_length:
        :return:
        """

        result = num_result[:, :, :]
        in_len = np.zeros(1, dtype=np.int32)
        in_len[0] = input_length
        r = backend.ctc_decode(result, in_len, greedy=True, beam_width=100, top_paths=1)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        r1 = r[0][0].eval(session=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
        tf.reset_default_graph()  # 然后重置tf图，这句很关键
        r1 = r1[0]

        return r1
