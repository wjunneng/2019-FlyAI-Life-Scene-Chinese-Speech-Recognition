# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

import pickle
import math
import numpy as np
import pandas as pd
import soundfile as sf
from collections import OrderedDict
import tensorflow as tf
from sklearn import preprocessing
from python_speech_features import logfbank

from keras import backend
from keras.utils import Sequence

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


class DataGenerator(Sequence):
    def __init__(self, audio_paths, labels, pinyins, hp, acoustic_vocab):
        self.paths = audio_paths
        self.hz_labels = labels
        self.py_labels = pinyins
        self.type = hp.data_type
        self.shuffle = hp.shuffle
        self.data_path = hp.data_path
        self.batch_size = hp.batch_size
        self.feature_max_length = hp.am_feature_max_length
        self.indexes = np.arange(len(self.paths))
        self.acoustic_vocab = acoustic_vocab

    def __getitem__(self, index):
        # 生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_datas = [self.paths[k] for k in batch_indexs]
        py_label_datas = [self.py_labels[k] for k in batch_indexs]
        # 生成数据
        X, y = self.data_generation(batch_datas, py_label_datas)
        return X, y

    def __len__(self):
        # 计算每一个epoch的迭代次数
        return math.ceil(len(self.paths) / float(self.batch_size))

    def data_generation(self, batch_datas, py_label_datas):
        # batch_wav_data.shape = (10 1600 200 1), inputs_length.shape = (10,)
        batch_wav_data = np.zeros((self.batch_size, self.feature_max_length, 200, 1), dtype=np.float)
        # batch_label_data.shape = (10 64) ,label_length.shape = (10,)
        batch_label_data = np.zeros((self.batch_size, 64), dtype=np.int64)
        # length
        input_length = []
        label_length = []
        error_count = []
        # 随机选取batch_size个wav数据组成一个batch_wav_data
        for i, path in enumerate(batch_datas):
            # Fbank特征提取函数(从feature_python)
            try:
                file1 = os.path.join(self.data_path, path)
                if os.path.isfile(file1):
                    signal, sample_rate = sf.read(file1)
                else:
                    print("file path Error")
                    return 0
                fbank = Util.compute_fbank_from_api(signal, sample_rate)
                input_data = fbank.reshape([fbank.shape[0], fbank.shape[1], 1])
                data_length = input_data.shape[0] // 8 + 1
                label = Util.pny2id(py_label_datas[i], self.acoustic_vocab)
                label = np.array(label)
                len_label = len(label)
                # 将错误数据进行抛出异常,并处理
                if input_data.shape[0] > self.feature_max_length:
                    raise ValueError
                if len_label > 64 or len_label > data_length:
                    raise ValueError

                input_length.append([data_length])
                label_length.append([len_label])
                batch_wav_data[i, 0:len(input_data)] = input_data
                batch_label_data[i, 0:len_label] = label
            except ValueError:
                error_count.append(i)
                continue
        # 删除异常语音信息
        if error_count != []:
            batch_wav_data = np.delete(batch_wav_data, error_count, axis=0)
            batch_label_data = np.delete(batch_label_data, error_count, axis=0)
        label_length = np.mat(label_length)
        input_length = np.mat(input_length)
        # CTC 输入长度0-1600//8+1
        # label label真实长度
        inputs = {'the_inputs': batch_wav_data,
                  'the_labels': batch_label_data,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        outputs = {'ctc': np.zeros((self.batch_size - len(error_count), 1), dtype=np.float32)}

        return inputs, outputs


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

    @staticmethod
    def pny2id(line, vocab):
        """
        拼音转向量 one-hot embedding，没有成功在vocab中找到索引抛出异常，交给上层处理
        :param line:
        :param vocab:
        :return:
        """
        try:
            line.strip().split(' ')

            return [vocab.index(pin) for pin in line]
        except ValueError:
            raise ValueError

    @staticmethod
    def compute_fbank_from_api(signal, sample_rate, nfilt=200):
        """
        Fbank特征提取, 结果进行零均值归一化操作
        :param wav_file: 文件路径
        :return: feature向量
        """
        feature = logfbank(signal, sample_rate, nfilt=nfilt)
        feature = preprocessing.scale(feature)

        return feature
