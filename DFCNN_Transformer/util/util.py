# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

import difflib
import pickle
import math
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from random import shuffle
from collections import OrderedDict
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
        self.feature_max_length = hp.feature_max_length
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


class TransformerUtil(object):
    @staticmethod
    def layer_norm(inputs, epsilon=1e-8, scope="ln", reuse=None):
        """
        LayerNorm
        :param inputs:
        :param epsilon:
        :param scope:
        :param reuse:
        :return:
        """
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            # 取最后一个维度
            params_shape = inputs_shape[-1:]
            # 计算均值和方差
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
            outputs = gamma * normalized + beta

        return outputs

    @staticmethod
    def embedding(inputs, vocab_size, num_units, zero_pad=True, scale=True, scope="embedding", reuse=None):
        """
        向量的embedding
        :param inputs:
        :param vocab_size: 语料库大小
        :param num_units:
        :param zero_pad: 是否补0
        :param scale:
        :param scope:
        :param reuse:
        :return:
        """
        with tf.variable_scope(scope, reuse=reuse):
            lookup_table = tf.get_variable('lookup_table',
                                           dtype=tf.float32,
                                           shape=[vocab_size, num_units],
                                           initializer=tf.contrib.layers.xavier_initializer())
            if zero_pad:
                lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)
            outputs = tf.nn.embedding_lookup(lookup_table, inputs)

            if scale:
                outputs = outputs * (num_units ** 0.5)

        return outputs

    @staticmethod
    def mask(inputs, queries=None, keys=None, type=None):
        if type in ("k", "key", "keys"):
            # 遮掩Key中为0的信息
            # Key Masking (-1,0,1), 当x<0,=0,>0时
            # 如果最后一个维度加起来为0，表示该长度上<MaxLength是没有拼音的。需要mask
            # 加起来不等于0，表示该长度上有拼音
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)
            paddings = tf.ones_like(inputs) * (-2 ** 32 + 1)
            # [80, 1164, 1164]内存爆炸
            outputs = tf.where(tf.equal(key_masks, 0), paddings, inputs)  # (h*N, T_q, T_k)

        elif type in ("q", "query", "querys"):
            # 遮掩Q中为0的信息
            # query中有信息的部分为1，没有信息的部分为0
            query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs = inputs * query_masks  # broadcasting. (h*N, T_q, T_k)

        elif type in ("f", "future", "right"):
            # 遮掩未来的信息, 实现一个三角矩阵，对未来的信息进行mask
            diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (h*N, T_q, T_k)
            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # (h*N, T_q, T_k)
        else:
            print("Check of upi entered type correctly !")

        return outputs

    @staticmethod
    def scaled_dot_product_attention(Q, K, V, causality=False, dropout_rate=0, is_training=True):
        # Multiplication Q乘以K的转置
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale 缩放降低维度 除以d(k)的平方根
        outputs = outputs / (K.get_shape().as_list()[-1] ** 0.5)

        outputs = TransformerUtil.mask(outputs, Q, K, type='key')

        # 遮掩未来的信息
        if causality:
            outputs = TransformerUtil.mask(outputs, Q, K, type="future")

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        # Query Masking
        outputs = TransformerUtil.mask(outputs, Q, K, type="query")
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum 加权平均
        outputs = tf.matmul(outputs, V)  # ( h*N, T_q, C/h)
        return outputs

    @staticmethod
    def multihead_attention(queries, keys, d_model=None, num_heads=8, dropout_rate=0,
                            is_training=True, causality=False, scope="multihead_attention", reuse=None):
        """
        多头注意力机制
        :param emb:
        :param queries: Q
        :param keys: K
        :param num_units: 层数
        :param num_heads: head个数, 通常为8个
        :param dropout_rate:
        :param is_training:
        :param causality:
        :param scope:
        :param reuse:
        :return:
        """
        with tf.variable_scope(scope, reuse=reuse):
            if d_model is None:
                d_model = queries.get_shape().as_list[-1]

            # Linear projections
            Q = tf.layers.dense(queries, d_model, activation=tf.nn.relu, use_bias=False)  # (N, T_q, C)
            K = tf.layers.dense(keys, d_model, activation=tf.nn.relu, use_bias=False)  # (N, T_k, C)
            V = tf.layers.dense(keys, d_model, activation=tf.nn.relu, use_bias=False)  # (N, T_k, C)

            # Split and concat 从第三个维度上划分成多头的QKV
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            outputs = TransformerUtil.scaled_dot_product_attention(Q_, K_, V_, causality, dropout_rate, is_training)

            # Restore shape 多头机制结合
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
            # Residual connection 参差链接，将query加起来。
            outputs += queries
            # Normalize layerNorm
            outputs = TransformerUtil.layer_norm(outputs)  # (N, T_q, C)

        return outputs

    @staticmethod
    def feedforward(inputs, num_units, scope="positionwise_ffnn", reuse=None):
        """Point-wise feed forward net.

        Args:
          inputs: A 3d tensor with shape of [N, T, C].
          num_units: A list of two integers.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer by the same name.

        Returns:
          A 3d tensor with the same shape and dtype as inputs
        """
        with tf.variable_scope(scope, reuse=reuse):
            # Inner layer
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1, "activation": tf.nn.relu,
                      "use_bias": True}
            # 1维卷积 构成了全连接
            outputs = tf.layers.conv1d(**params)

            # Readout layer
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1, "activation": None,
                      "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            # Residual connection 残差+LN
            outputs += inputs
            # Normalize
            outputs = TransformerUtil.layer_norm(outputs)

        return outputs

    @staticmethod
    def label_smoothing(inputs, epsilon=0.1):
        """
        平滑
        :param inputs:
        :param epsilon:
        :return:
        """
        K = inputs.get_shape().as_list()[-1]  # number of channels

        return ((1 - epsilon) * inputs) + (epsilon / K)


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
    def get_language_vocab_list():
        text = pd.read_table(args.hanzi_dir, header=None)
        list_lm = text.iloc[:, 0].tolist()
        list_lm.append('_')
        hanzi_num = len(list_lm)

        return hanzi_num, list_lm

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
            line = line.strip().split(' ')

            return [vocab.index(pin) for pin in line]
        except ValueError:
            raise ValueError

    @staticmethod
    def han2id(line, vocab, PAD_FLAG, SOS_FLAG, EOS_FLAG, PAD, SOS, EOS):
        """
        文字转向量 one-hot embedding，没有成功在vocab中找到索引抛出异常，交给上层处理
        :param line:
        :param vocab:
        :return:
        """
        try:
            line = line.strip()
            res = []
            for han in line:
                if han == PAD_FLAG:
                    res.append(PAD)
                elif han == SOS_FLAG:
                    res.append(SOS)
                elif han == EOS_FLAG:
                    res.append(EOS)
                else:
                    res.append(vocab.index(han))
            return res
        except ValueError:
            raise ValueError

    @staticmethod
    def compute_fbank_from_api(signal, sample_rate, nfilt=200):
        """
        Fbank特征提取, 结果进行零均值归一化操作
        :param wav_file: 文件路径
        :return: feature向量
        """
        feature = logfbank(signal=signal, samplerate=sample_rate, nfilt=nfilt, nfft=2048)
        feature = preprocessing.scale(feature)

        return feature

    @staticmethod
    def get_lm_batch(args, pny_lst, han_lst, acoustic_vocab, language_vocab):
        """
        训练语言模型batch数据，拼音到汉字
        :return:
        """
        shuffle_list = [i for i in range(len(pny_lst))]
        if args.shuffle is True:
            shuffle(shuffle_list)
        batch_num = len(pny_lst) // args.batch_size
        for k in range(batch_num):
            begin = k * args.batch_size
            end = begin + args.batch_size
            index_list = shuffle_list[begin:end]
            # max_len = max([len(pny_lst[index]) for index in index_list])
            max_len = args.max_len
            input_data = []
            label_data = []
            for i in index_list:
                try:
                    py_vec = Util.pny2id(pny_lst[i], acoustic_vocab) + [0] * (
                            max_len - len(pny_lst[i].strip().split(' ')))
                    han_vec = Util.han2id(han_lst[i], language_vocab, args.PAD_FLAG, args.SOS_FLAG, args.EOS_FLAG,
                                          args.PAD, args.SOS, args.EOS) + [0] * (max_len - len(han_lst[i].strip()))
                    input_data.append(py_vec)
                    label_data.append(han_vec)
                except ValueError:
                    continue
            input_data = np.array(input_data)
            label_data = np.array(label_data)
            yield input_data, label_data
        pass

    @staticmethod
    def predict_pinyin(model, inputs, input_length, acoustic_vocab):
        predict = model.predict(inputs, input_length)
        print('predict: {}'.format(predict))
        text = []
        for k in predict:
            text.append(acoustic_vocab[k])

        return predict, ' '.join(text)

    @staticmethod
    def GetEditDistance(str1, str2):
        """
        计算字错误率
        :param str1:
        :param str2:
        :return:
        """
        leven_cost = 0
        s = difflib.SequenceMatcher(None, str1, str2)
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            if tag == 'replace':
                leven_cost += max(i2 - i1, j2 - j1)
            elif tag == 'insert':
                leven_cost += (j2 - j1)
            elif tag == 'delete':
                leven_cost += (i2 - i1)
        return leven_cost

    # @staticmethod
    # def get_fbank_and_pinyin_data(file, acoustic_vocab):
    #     """
    #     获取一条语音数据的Fbank与拼音信息
    #     :param index: 索引位置
    #     :return:
    #         input_data: 语音特征数据
    #         data_length: 语音特征数据长度
    #         label: 语音标签的向量
    #         acoustic_vocab: 字典
    #     """
    #     try:
    #         # Fbank特征提取函数(从feature_python)
    #         fbank = None
    #         if os.path.isfile(file):
    #             fbank = Util.compute_fbank_from_file(file)
    #
    #         input_data = fbank.reshape([fbank.shape[0], fbank.shape[1], 1])
    #         data_length = input_data.shape[0] // 8 + 1
    #         label = Util.pny2id(self.pny_lst[index], acoustic_vocab)
    #         label = np.array(label)
    #         len_label = len(label)
    #         # 将错误数据进行抛出异常,并处理
    #         if input_data.shape[0] > self.feature_max_length:
    #             raise ValueError
    #         if len_label > 64 or len_label > data_length:
    #             raise ValueError
    #         return input_data, data_length, label, len_label
    #     except ValueError:
    #         raise ValueError
