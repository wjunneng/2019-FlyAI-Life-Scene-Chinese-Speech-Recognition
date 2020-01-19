# -*- coding:utf-8 -*-

import os
import sys

os.chdir(sys.path[0])
import tensorflow as tf
import numpy as np
from keras.layers import Input, Reshape, Dense, Lambda, Dropout
from keras.layers.recurrent import GRU
from keras.layers.merge import add
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from keras.utils import multi_gpu_model

from DFCNN_Transformer.util.util import Util


class CNNRNNCTCModel(object):
    """
        docstring for Amodel.
    """

    def __init__(self, args, vocab_size):
        # 神经网络最终输出的每一个字符向量维度的大小
        self.vocab_size = vocab_size
        self.gpu_nums = args.am_gpu_nums
        self.lr = args.am_lr
        self.feature_length = 200
        self.is_training = args.am_is_training
        self._model_init()
        if self.is_training:
            self._ctc_init()
            self.opt_init()

    def _model_init(self):
        self.inputs = Input(name='the_inputs', shape=(None, 200, 1))
        x = Reshape((-1, 200))(self.inputs)
        x = CNNRNNCTCModel.dense(512, x)
        x = CNNRNNCTCModel.dense(512, x)
        x = CNNRNNCTCModel.bi_gru(512, x)
        x = CNNRNNCTCModel.bi_gru(512, x)
        x = CNNRNNCTCModel.bi_gru(512, x)
        x = CNNRNNCTCModel.dense(512, x)
        self.outputs = CNNRNNCTCModel.dense(self.vocab_size, x, activation='softmax')
        self.model = Model(inputs=self.inputs, outputs=self.outputs)
        self.model.summary()

    def _ctc_init(self):
        self.labels = Input(name='the_labels', shape=[None], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')
        self.loss_out = Lambda(CNNRNNCTCModel.ctc_lambda, output_shape=(1,), name='ctc') \
            ([self.labels, self.outputs, self.input_length, self.label_length])
        self.ctc_model = Model(inputs=[self.labels, self.inputs, self.input_length, self.label_length],
                               outputs=self.loss_out)

    def predict(self, data_input, length, batch_size=1):
        """
        返回预测结果
        :param date_input:
        :param input_len:
        :return:
        """
        x_in = np.zeros((batch_size, 1600, self.feature_length, 1), dtype=np.float32)
        for i in range(batch_size):
            if len(data_input) > 1600:
                x_in[i, 0:1600] = data_input[:1600]
            else:
                x_in[i, 0:len(data_input)] = data_input

        # 通过输入，得到预测的值，预测的值即为网络中softmax的输出
        # shape = [1, 200, 1424]
        # 还需要通过ctc_loss的网络进行解码
        pred = self.model.predict(x_in, steps=1)

        return Util.decode_ctc(pred, length)

    def opt_init(self):
        opt = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.01, epsilon=10e-8)
        if self.gpu_nums > 1:
            self.ctc_model = multi_gpu_model(self.ctc_model, gpus=self.gpu_nums)
        self.ctc_model.compile(loss={'ctc': lambda y_true, output: output}, optimizer=opt)

    @staticmethod
    def bi_gru(units, x):
        x = Dropout(0.2)(x)
        y1 = GRU(units, return_sequences=True, kernel_initializer='he_normal')(x)
        y2 = GRU(units, return_sequences=True, go_backwards=True, kernel_initializer='he_normal')(x)
        y = add([y1, y2])
        return y

    @staticmethod
    def dense(units, x, activation="relu"):
        x = Dropout(0.2)(x)
        y = Dense(units, activation=activation, use_bias=True, kernel_initializer='he_normal')(x)
        return y

    @staticmethod
    def ctc_lambda(args):
        labels, y_pred, input_length, label_length = args
        y_pred = y_pred[:, :, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    @staticmethod
    def am_hparams():
        params = tf.contrib.training.HParams(
            vocab_size=50,
            lr=0.001,
            gpu_nums=1,
            is_training=True)
        return params
