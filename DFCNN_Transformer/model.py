# -*- coding: utf-8 -*
import os
import sys

os.chdir(sys.path[0])
import torch
import numpy as np
import datetime
from numpy import random
import tensorflow as tf
import soundfile as sf
from flyai.model.base import Base

from DFCNN_Transformer.module.am_cnn_ctc import CNNCTCModel
from DFCNN_Transformer.module.lm_transformer import TransformerModel
from DFCNN_Transformer.util.util import Util
from DFCNN_Transformer import args

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Model(Base):
    def __init__(self, dataset):
        self.dataset = dataset
        self.acoustic_vocab_size, self.acoustic_vocab = Util.get_acoustic_vocab_list()
        self.language_vocab_size, self.language_vocab = Util.get_language_vocab_list()

        self.am_args, self.lm_args = args, args
        self.am_args.data_length = None

        self.lm_args.lr = args.lm_lr
        self.lm_args.is_training = False
        self.lm_args.max_len = args.lm_max_len
        self.lm_args.hidden_units = args.lm_hidden_units
        self.lm_args.feature_dim = args.lm_feature_dim
        self.lm_args.num_heads = args.lm_num_heads
        self.lm_args.num_blocks = args.lm_num_blocks
        self.lm_args.position_max_length = args.lm_position_max_length
        self.lm_args.dropout_rate = args.lm_dropout_rate

    def speech_predict(self, am_model, lm_model, predict_data, num, sess):
        """
        预测结果
        :param am_model:
        :param lm_model:
        :param predict_data:
        :param num:
        :param sess:
        :return:
        """

        if os.path.exists(self.lm_args.PredResultFolder):
            os.mkdir(self.lm_args.PredResultFolder)

        num_data = len(predict_data.pny_lst)
        length = predict_data.data_length
        if length is None:
            length = num_data
        ran_num = random.randint(0, length - 1)
        words_num, word_error_num, han_num, han_error_num = 0, 0, 0, 0
        data = ''
        for i in range(num):
            print('\nthe ', i + 1, 'th example.')
            # 载入训练好的模型，并进行识别
            index = (ran_num + i) % num_data
            try:
                hanzi = predict_data.han_lst[index]
                hanzi_vec = [self.language_vocab.index(idx) for idx in hanzi]
                inputs, input_length, label, _ = Util.get_fbank_and_pinyin_data(index=index,
                                                                                acoustic_vocab=self.acoustic_vocab)
                pred, pinyin = Util.predict_pinyin(model=am_model, inputs=inputs, input_length=input_length,
                                                   acoustic_vocab=self.acoustic_vocab)
                y = predict_data.pny_lst[index]

                # 语言模型预测
                with sess.as_default():
                    py_in = pred.reshape(1, -1)
                    han_pred = sess.run(lm_model.preds, {lm_model.x: py_in})
                    han = ''.join(self.language_vocab[idx] for idx in han_pred[0])
            except ValueError:
                continue
            print('原文汉字结果:', ''.join(hanzi))
            print('原文拼音结果:', ''.join(y))
            print('预测拼音结果:', pinyin)
            print('预测汉字结果:', han)
            data += '原文汉字结果:' + ''.join(hanzi) + '\n'
            data += '原文拼音结果:' + ''.join(y) + '\n'
            data += '预测拼音结果:' + pinyin + '\n'
            data += '预测汉字结果:' + han + '\n'

            words_n = label.shape[0]
            # 把句子的总字数加上
            words_num += words_n
            py_edit_distance = Util.GetEditDistance(label, pred)
            # 拼音距离
            # 当编辑距离小于等于句子字数时
            if (py_edit_distance <= words_n):
                # 使用编辑距离作为错误字数
                word_error_num += py_edit_distance
                # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字
            else:
                # 就直接加句子本来的总字数就好了
                word_error_num += words_n

                # 汉字距离
            words_n = np.array(hanzi_vec).shape[0]
            # 把句子的总字数加上
            han_num += words_n
            han_edit_distance = Util.GetEditDistance(np.array(hanzi_vec), han_pred[0])
            # 当编辑距离小于等于句子字数时
            if han_edit_distance <= words_n:
                # 使用编辑距离作为错误字数
                han_error_num += han_edit_distance
            # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字
            else:
                # 就直接加句子本来的总字数就好了
                han_error_num += words_n

        data += '*[Predict Result] Speech Recognition set word accuracy ratio: ' + str(
            (1 - word_error_num / words_num) * 100) + '%'
        filename = str(datetime.datetime.now()) + '_' + str(num)
        with open(os.path.join(self.lm_args.PredResultFolder, filename), mode='w', encoding='utf-8') as f:
            f.writelines(data)
        print('*[Predict Result] Speech Recognition set 拼音 word accuracy ratio: ',
              (1 - word_error_num / words_num) * 100, '%')
        print('*[Predict Result] Speech Recognition set 汉字 word accuracy ratio: ',
              (1 - han_error_num / han_num) * 100, '%')

    def predict(self, **data):
        if os.path.exists(self.lm_args.PredResultFolder):
            os.mkdir(self.lm_args.PredResultFolder)
        audio_path = self.dataset.predict_data(**data)[0]

        # 声学模型
        am_model = CNNCTCModel(args=self.am_args, vocab_size=self.acoustic_vocab_size)
        am_model.load_model(os.path.join(self.am_args.AmModelFolder, self.am_args.am_ckpt))

        # 语言模型
        lm_model = TransformerModel(arg=self.lm_args, acoustic_vocab_size=self.acoustic_vocab_size,
                                    language_vocab_size=self.language_vocab_size)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        sess = tf.Session(graph=lm_model.graph, config=tf.ConfigProto(gpu_options=gpu_options))
        with lm_model.graph.as_default():
            saver = tf.train.Saver()
        latest = tf.train.latest_checkpoint(self.lm_args.LmModelFolder)
        saver.restore(sess, latest)

        # 声学模型预测
        signal, sample_rate = sf.read(audio_path)
        fbank = Util.compute_fbank_from_api(signal, sample_rate)
        input_data = fbank.reshape([fbank.shape[0], fbank.shape[1], 1])
        data_length = input_data.shape[0] // 8 + 1
        predict, pinyin = Util.predict_pinyin(model=am_model, inputs=input_data, input_length=data_length,
                                              acoustic_vocab=self.acoustic_vocab)
        # 语言模型预测
        with sess.as_default():
            py_in = predict.reshape(1, -1)
            han_pred = sess.run(lm_model.preds, {lm_model.x: py_in})
            han = ''.join(self.language_vocab[idx] for idx in han_pred[0])

        print(han)

    def predict_all(self, datas):
        labels = []
        for data in datas:
            predicts = self.predict(audio_path=data['audio_path'])

            labels.append(predicts)

        return labels
