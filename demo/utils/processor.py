# -*- coding: utf-8 -*
import os
import json
import numpy as np

from demo.utils import features
from demo.configuration.configuration import Configuration


class Processor(object):
    def __init__(self):
        self.max_audio_len = Configuration.max_audio_len
        self.max_tgt_len = Configuration.max_tgt_len
        self.char_dict = dict()
        self.char_dict_res = dict()

        # 构建字典
        with open(Configuration.WORDS_PATH) as fin:
            words = json.loads(fin.read())
        words = list(words.keys())
        words = [" ", "<unk>"] + words
        for i, word in enumerate(words):
            self.char_dict[word] = i
            self.char_dict_res[i] = word
        print('vocab len: %d' % len(words))

    def input_x(self, audio_path):
        """
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        【该参数需要与app.yaml的Model的input-->columns->name 一一对应】
        :param audio_path: wav路径
        :return:
        """
        wav_features = features.Features(os.path.join(Configuration.DATA_PATH, audio_path)).method_2()

        return wav_features

    def input_y(self, label):
        """
        获取中文序列的索引及长度
        【该参数需要与app.yaml的Model的input-->columns->name 一一对应】
        :param label:
        :return:
        """
        print('label: %s' % label)
        # 获取单词索引
        word_list = [self.char_dict.get(word) for word in label if self.char_dict.get(word) is not None]

        origanal_len = len(word_list)
        if len(word_list) >= self.max_tgt_len:
            origanal_len = self.max_tgt_len
            word_list = word_list[:self.max_tgt_len]
        else:
            for i in range(len(word_list), self.max_tgt_len):
                # 不够长度则补0
                word_list.append(0)
        # 最后一个元素为句子长度x
        word_list.append(origanal_len)

        return word_list

    def output_y(self, data):
        """
        验证时使用，把模型输出的y转为对应的结果
        :param data:
        :return:
        """
        output_words = [self.char_dict_res[np.argmax(word_prob)] for word_prob in data]

        return output_words

    def get_batch(self, input_data, label_data, batch_size):
        """
        获取batch数据
        :param input_data:
        :param label_data:
        :param batch_size:
        :return:
        """
        # 计算batch数目
        batch_num = len(input_data) // batch_size
        # 遍历
        for k in range(batch_num):
            begin = k * batch_size
            end = begin + batch_size
            input_batch = input_data[begin:end]
            label_batch = label_data[begin:end]

            yield np.array(input_batch), np.array(label_batch)
