# -*- coding: utf-8 -*
import os
import json
import numpy as np

from utils.features import Features
from configurations.constant import Constant

# 模型类型
# TYPE = 'seq2seq'
TYPE = 'transformer'


class Processor(object):
    def __init__(self):
        self.project_path = Constant(type=TYPE).get_project_path()
        self.configuration = Constant(type=TYPE).get_configuration()
        self.max_audio_len = self.configuration.max_audio_len
        self.max_tgt_len = self.configuration.max_tgt_len
        self.char_dict = dict()
        self.char_dict_res = dict()
        self.eos_id = self.configuration.EOS

        self.MODEL_PATH = os.path.join(self.project_path, self.configuration.MODEL_PATH)
        self.WORDS_PATH = os.path.join(self.project_path, self.configuration.WORDS_PATH)
        self.DEV_PATH = os.path.join(self.project_path, self.configuration.DEV_PATH)
        self.DATA_PATH = os.path.join(self.project_path, self.configuration.DATA_PATH)

        # 构建字典
        with open(self.WORDS_PATH) as fin:
            words = json.loads(fin.read())
        words = list(words.keys())
        if TYPE == 'seq2seq':
            # 去除
            words = [" ", "<unk>"] + words
        elif TYPE == 'transformer':
            # 新增
            words = [self.configuration.PAD_FLAG,
                     self.configuration.UNK_FLAG,
                     self.configuration.SOS_FLAG,
                     self.configuration.EOS_FLAG,
                     self.configuration.SPACE_FLAG] + words

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
        audio_path = os.path.join(self.DATA_PATH, audio_path)
        wav_features = None
        try:
            # 方法一
            wav_features = Features(wav_path=audio_path).method_1()
            # 方法二
            # wav_features = Features(wav_path=path, type=self.type).method_2()
        except Exception as e:
            print('error %s' % e)

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

        if TYPE == 'seq2seq':
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

        elif TYPE == 'transformer':
            # 添加eos_id
            word_list.append(self.eos_id)

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
