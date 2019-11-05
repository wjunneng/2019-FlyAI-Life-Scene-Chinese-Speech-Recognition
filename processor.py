# -*- coding: utf-8 -*
import json
import numpy as np
import librosa

from flyai.processor.download import check_download
from flyai.processor.base import Base
from path import DATA_PATH, WORDS_PATH
from configuration.configuration import Configuration

# 汉明窗
w = 0.54 - 0.46 * np.cos(2 * np.pi * np.linspace(0, 400 - 1, 400, dtype=np.int64) / (400 - 1))


class Processor(Base):
    def __init__(self):
        self.max_audio_len = Configuration.max_audio_len
        self.max_tgt_len = Configuration.max_tgt_len
        self.char_dict = dict()
        self.char_dict_res = dict()

        # 构建字典
        with open(WORDS_PATH) as fin:
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
        # mfcc 梅尔倒谱系数
        mfcc = None
        try:
            path = check_download(audio_path, DATA_PATH)
            wav, sr = librosa.load(path, mono=True)
            mfcc = librosa.feature.mfcc(wav, sr, hop_length=int(0.010 * sr), n_fft=int(0.025 * sr))
            mfcc = mfcc.transpose((1, 0))
        except Exception as e:
            print('mfcc error %s' % e)

        try:
            if len(mfcc) >= self.max_audio_len:
                mfcc = mfcc[:self.max_audio_len]
                origanal_len = self.max_audio_len
            else:
                origanal_len = len(mfcc)
                mfcc = np.concatenate(
                    (mfcc, np.zeros([self.max_audio_len - origanal_len, Configuration.embedding_dim])), 0)

            # 最后一行元素为句子实际长度
            mfcc = np.concatenate(
                (mfcc, np.array([origanal_len for _ in range(Configuration.embedding_dim)]).reshape(
                    [1, Configuration.embedding_dim])))
        except Exception as e:
            print('conc error %s' % e)

        return mfcc

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
