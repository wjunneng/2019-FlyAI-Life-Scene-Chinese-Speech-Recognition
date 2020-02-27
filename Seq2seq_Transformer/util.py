# -*- coding=utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

import torch
import copy
import pandas as pd
import numpy as np
from flyai.dataset import Dataset

from Seq2seq_Transformer import args


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


if __name__ == '__main__':
    DataUtil()
