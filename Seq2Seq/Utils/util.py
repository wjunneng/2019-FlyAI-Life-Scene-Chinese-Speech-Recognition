# -*- coding:utf-8 -*-
from collections import OrderedDict


class SortedByCountsDict(object):
    """
    构建具备自动排序的字典类
    """

    def __init__(self):
        self.vocab = OrderedDict()
        self.i_vocab = OrderedDict()

    def append_token(self, token: str):
        if token not in self.vocab:
            self.vocab[token] = 1
        else:
            self.vocab[token] += 1
            self.vocab = OrderedDict(sorted(self.vocab.items(), key=lambda item: item[1], reverse=True))

    def append_tokens(self, tokens: list):
        for token in tokens:
            self.append_token(token)

    def get_vocab(self):
        return self.vocab

    def get_i_vocab(self):
        return self.i_vocab
