# -*- coding: utf-8 -*
import os
import sys

os.chdir(sys.path[0])
import argparse
import logging
import torch
import random
import numpy as np
from time import strftime
from time import localtime

from Seq2Seq import args
from flyai.dataset import Dataset

from Seq2Seq.Utils.util import SortedByCountsDict

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Instructor(object):
    """
    特点：使用flyai字典的get all data  | 自己进行划分next batch
    """

    def __init__(self, args):
        self.args = args
        self.generate()

    def generate(self):
        self.data = Dataset(epochs=self.args.EPOCHS, batch=self.args.BATCH, val_batch=self.args.BATCH)
        audio_paths, labels, _, _ = self.data.get_all_data()

        # wav文件路径
        audio_paths = [i['audio_path'] for i in audio_paths]
        # waw文本数据 TODO：此处包含空格, 测试去掉空格能否提升模型性能
        labels = [list(i['label']) for i in labels]

        # 构建字典
        sortedDict = SortedByCountsDict()
        for label in labels:
            sortedDict.append_tokens(label)
        vocab = sortedDict.get_vocab()
        ivocab = sortedDict.get_i_vocab()



    def run(self):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASR')
    parser.add_argument("-e", '--EPOCHS', default=10, type=int, help='train epochs')
    parser.add_argument('-b', '--BATCH', default=4, type=int, help='batch size')
    config = parser.parse_args()

    args.EPOCHS = config.EPOCHS
    args.BATCH = config.BATCH

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(args.seed)

    if os.path.exists(args.log_dir) is False:
        os.mkdir(args.log_dir)

    log_file = '{}-{}.log'.format(args.model_name, strftime('%y%m%d-%H%M', localtime()))
    logger.addHandler(logging.FileHandler(os.path.join(args.log_dir, log_file)))

    instructor = Instructor(args=args)
    instructor.run()
