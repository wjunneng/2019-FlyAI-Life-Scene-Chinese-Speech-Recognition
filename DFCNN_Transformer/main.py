# -*- coding: utf-8 -*
import os
import sys

os.chdir(sys.path[0])
import warnings

warnings.filterwarnings('ignore')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import logging
import torch
import random
import shutil
import keras
import numpy as np
from time import strftime, localtime
from pypinyin import pinyin, Style
from flyai.dataset import Dataset
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from DFCNN_Transformer import args
from DFCNN_Transformer.util.util import SortedByCountsDict, Util, DataGenerator
from DFCNN_Transformer.module.am_cnn_ctc import CNNCTCModel
from DFCNN_Transformer.module.lm_transformer import TransformerModel

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
        self.sortedDict = SortedByCountsDict(dump_dir=self.args.vocab_dump_dir)
        self.acoustic_vocab_size, self.acoustic_vocab = Util.get_acoustic_vocab_list()
        self.language_vocab_size, self.language_vocab = Util.get_language_vocab_list()

    def generate(self):
        self.data = Dataset(epochs=self.args.EPOCHS, batch=self.args.BATCH, val_batch=self.args.BATCH)
        audio_paths, labels, _, _ = self.data.get_all_data()

        # wav文件路径
        audio_paths = [i['audio_path'] for i in audio_paths]
        # wav文本数据 TODO：此处包含空格, 测试去掉空格能否提升模型性能
        audio_labels = []
        # wav文本拼音
        audio_pinyins = []
        for label in labels:
            label = label['label'].split(' ')
            audio_labels.append(''.join(label))
            audio_pinyins.append(' '.join(
                [' '.join([' '.join(j) for j in pinyin(i, style=Style.TONE3, heteronym=False)]) for i in label]))

        # 构建字典
        for label in labels:
            self.sortedDict.append_tokens(label)
        self.sortedDict.dump_pkl()

        # 划分训练/验证
        audio_paths = np.asarray(audio_paths)
        audio_labels = np.asarray(audio_labels)
        audio_pinyins = np.asarray(audio_pinyins)

        index = [i for i in range(len(audio_paths))]
        np.random.shuffle(np.asarray(index))
        train_audio_paths, dev_audio_paths = audio_paths[index[0:int(len(index) * 0.9)]], audio_paths[
            index[int(len(index) * 0.9):]]
        train_labels, dev_labels = audio_labels[index[0:int(len(index) * 0.9)]], audio_labels[
            index[int(len(index) * 0.9):]]
        train_pinyins, dev_pinyins = audio_pinyins[index[0:int(len(index) * 0.9)]], audio_pinyins[
            index[int(len(index) * 0.9):]]

        return train_audio_paths.tolist(), train_labels.tolist(), train_pinyins.tolist(), dev_audio_paths.tolist(), dev_labels.tolist(), dev_pinyins.tolist()

    def train_am(self, train_audio_paths, train_labels, train_pinyins, dev_audio_paths, dev_labels, dev_pinyins):
        """
        训练声学模型
        :param train_audio_paths:
        :param train_labels:
        :param train_pinyins:
        :param dev_audio_paths:
        :param dev_labels:
        :param dev_pinyins:
        :return:
        """
        model = CNNCTCModel(args=self.args, vocab_size=self.acoustic_vocab_size)

        hp = self.args
        hp.batch_size = self.args.am_batch_size
        hp.epochs = self.args.am_epochs
        hp.data_path = self.args.wav_dir
        hp.data_type = 'train'
        hp.feature_max_length = hp.am_feature_max_length
        train_generator = DataGenerator(audio_paths=train_audio_paths, labels=train_labels, pinyins=train_pinyins,
                                        hp=hp, acoustic_vocab=self.acoustic_vocab)
        hp.data_type = 'dev'
        dev_generator = DataGenerator(audio_paths=dev_audio_paths, labels=dev_labels, pinyins=dev_pinyins,
                                      hp=hp, acoustic_vocab=self.acoustic_vocab)
        ckpt = "model_{epoch:02d}-{val_loss:.2f}.hdf5"
        cpCallBack = ModelCheckpoint(os.path.join(self.args.AmModelFolder, ckpt), verbose=1, save_best_only=True)
        tbCallBack = keras.callbacks.TensorBoard(log_dir=self.args.AmModelTensorBoard, histogram_freq=0,
                                                 write_graph=True, write_images=True, update_freq='epoch')

        select_model = '0'
        if os.path.exists(hp.AmModelFolder + select_model + '.hdf5'):
            print('load acoustic model...')
            model.load_model(select_model)

        model.ctc_model.fit_generator(train_generator,
                                      steps_per_epoch=len(train_pinyins) // hp.batch_size,
                                      validation_data=dev_generator,
                                      validation_steps=20,
                                      epochs=hp.epochs,
                                      workers=10,
                                      use_multiprocessing=False,
                                      callbacks=[cpCallBack, tbCallBack]
                                      )

    def train_lm(self, train_labels, train_pinyins):
        """
        训练语言学模型
        :param train_labels:
        :param train_pinyins:
        :param dev_audio_paths:
        :param dev_labels:
        :param dev_pinyins:
        :return:
        """
        hp = self.args
        hp.batch_size = self.args.lm_batch_size
        hp.epochs = self.args.lm_epochs
        hp.data_type = 'train'
        hp.max_len = self.args.lm_max_len
        hp.hidden_units = self.args.lm_hidden_units
        hp.is_training = self.args.lm_is_training
        hp.feature_dim = self.args.lm_feature_dim
        hp.num_heads = self.args.lm_num_heads
        hp.num_blocks = self.args.lm_num_blocks
        hp.position_max_length = self.args.lm_position_max_length
        hp.lr = self.args.lm_lr
        hp.dropout_rate = self.args.lm_dropout_rate

        epochs = hp.epochs
        lm_model = TransformerModel(arg=hp, acoustic_vocab_size=self.acoustic_vocab_size,
                                    language_vocab_size=self.language_vocab_size)

        batch_num = len(train_pinyins) // hp.batch_size
        with lm_model.graph.as_default():
            saver = tf.train.Saver(max_to_keep=50)
            config = tf.ConfigProto()
            # 占用GPU90%的显存
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
        with tf.Session(graph=lm_model.graph, config=config) as sess:
            merged = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())
            add_num = 0
            if os.path.exists(hp.LmModelFolder):
                print('loading language model...')
                latest = tf.train.latest_checkpoint(hp.LmModelFolder)
                if latest is not None:
                    add_num = int(latest.split('_')[-2])
                    saver.restore(sess, latest)
            writer = tf.summary.FileWriter(hp.LmModelTensorboard, tf.get_default_graph())
            for k in range(epochs):
                total_loss = 0
                batch = Util.get_lm_batch(args=hp, pny_lst=train_pinyins, han_lst=train_labels,
                                          acoustic_vocab=self.acoustic_vocab, language_vocab=self.language_vocab)
                for i in range(batch_num):
                    input_batch, label_batch = next(batch)
                    feed = {lm_model.x: input_batch, lm_model.y: label_batch}
                    print('input_batch:{}'.format(input_batch))
                    print('label_batch:{}'.format(label_batch))
                    print('len:{}'.format(len(label_batch)))
                    cost, _ = sess.run([lm_model.mean_loss, lm_model.train_op], feed_dict=feed)
                    total_loss += cost
                    if i % 10 == 0:
                        print("epoch: %d step: %d/%d  train loss=6%f" % (k + 1, i, batch_num, cost))
                        if i % 5000 == 0:
                            rs = sess.run(merged, feed_dict=feed)
                            writer.add_summary(rs, k * batch_num + i)
                print('epochs', k + 1, ': average loss = ', total_loss / batch_num)
                saver.save(sess, hp.LmModelFolder + 'model_%d_%.3f.ckpt' % (k + 1 + add_num, total_loss / batch_num))
            writer.close()
        pass

    def run(self):
        # 拷贝文件
        for name, after_dir in zip(['dict.txt', 'hanzi.txt', 'mixdict.txt'],
                                   [self.args.dict_dir, self.args.hanzi_dir, self.args.mixdict_dir]):
            before_dir = os.path.join(os.getcwd(), 'attach_data', name)
            logger.info('>>>name:{}'.format(name))
            logger.info('>before_dir:{}'.format(before_dir))
            logger.info('>after_dir:{}'.format(after_dir))
            shutil.copyfile(before_dir, after_dir)

        train_audio_paths, train_labels, train_pinyins, dev_audio_paths, dev_labels, dev_pinyins = self.generate()
        logger.info('start train am model!')
        self.train_am(train_audio_paths, train_labels, train_pinyins, dev_audio_paths, dev_labels, dev_pinyins)
        logger.info('end train am model!')

        logger.info('start train lm model!')
        self.train_lm(train_labels=train_labels, train_pinyins=train_pinyins)
        logger.info('end train lm model!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASR')
    parser.add_argument("-e", '--EPOCHS', default=20, type=int, help='train epochs')
    parser.add_argument('-b', '--BATCH', default=5, type=int, help='batch size')
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
