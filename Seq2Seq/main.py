# -*- coding: utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
import argparse
import logging
import torch
import random
import numpy as np
import time
from time import strftime
from time import localtime
from torch.utils.data import DataLoader
from Seq2Seq import args
from flyai.dataset import Dataset

from Seq2Seq.transformer.encoder import Encoder
from Seq2Seq.transformer.decoder import Decoder
from Seq2Seq.utils.util import SortedByCountsDict, TransformerOptimizer, AiShellDataset, AudioDataset, AudioDataLoader, \
    Util, AverageMeter, Solver
from Seq2Seq.net import Net

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
        self.sortedDict = SortedByCountsDict(dump_dir=self.args.vocab_dump_dir, type='train')

        self.run()

    def generate(self):
        self.data = Dataset(epochs=self.args.EPOCHS, batch=self.args.BATCH, val_batch=self.args.BATCH)
        audio_paths, labels, _, _ = self.data.get_all_data()

        # wav文件路径
        audio_paths = [i['audio_path'] for i in audio_paths]
        # waw文本数据 TODO：此处包含空格, 测试去掉空格能否提升模型性能
        labels = [[j for j in list(i['label']) if j != '' and j != ' '] for i in labels]
        # labels = [list(i['label']) for i in labels]

        # 构建字典
        for label in labels:
            self.sortedDict.append_tokens(label)
        self.sortedDict.dump_pkl()

        audio_paths = np.asarray(audio_paths)
        labels = np.asarray(labels)
        index = [i for i in range(len(audio_paths))]
        np.random.shuffle(np.asarray(index))
        train_audio_paths, dev_audio_paths = audio_paths[index[0:int(len(index) * 0.9)]], audio_paths[
            index[int(len(index) * 0.9):]]
        train_labels, dev_labels = labels[index[0:int(len(index) * 0.9)]], labels[index[int(len(index) * 0.9):]]

        return train_audio_paths, train_labels, dev_audio_paths, dev_labels

    def train(self, train_audio_paths, train_labels, dev_audio_paths, dev_labels):
        best_loss = float('inf')
        epochs_since_improvement = 0
        vocab = self.sortedDict.get_vocab()

        encoder = Encoder(d_input=self.args.d_input * self.args.LFR_m, n_layers=self.args.n_layers_enc,
                          n_head=self.args.n_head, d_k=self.args.d_k, d_v=self.args.d_v, d_model=self.args.d_model,
                          d_inner=self.args.d_inner, dropout=self.args.dropout, pe_maxlen=self.args.pe_maxlen)

        decoder = Decoder(sos_id=self.args.SOS, eos_id=self.args.EOS, n_tgt_vocab=len(vocab),
                          d_word_vec=self.args.d_word_vec, n_layers=self.args.n_layers_dec, n_head=self.args.n_head,
                          d_k=self.args.d_k, d_v=self.args.d_v, d_model=self.args.d_model, d_inner=self.args.d_inner,
                          dropout=self.args.dropout, tgt_emb_prj_weight_sharing=self.args.tgt_emb_prj_weight_sharing,
                          pe_maxlen=self.args.pe_maxlen)

        model = Net(encoder=encoder, decoder=decoder)
        model = model.to(DEVICE)

        optimizer = TransformerOptimizer(torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
                                         self.args.k,
                                         self.args.d_model,
                                         self.args.warmup_steps)

        train_dataset = AudioDataset(args=self.args,
                                     samples=[{'wav': i, 'trn': j} for (i, j) in zip(train_audio_paths, train_labels)])

        train_loader = AudioDataLoader(train_dataset,
                                       batch_size=self.args.BATCH,
                                       shuffle=args.shuffle,
                                       feature_dim=args.d_input,
                                       char_list=vocab,
                                       path_list=train_audio_paths,
                                       label_list=train_labels,
                                       arguments=self.args)

        valid_dataset = AudioDataset(args=self.args,
                                     samples=[{'wav': i, 'trn': j} for (i, j) in zip(dev_audio_paths, dev_labels)])

        valid_loader = AudioDataLoader(valid_dataset,
                                       batch_size=self.args.BATCH,
                                       feature_dim=args.d_input,
                                       char_list=vocab,
                                       path_list=dev_audio_paths,
                                       label_list=dev_labels,
                                       arguments=self.args)

        # # Epochs
        # for epoch in range(self.args.EPOCHS):
        #     # One epoch's training
        #     # train mode (dropout and batchnorm is used)
        #     model.train()
        #     losses = AverageMeter()
        #
        #     # Batches
        #     for i, data in enumerate(train_loader):
        #         # Move to GPU, if available
        #         data = tuple(t.to(DEVICE) for t in data)
        #         padded_input, padded_target, input_lengths = data
        #
        #         # Forward prop.
        #         pred, gold = model(padded_input, input_lengths, padded_target)
        #         loss, n_correct = Util.cal_performance(pred=pred, gold=gold, smoothing=args.label_smoothing)
        #         # Back prop.
        #         optimizer.zero_grad()
        #         loss.backward()
        #         # Update weights
        #         optimizer.step()
        #         # Keep track of metrics
        #         losses.update(loss.item())
        #
        #         # Print status
        #         if i % self.args.print_freq == 0:
        #             logger.info('Epoch: [{0}][{1}/{2}]\t'
        #                         'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(epoch, i, len(train_loader), loss=losses))
        #
        #     logger.info('train_loss:{}/epoch:{}'.format(losses.avg, epoch))
        #     lr = optimizer._update_lr()
        #     logger.info('learning_rate:{}/epoch:{}'.format(lr, epoch))
        #     step_num = optimizer.step_num
        #     logger.info('Step num: {}\n'.format(step_num))
        #
        #     # One epoch's validation
        #     model.eval()
        #     losses = AverageMeter()
        #     # Batches
        #     for data in valid_loader:
        #         # Move to GPU, if available
        #         data = tuple(t.to(DEVICE) for t in data)
        #         padded_input, padded_target, input_lengths = data
        #
        #         with torch.no_grad():
        #             # Forward prop.
        #             pred, gold = model(padded_input, input_lengths, padded_target)
        #             loss, n_correct = Util.cal_performance(pred, gold, smoothing=args.label_smoothing)
        #
        #         # Keep track of metrics
        #         losses.update(loss.item())
        #
        #     # Print status
        #     logger.info('\nValidation Loss {loss.val:.5f} ({loss.avg:.5f})\n'.format(loss=losses))
        #
        #     # Check if there was an improvement
        #     is_best = losses.avg < best_loss
        #     best_loss = min(losses.avg, best_loss)
        #     if not is_best:
        #         epochs_since_improvement += 1
        #         print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        #     else:
        #         epochs_since_improvement = 0
        #
        #     # Save checkpoint
        #     Util.save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best,
        #                          output_dir=self.args.output_dir)

        solver = Solver(tr_loader=train_loader, cv_loader=valid_loader, model=model, optimizer=optimizer,
                        args=self.args)
        solver.train()

    def run(self):
        train_audio_paths, train_labels, dev_audio_paths, dev_labels = self.generate()
        self.train(train_audio_paths, train_labels, dev_audio_paths, dev_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASR')
    parser.add_argument("-e", '--EPOCHS', default=100, type=int, help='train epochs')
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
