# -*- coding: utf-8 -*
import os
from tqdm import tqdm
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data.dataloader import default_collate


import net
import net_1
from model import Model
from path import MODEL_PATH
from processor import Processor
from flyai.dataset import Dataset
from configurations.constant import Constant
from utils.util import AverageMeter, Util, AiShellDataset


class Seq2seq(object):
    def __init__(self):
        self.configuration = Constant(type='seq2seq').get_configuration()
        self.embedding_dim = self.configuration.embedding_dim
        self.hidden_dim = self.configuration.hidden_dim
        self.output_dim = self.configuration.output_dim

    def main(self):
        # 超参
        parser = argparse.ArgumentParser()
        # 训练轮数
        parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
        # 训练批次
        parser.add_argument("-b", "--BATCH", default=4, type=int, help="batch size")
        # 获取超参数
        args = parser.parse_args()

        # 获取device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 创建文件夹
        if os.path.exists(MODEL_PATH) is False:
            os.makedirs(MODEL_PATH)

        # 数据获取辅助类
        data = Dataset(epochs=args.EPOCHS, batch=args.BATCH, val_batch=args.BATCH)

        # 定义网络
        encoder = net.Encoder(embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim)
        decoder = net.Decoder(output_dim=self.output_dim, embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim)
        network = net.Net(encoder=encoder, decoder=decoder, device=device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = Adam(network.parameters(), lr=0.001)
        model = Model(dataset=data, configuration=self.configuration)

        lowest_loss = 10
        # 得到训练和测试的数据
        for i in range(data.get_step()):
            # 开启训练状态
            network.train()

            # 得到训练和测试的数据
            # X_train: shape:(batch, sen_len, embedding) eg.(16, 901, 20)
            # y_train: shape:(batch, 语音文字最大长度) eg.(16, 39)
            x_train, y_train = data.next_train_batch()

            # 每段语音序列长度 input_length: (batch_size) .eg(4)
            input_lengths = [int(x) for x in x_train[:, -1, 0].tolist()]
            input_lengths = torch.tensor(input_lengths).long().to(device)

            # 每段语音对应中文序列长度 y_lengths: (batch_size) .eg(4)
            y_lengths = y_train[:, -1].tolist()
            y_lengths = torch.tensor(y_lengths).long().to(device)

            # 去除长度信息 .eg(4, 900, 20)
            x_train = x_train[:, :-1, :]
            # 去除长度信息 .eg(4, 38)
            y_train = y_train[:, :-1]

            # 转化为tensor
            x_train = torch.from_numpy(x_train).float().to(device)
            # 转化为tensor
            y_train = torch.from_numpy(y_train).long().to(device)

            seq_pairs = sorted(zip(x_train.contiguous(), y_train.contiguous(), input_lengths, y_lengths),
                               key=lambda x: x[2], reverse=True)

            x_train, y_train, input_lengths, y_lengths = zip(*seq_pairs)

            # 更换维度 eg.(batch_size, audio_en_size, emb_size) -> (audio_en_size, batch_size, emb_size)
            x_train = torch.stack(x_train, dim=0).permute(1, 0, 2).contiguous()
            y_train = torch.stack(y_train, dim=0).permute(1, 0).contiguous()

            # 输出结果
            outputs = network(src_seqs=x_train, src_lengths=input_lengths, trg_seqs=y_train).float().to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 计算损失
            loss = loss_function(outputs.view(-1, outputs.shape[2]), y_train.view(-1))
            # 反向传播
            loss.backward()
            # 使用adam调整参数
            optimizer.step()

            print(loss)
            if loss < lowest_loss:
                lowest_loss = loss
                model.save_model(network, MODEL_PATH, overwrite=True)
                print("step %d, best lowest_loss %g" % (i, lowest_loss))
            print(str(i) + "/" + str(data.get_step()))


class Transformer(object):
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        configuration = Constant(type='transformer').get_configuration()
        self.label_smoothing = configuration.label_smoothing
        self.print_freq = configuration.print_freq
        self.vocab_size = configuration.vocab_size

        self.LFR_m = configuration.LFR_m
        self.LFR_n = configuration.LFR_n
        self.d_input = configuration.d_input
        self.n_layers_enc = configuration.n_layers_enc
        self.n_head = configuration.n_head
        self.d_k = configuration.d_k
        self.d_v = configuration.d_v
        self.d_model = configuration.d_model
        self.d_inner = configuration.d_inner
        self.dropout = configuration.dropout
        self.pe_maxlen = configuration.pe_maxlen
        self.d_word_vec = configuration.d_word_vec
        self.n_layers_dec = configuration.n_layers_dec
        self.tgt_emb_prj_weight_sharing = configuration.tgt_emb_prj_weight_sharing
        self.label_smoothing = configuration.label_smoothing
        self.epochs = configuration.epochs
        self.shuffle = configuration.shuffle
        self.batch_size = configuration.batch_size
        self.batch_frames = configuration.batch_frames
        self.maxlen_in = configuration.maxlen_in
        self.maxlen_out = configuration.maxlen_out
        self.num_workers = configuration.num_workers
        self.k = configuration.k
        self.lr = configuration.lr
        self.warmup_steps = configuration.warmup_steps
        self.checkpoint = configuration.checkpoint
        self.pad_id = configuration.PAD
        self.sos_id = configuration.SOS
        self.eos_id = configuration.EOS
        # 获取device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.configuration = configuration

    def pad_collate(self, batch):
        max_input_len = float('-inf')
        max_target_len = float('-inf')

        for elem in batch:
            feature, trn = elem
            max_input_len = max_input_len if max_input_len > feature.shape[0] else feature.shape[0]
            max_target_len = max_target_len if max_target_len > len(trn) else len(trn)

        for i, elem in enumerate(batch):
            feature, trn = elem
            input_length = feature.shape[0]
            input_dim = feature.shape[1]
            padded_input = np.zeros((max_input_len, input_dim), dtype=np.float32)
            padded_input[:input_length, :] = feature
            padded_target = np.pad(trn, (0, max_target_len - len(trn)), 'constant', constant_values=self.pad_id)
            batch[i] = (padded_input, padded_target, input_length)

        # sort it by input lengths (long to short)
        batch.sort(key=lambda x: x[2], reverse=True)

        return default_collate(batch)

    def main(self):
        torch.manual_seed(7)
        np.random.seed(7)
        checkpoint = self.checkpoint
        start_epoch = 0
        best_loss = float('inf')
        epochs_since_improvement = 0

        # 创建文件夹
        if os.path.exists(MODEL_PATH) is False:
            os.makedirs(MODEL_PATH)

        # 数据获取辅助类
        x_train, y_train_old, x_val, y_val_old = Dataset(epochs=self.epochs, batch=self.batch_size,
                                                 val_batch=self.batch_size).get_all_data()

        y_train = [{'label': Processor().input_y(label=i['label'])} for i in y_train_old]
        y_val = [{'label': Processor().input_y(label=i['label'])} for i in y_val_old]

        train = list(zip(x_train, y_train))
        val = list(zip(x_val, y_val))

        # Initialize / load checkpoint
        if checkpoint == 'None':
            # model
            encoder = net_1.Encoder(d_input=self.d_input * self.LFR_m,
                                    n_layers=self.n_layers_enc,
                                    n_head=self.n_head,
                                    d_k=self.d_k,
                                    d_v=self.d_v,
                                    d_model=self.d_model,
                                    d_inner=self.d_inner,
                                    dropout=self.dropout,
                                    pe_maxlen=self.pe_maxlen)
            decoder = net_1.Decoder(pad_id=self.pad_id,
                                    sos_id=self.sos_id,
                                    eos_id=self.eos_id,
                                    n_tgt_vocab=self.vocab_size,
                                    d_word_vec=self.d_word_vec,
                                    n_layers=self.n_layers_dec,
                                    n_head=self.n_head,
                                    d_k=self.d_k,
                                    d_v=self.d_v,
                                    d_model=self.d_model,
                                    d_inner=self.d_inner,
                                    dropout=self.dropout,
                                    tgt_emb_prj_weight_sharing=self.tgt_emb_prj_weight_sharing,
                                    pe_maxlen=self.pe_maxlen)
            model = net_1.Transformer(encoder, decoder)

            # optimizer
            optimizer = net_1.TransformerOptimizer(
                torch.optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-09))

        else:
            checkpoint = torch.load(checkpoint)
            start_epoch = checkpoint['epoch'] + 1
            epochs_since_improvement = checkpoint['epochs_since_improvement']
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']

        logger = Util.get_logger()

        # Move to GPU, if available
        model = model.to(self.device)

        # Custom dataloaders
        train_dataset = AiShellDataset(args=self.configuration, samples=train, split='train')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                   collate_fn=self.pad_collate,
                                                   pin_memory=True, shuffle=True, num_workers=self.num_workers)
        valid_dataset = AiShellDataset(args=self.configuration, samples=val, split='val')
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size,
                                                   collate_fn=self.pad_collate,
                                                   pin_memory=True, shuffle=False, num_workers=self.num_workers)

        # Epochs
        for epoch in range(start_epoch, self.epochs):
            # One epoch's training
            train_loss = self.train(train_loader=train_loader,
                                    model=model,
                                    optimizer=optimizer,
                                    epoch=epoch,
                                    logger=logger)
            print('epoch: %d, train_loss: %s' % (epoch, str(train_loss)))

            lr = optimizer.lr
            print('\nLearning rate: {}'.format(lr))
            step_num = optimizer.step_num
            print('Step num: {}\n'.format(step_num))

            # One epoch's validation
            valid_loss = self.valid(valid_loader=valid_loader,
                                    model=model,
                                    logger=logger)

            # Check if there was an improvement
            is_best = valid_loss < best_loss
            best_loss = min(valid_loss, best_loss)
            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

            # Save checkpoint
            Util.save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)

    def train(self, train_loader, model, optimizer, epoch, logger):
        # train mode (dropout and batchnorm is used)
        model.train()

        losses = AverageMeter()

        # Batches
        for i, (data) in enumerate(train_loader):
            # Move to GPU, if available
            padded_input, padded_target, input_lengths = data
            padded_input = padded_input.to(self.device)
            padded_target = padded_target.to(self.device)
            input_lengths = input_lengths.to(self.device)

            # Forward prop.
            pred, gold = model(padded_input, input_lengths, padded_target)
            loss, n_correct = net_1.Util.cal_performance(pad_id=self.pad_id, pred=pred, gold=gold,
                                                         smoothing=self.label_smoothing)

            # Back prop.
            optimizer.zero_grad()
            loss.backward()

            # Update weights
            optimizer.step()

            # Keep track of metrics
            losses.update(loss.item())

            # Print status
            if i % self.print_freq == 0:
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                            'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(epoch, i, len(train_loader), loss=losses))

        return losses.avg

    def valid(self, valid_loader, model, logger):
        model.eval()

        losses = AverageMeter()

        # Batches
        for data in tqdm(valid_loader):
            # Move to GPU, if available
            padded_input, padded_target, input_lengths = data
            padded_input = padded_input.to(self.device)
            padded_target = padded_target.to(self.device)
            input_lengths = input_lengths.to(self.device)

            with torch.no_grad():
                # Forward prop.
                pred, gold = model(padded_input, input_lengths, padded_target)
                loss, n_correct = net_1.Util.cal_performance(pad_id=self.pad_id, pred=pred, gold=gold,
                                                             smoothing=self.label_smoothing)

            # Keep track of metrics
            losses.update(loss.item())

        # Print status
        logger.info('\nValidation Loss {loss.val:.5f} ({loss.avg:.5f})\n'.format(loss=losses))

        return losses.avg


if __name__ == '__main__':
    Transformer().main()
