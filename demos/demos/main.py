# -*- coding: utf-8 -*
from torch.optim import Adam
from torch import nn
import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate

from demos.demos.processor import Processor
from demos.models import net
from demos.models import net_1
from demos.models.model import Model
from configurations.constant import Constant
from utils.util import AverageMeter, Util, AiShellDataset


class Seq2Seq(object):
    def __init__(self):
        type = 'seq2seq'
        self.project_path = Constant(type=type).get_project_path()
        self.configuration = Constant(type=type).get_configuration()

        self.epoch = self.configuration.epoch
        self.batch_size = self.configuration.batch_size

        self.embedding_dim = self.configuration.embedding_dim
        self.hidden_dim = self.configuration.hidden_dim
        self.output_dim = self.configuration.output_dim
        self.Torch_MODEL_NAME = self.configuration.Torch_MODEL_NAME

        self.MODEL_PATH = os.path.join(self.project_path, self.configuration.MODEL_PATH)
        self.DEV_PATH = os.path.join(self.project_path, self.configuration.DEV_PATH)

    def main(self):
        # 获取device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 创建文件夹
        if os.path.exists(self.MODEL_PATH) is False:
            os.makedirs(self.MODEL_PATH)

        # audio_path/label
        data = pd.read_csv(self.DEV_PATH)
        audio_path = list(data['audio_path'])
        label = list(data['label'])

        # 处理器实例
        pro = Processor()
        X = []
        y = []
        for index in range(data.shape[0]):
            X.append(pro.input_x(audio_path=audio_path[index]))
            y.append(pro.input_y(label=label[index]))

        # 定义网络
        encoder = net.Encoder(embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim)
        decoder = net.Decoder(output_dim=self.output_dim, embedding_dim=self.embedding_dim,
                              hidden_dim=self.hidden_dim)
        network = net.Net(encoder=encoder, decoder=decoder, device=device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = Adam(network.parameters(), lr=0.001)
        model = Model(dataset=data)

        count = 0
        loss_min_value = 1e10
        for epoch in range(self.epoch):
            batch = pro.get_batch(X, y, self.batch_size)
            while True:
                try:
                    # x_train: (batch_size, audio_en_size, emb_size) .eg(4, 901,20)
                    # y_train: (batch_size, audio_cn_size) .eg(4, 39)
                    x_train, y_train = next(batch)

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

                    if loss < loss_min_value:
                        loss_min_value = loss
                        print('loss: %f' % loss)
                        model.save_model(network=network, path=self.MODEL_PATH, name=self.Torch_MODEL_NAME)
                        print("step %d, best lowest_loss %g" % (count, loss_min_value))
                    print(str(count))

                    count += 1
                except Exception as StopIteration:
                    print('StopIteration: ', StopIteration)
                    break
        print(count)


class Transformer(object):
    def __init__(self):
        TYPE = 'transformer'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        configuration = Constant(type=TYPE).get_configuration()
        self.project_path = Constant(type=TYPE).get_project_path()

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
        self.DEV_PATH = self.configuration.DEV_PATH

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

        # 数据获取辅助类
        # audio_path/label
        data = pd.read_csv(os.path.join(self.project_path, self.DEV_PATH))
        x_train = list(data['audio_path'])
        x_val = list(data['audio_path'])
        y_train = list(data['label'])
        y_val = list(data['label'])

        y_train = [{'label': Processor().input_y(label=i)} for i in y_train]
        y_val = [{'label': Processor().input_y(label=i)} for i in y_val]
        x_train = [{'audio_path': i} for i in x_train]
        x_val = [{'audio_path': i} for i in x_val]
        train = list(zip(x_train, y_train))
        val = list(zip(x_val, y_val))

        train = [dict(sample[0], **sample[1]) for sample in train]
        val = [dict(sample[0], **sample[1]) for sample in val]

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
    # Seq2Seq().main()
    Transformer().main()
