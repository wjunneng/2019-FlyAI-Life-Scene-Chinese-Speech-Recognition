# -*- coding: utf-8 -*
import argparse
import torch

import pandas as pd
import os
from torch.optim import Adam
from torch import nn

from demos.models.net import Decoder, Encoder, Net
from demos.models.model import Model
from demos.demos import processor
from configurations.constant import Constant


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
        pro = processor.Processor()
        X = []
        y = []
        for index in range(data.shape[0]):
            X.append(pro.input_x(audio_path=audio_path[index]))
            y.append(pro.input_y(label=label[index]))

        # 定义网络
        encoder = Encoder(embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim)
        decoder = Decoder(output_dim=self.output_dim, embedding_dim=self.embedding_dim,
                          hidden_dim=self.hidden_dim)
        network = Net(encoder=encoder, decoder=decoder, device=device)
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


if __name__ == '__main__':
    Seq2Seq().main()
