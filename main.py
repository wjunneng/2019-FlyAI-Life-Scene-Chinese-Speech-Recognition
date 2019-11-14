# -*- coding: utf-8 -*
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
import os

from net import Decoder, Encoder, Net
from path import MODEL_PATH, LOG_PATH
from flyai.dataset import Dataset
from model import Model
from configuration.constant import Constant


class Seq2seq(object):
    def __init__(self):
        configuration = Constant(type='seq2seq').get_configuration()
        self.embedding_dim = configuration.embedding_dim
        self.hidden_dim = configuration.hidden_dim
        self.output_dim = configuration.output_dim

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
        encoder = Encoder(embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim)
        decoder = Decoder(output_dim=self.output_dim, embedding_dim=self.embedding_dim,
                          hidden_dim=self.hidden_dim)
        network = Net(encoder=encoder, decoder=decoder, device=device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = Adam(network.parameters(), lr=0.001)
        model = Model(data)

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


class Tramsformer(object):
    def __init__(self):
        configuration = Constant(type='tramsformer').get_configuration()

    def main(self):
        pass


if __name__ == '__main__':
    Seq2seq().main()
