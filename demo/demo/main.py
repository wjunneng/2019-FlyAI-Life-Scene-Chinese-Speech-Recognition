# -*- coding: utf-8 -*
import argparse
import torch
from demo.utils import processor
import pandas as pd
import os
from torch.optim import Adam
from torch import nn

from demo.models.net import Decoder, Encoder, Net
from demo.models.model import Model
from demo.configuration.configuration import Configuration


def main():
    # 超参
    parser = argparse.ArgumentParser()
    # 训练轮数
    parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
    # 训练批次
    parser.add_argument("-b", "--BATCH", default=4, type=int, help="batch size")
    # 获取超参数
    args = parser.parse_args()

    # 获取device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建文件夹
    if os.path.exists(Configuration.MODEL_PATH) is False:
        os.makedirs(Configuration.MODEL_PATH)

    # audio_path/label
    data = pd.read_csv(Configuration.DEV_PATH)
    audio_path = list(data['audio_path'])
    label = list(data['label'])

    # 处理器实例
    pro = processor.Processor()
    X = []
    y = []
    for index in range(data.shape[0]):
        X.append(pro.input_x(audio_path=audio_path[index]))
        y.append(pro.input_y(label=label[index]))

    batch = pro.get_batch(X, y, args.BATCH)

    # 定义网络
    encoder = Encoder(embedding_dim=Configuration.embedding_dim, hidden_dim=Configuration.hidden_dim)
    decoder = Decoder(output_dim=Configuration.output_dim, embedding_dim=Configuration.embedding_dim,
                      hidden_dim=Configuration.hidden_dim)
    network = Net(encoder=encoder, decoder=decoder, device=device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(network.parameters())
    model = Model()

    count = 0
    loss_min_value = 1e10
    while True:
        try:
            # x_train: (batch_size, audio_en_size, emb_size) .eg(4, 901,20)
            # y_train: (batch_size, audio_cn_size) .eg(4, 39)
            x_train, y_train = next(batch)

            # 每段语音序列长度 input_length: (batch_size) .eg(4)
            input_lengths = [int(x) for x in x_train[:, -1, 0].tolist()]
            input_lengths = torch.tensor(input_lengths).long()

            # 每段语音对应中文序列长度 y_lengths: (batch_size) .eg(4)
            y_lengths = y_train[:, -1].tolist()
            y_lengths = torch.tensor(y_lengths).long()

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
            outputs = network(src_seqs=x_train, src_lengths=input_lengths, trg_seqs=y_train).float()
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
                model.save_model(network=network, path=Configuration.MODEL_PATH, name=Configuration.Torch_MODEL_NAME)
                print("step %d, best lowest_loss %g" % (count, loss_min_value))
            print(str(count))

            count += 1
        except Exception as StopIteration:
            break
    print(count)


if __name__ == '__main__':
    main()
