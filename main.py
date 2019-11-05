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
from configuration.configuration import Configuration

# 超参
parser = argparse.ArgumentParser()
# 训练轮数
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
# 训练批次
parser.add_argument("-b", "--BATCH", default=16, type=int, help="batch size")
args = parser.parse_args()

# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

# 数据获取辅助类
data = Dataset(epochs=args.EPOCHS, batch=args.BATCH, val_batch=args.BATCH)
# (embedding_dim, hidden_dim) eg.(20, 64)
en = Encoder(Configuration.embedding_dim, Configuration.hidden_dim)
# (output_dim, embedding_dim, hidden_dim) eg.(3507, 20, 64)
de = Decoder(Configuration.output_dim, Configuration.embedding_dim, Configuration.hidden_dim)
# 定义网络
network = Net(en, de, device)
# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 优化器
optimizer = Adam(network.parameters())
model = Model(data)
iteration = 0
lowest_loss = 10

# 得到训练和测试的数据
for i in range(data.get_step()):
    # 开启训练状态
    network.train()

    # 得到训练和测试的数据
    # X_train: shape:(batch, sen_len, embedding) eg.(16, 901, 20)
    # y_train: shape:(batch, 语音文字最大长度) eg.(16, 39)
    x_train, y_train = data.next_train_batch()
    # 读取数据; shape:(batch, sen_len, embedding)
    x_test, y_test = data.next_validation_batch()
    # batch_size eg.(16)
    batch_len = y_train.shape[0]
    # seq_len中的最后一行 标记当前seq_len中非零的长度（真实的seq_len） # 转换为int
    input_lengths = [int(x) for x in x_train[:, -1, 0].tolist()]
    # 每段文字长度 eg.(6, 即该段语音对应的中文长度为6)
    y_lengths = y_train[:, -1].tolist()

    # 除去长度信息 eg.(16, 900, 20)
    x_train = x_train[:, :-1, :]
    # 除去长度信息 eg.(16, 38)
    y_train = y_train[:, :-1]

    # shape:(batch,sen_len,embedding)
    x_train = torch.from_numpy(x_train).float().to(device)
    # shape:(batch,sen_len)
    y_train = torch.from_numpy(y_train).long().to(device)

    # 依据input_length 进行从大到小的排序
    seq_pairs = sorted(zip(x_train.contiguous(), y_train.contiguous(), input_lengths, y_lengths), key=lambda x: x[2],
                       reverse=True)
    # 皆为tuple类型
    x_train, y_train, input_lengths, y_lengths = zip(*seq_pairs)
    # permute：更换tensor的维度 x_train: eg.(900, 16, 20)
    x_train = torch.stack(x_train, dim=0).permute(1, 0, 2).contiguous()
    # y_train: eg.(38, 16)
    y_train = torch.stack(y_train, dim=0).permute(1, 0).contiguous()
    input_length_tensor = torch.tensor(input_lengths).long()
    y_lengths = torch.tensor(y_lengths).long()
    # x_train: eg.(900, 16, 20) input_lengths: eg.(16) y_train: eg.(38, 16)
    outputs = network(x_train, input_lengths, y_train)
    # 梯度清零
    optimizer.zero_grad()
    # output: eg.(38, 16, 3507)
    outputs = outputs.float()
    # outputs.view(-1, outputs.shape[2]): eg.(38*16, 3507)    y_train.view(-1): eg.(38*16)
    loss = loss_fn(outputs.view(-1, outputs.shape[2]), y_train.view(-1))

    # backward transmit loss
    loss.backward()
    # adjust parameters using Adam
    optimizer.step()
    print(loss)

    if loss < lowest_loss:
        lowest_loss = loss
        model.save_model(network, MODEL_PATH, overwrite=True)
        print("step %d, best lowest_loss %g" % (i, lowest_loss))
    print(str(i) + "/" + str(data.get_step()))
