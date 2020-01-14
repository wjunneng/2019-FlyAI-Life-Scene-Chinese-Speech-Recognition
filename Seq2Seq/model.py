# -*- coding: utf-8 -*
# -*- coding: utf-8 -*
import numpy
import os
import torch
from flyai.model.base import Base

__import__('net', fromlist=["Net"])

Torch_MODEL_NAME = "model.pkl"

# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

device = torch.device(device)


class Model(Base):
    def __init__(self, dataset):
        self.dataset = dataset

    def predict(self, path, name=Torch_MODEL_NAME, **data):
        network = torch.load(os.path.join(path, name))
        network = network.to(device)
        network.eval()
        # x_data shape: (batch,sen_len,embedding)
        x_data = self.dataset.predict_data(**data)
        # 因为输入batch为1，取第0个元素。 最后一行 的所有数均为句子实际长度
        length = [int(x_data[0, -1, 0])]
        # 除去长度信息
        x_data = x_data[:, :-1, :]
        x_data = torch.from_numpy(x_data)
        x_data = x_data.permute(1, 0, 2)
        x_data = x_data.float().to(device)

        outputs, _ = network.predict(x_data, length)
        outputs = outputs.squeeze(1).cpu().detach().numpy()
        output_words = self.dataset.to_categorys(outputs)

        report = ''
        for i in output_words:
            report = report + i
        report = report.strip('~“”')
        return report

    def predict_all(self, datas):
        network = torch.load(os.path.join(MODEL_PATH, Torch_MODEL_NAME))
        network = network.to(device)
        network.eval()

        prediction = []
        for data in datas:
            # x_data shape: (batch,sen_len,embedding)
            x_data = self.dataset.predict_data(**data)
            # 因为输入batch为1，取第0个元素。 最后一行 的所有数均为句子实际长度
            length = [int(x_data[0, -1, 0])]
            # 除去长度信息
            x_data = x_data[:, :-1, :]
            x_data = torch.from_numpy(x_data)
            x_data = x_data.permute(1, 0, 2)
            x_data = x_data.float().to(device)

            outputs, _ = network.predict(x_data, length)
            outputs = outputs.squeeze(1).cpu().detach().numpy()
            output_words = self.dataset.to_categorys(outputs)

            report = ''
            for i in output_words:
                report = report + i
            report = report.strip()
            prediction.append(report)
        return prediction


