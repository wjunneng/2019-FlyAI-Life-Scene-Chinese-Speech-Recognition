# -*- coding: utf-8 -*
import numpy
import os
import torch

from configurations.configuration import Configuration
from flyai.model.base import Base
from path import MODEL_PATH

# 动态加载类和函数.
__import__('net', fromlist=["Net"])

# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)


class Model(Base):
    def __init__(self, dataset):
        self.dataset = dataset

    def predict(self, path, name=Configuration.Torch_MODEL_NAME, **data):
        """
        预测单条数据
        :param path:
        :param name:
        :param data:
        :return:
        """
        # 加载网络
        network = torch.load(os.path.join(path, name))
        network = network.to(device)
        network.eval()

        # x_data shape: (batch, sen_len, embedding)
        x_data = self.dataset.predict_data(**data)
        # 因为输入batch为1，取第0个元素。 最后一行 的所有数均为句子实际长度
        length = [int(x_data[0, -1, 0])]
        # 除去长度信息
        x_data = x_data[:, :-1, :]
        x_data = torch.from_numpy(x_data)
        x_data = x_data.permute(1, 0, 2)
        x_data = x_data.float().to(device)

        # outputs: eg.(src_seqs, src_lengths, max_trg_len)
        outputs, _ = network.predict(src_seqs=x_data, src_lengths=length, max_trg_len=Configuration.max_tgt_len)
        outputs = outputs.squeeze(1).cpu().detach().numpy()
        output_words = self.dataset.to_categorys(outputs)

        report = ''
        for i in output_words:
            report = report + i
        report = report.strip('~“”')
        return report

    def predict_all(self, datas):
        """
        预测所有数据
        :param datas:
        :return:
        """
        print('模型路径: %s' % os.path.join(MODEL_PATH, Configuration.Torch_MODEL_NAME))
        # 加载网络
        network = torch.load(os.path.join(MODEL_PATH, Configuration.Torch_MODEL_NAME))
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

            outputs, _ = network.predict(src_seqs=x_data, src_lengths=length, max_trg_len=Configuration.max_tgt_len)
            outputs = outputs.squeeze(1).cpu().detach().numpy()
            output_words = self.dataset.to_categorys(outputs)

            report = ''
            for i in output_words:
                report = report + i
            report = report.strip()
            prediction.append(report)
        return prediction

    def batch_iter(self, x, y, batch_size=1):
        """
        生成批次数据
        :param x:
        :param y:
        :param batch_size:
        :return:
        """
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        indices = numpy.random.permutation(numpy.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

    def save_model(self, network, path, name=Configuration.Torch_MODEL_NAME, overwrite=False):
        """
        保存模型
        :param network:
        :param path:
        :param name:
        :param overwrite:
        :return:
        """
        super().save_model(network, path, name, overwrite)
        torch.save(network, os.path.join(path, name))

    def delete_file(self, path):
        """
        删除文件
        :param path:
        :return:
        """
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
