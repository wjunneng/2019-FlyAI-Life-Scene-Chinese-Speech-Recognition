# -*- coding: utf-8 -*
import numpy
import os
import torch

from demo.configuration.configuration import Configuration

# 动态加载类和函数.
__import__('net', fromlist=["Net"])

# 获取device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(object):
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
        print('模型路径: %s' % os.path.join(Configuration.MODEL_PATH, Configuration.Torch_MODEL_NAME))
        # 加载网络
        network = torch.load(os.path.join(Configuration.MODEL_PATH, Configuration.Torch_MODEL_NAME))
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
