# -*- coding: utf-8 -*
"""
实现模型的预测
"""
from demo.models import model
from demo.utils import processor
import numpy as np


def predict(data_paths):
    """
    wav 数据路径
    :param datas:
    :return:
    """
    inputs = []
    for data_path in data_paths:
        pro = processor.Processor()

        inputs.append(pro.input_x(audio_path=data_path['audio_path']))

    # 预测值
    prediction = model.Model().predict_all(np.array(inputs))
    print(prediction)


if __name__ == '__main__':
    datas = [{
        'audio_path': "/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition"
                      "/data/input/wav/common_voice_zh-CN_18531674.wav"}]

    predict(datas)
