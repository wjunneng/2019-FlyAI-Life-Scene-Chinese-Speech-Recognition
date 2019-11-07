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

    datas = [
        # 以下列出列夫 托尔斯泰所着小说 战争与和平 中的人物 括号给出其首次出现章节
        {'audio_path': "/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/data/input/"
                       "wav/common_voice_zh-CN_18531674.wav"},
        # 从来都不是停止练习的借口
        {'audio_path': "/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/data/input/"
                       "wav/common_voice_zh-CN_18536155.wav"},
        # 比如生活的意义 上帝 真理等
        {'audio_path': "/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/data/input/"
                       "wav/common_voice_zh-CN_18536460.wav"},
        # 同时 他还担任过政治作战学校副教授 中国文化学院教授等
        {'audio_path': "/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/data/input/"
                       "wav/common_voice_zh-CN_18536592.wav"},
        # 保罗斯普尔是位于美国亚利桑那州科奇斯县的一个非建制地区
        {'audio_path': "/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/data/input/"
                       "wav/common_voice_zh-CN_18547568.wav"},
        # 武定州 中国唐朝时设置的州
        {'audio_path': "/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/data/input/"
                       "wav/common_voice_zh-CN_18567452.wav"},
        # 鲍曼普莱斯是位于美国加利福尼亚州门多西诺县的一个非建制地区
        {'audio_path': "/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/data/input/"
                       "wav/common_voice_zh-CN_18571522.wav"},
        # 明朝政治人物
        {'audio_path': "/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/data/input/"
                       "wav/common_voice_zh-CN_18571525.wav"},
        # 谷德刚 毕业于实践大学
        {'audio_path': "/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/data/input/"
                       "wav/common_voice_zh-CN_18571604.wav"},
        # 阿尔利河畔普拉人口变化图示
        {'audio_path': "/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/data/input/"
                       "wav/common_voice_zh-CN_18585207.wav"},
    ]

    predict(datas)
