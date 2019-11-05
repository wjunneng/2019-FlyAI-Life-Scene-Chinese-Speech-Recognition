# -*- coding: utf-8 -*
"""
实现模型的预测
"""
from flyai.dataset import Dataset

from model import Model
from path import MODEL_PATH

data = Dataset()
model = Model(data)
# p = model.predict(MODEL_PATH, upper='竹 杖 芒 鞋 轻 胜 马')
# print(p)

# 越南广播电台
q = model.predict_all(
    datas=[{
        'audio_path': "/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition"
                      "/data/input/wav/common_voice_zh-CN_18531674.wav"}]
)

print(q)
