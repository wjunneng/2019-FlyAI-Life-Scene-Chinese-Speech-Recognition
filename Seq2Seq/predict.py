# -*- coding: utf-8 -*
"""
实现模型的预测
"""
from flyai.dataset import Dataset

from Seq2Seq.model import Model
from Seq2Seq.path import MODEL_PATH

data = Dataset()
model = Model(data)
p = model.predict(MODEL_PATH, upper='竹 杖 芒 鞋 轻 胜 马')
print(p)
