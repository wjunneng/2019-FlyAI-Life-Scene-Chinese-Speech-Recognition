# -*- coding: utf-8 -*-
"""
实现模型的预测
"""
from flyai.dataset import Dataset

from Seq2Seq.model import Model

data = Dataset()
model = Model(data)
# p = model.predict(
#     audio_path='/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/Seq2Seq/data/input/wav/common_voice_zh-CN_18585358.wav')

p = model.predict_all(
    [{'audio_path': '/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/Seq2Seq/data/input/wav/common_voice_zh-CN_18585358.wav'},
     {'audio_path': '/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/Seq2Seq/data/input/wav/common_voice_zh-CN_18594110.wav'},
     {'audio_path': '/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/Seq2Seq/data/input/wav/common_voice_zh-CN_18813047.wav'},
     {'audio_path': '/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/Seq2Seq/data/input/wav/common_voice_zh-CN_18626711.wav'},
     {'audio_path': '/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/Seq2Seq/data/input/wav/common_voice_zh-CN_18817087.wav'},
     {'audio_path': '/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/Seq2Seq/data/input/wav/common_voice_zh-CN_18813359.wav'}])

print(p)
