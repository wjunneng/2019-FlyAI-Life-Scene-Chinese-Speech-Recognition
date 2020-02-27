# -*- coding: utf-8 -*-
"""
实现模型的预测
"""
from flyai.dataset import Dataset
from Seq2seq_Transformer.model import Model

data = Dataset()
model = Model(data)
# p = model.predict(audio_path='wav/common_voice_zh-CN_18772660.wav')

# wav/common_voice_zh-CN_18772660.wav,越南广播电台
# wav/common_voice_zh-CN_18585358.wav,人民民主独立和社会主义组织是冈比亚的激进社会主义政党
# wav/common_voice_zh-CN_18594110.wav,此站的月台间隙较大
# wav/common_voice_zh-CN_18813047.wav,这些小说主要是描绘新加坡人们于国家独立后的生活的点滴
# wav/common_voice_zh-CN_18626711.wav,阿曼控制了桑给巴尔岛和斯瓦希里海岸 成为重要的海上贸易国

p = model.predict_all([{'audio_path': 'wav/common_voice_zh-CN_18772660.wav'},
                       {'audio_path': 'wav/common_voice_zh-CN_18585358.wav'},
                       {'audio_path': 'wav/common_voice_zh-CN_18594110.wav'},
                       {'audio_path': 'wav/common_voice_zh-CN_18813047.wav'},
                       {'audio_path': 'wav/common_voice_zh-CN_18626711.wav'}])

print(p)
