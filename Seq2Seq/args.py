# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

# 模型类型
model_name = 'Seq2Seq'

# 日志路径
log_dir = os.path.join(os.getcwd(), 'data/log')

# Model parameters
input_dim = 80  # dimension of feature
window_size = 25  # window size for FFT (ms)
stride = 10  # window stride for FFT (ms)
hidden_size = 512
embedding_dim = 512
cmvn = True  # apply CMVN on feature
num_layers = 4
LFR_m = 4
LFR_n = 3
seed = 42  # 随机种子
