# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

EPOCHS = 20
BATCH = 5
model_name = 'DFCNN_Transformer'

# wav 路径
wav_dir = os.path.join(os.getcwd(), 'data/input')
# 日志路径
log_dir = os.path.join(os.getcwd(), 'data/log')
# 输出路径
# output_dir = os.path.join(os.getcwd(), 'data/output')
# 字典保存路径
vocab_dump_dir = os.path.join(os.getcwd(), 'data/vocab.pkl')
# dict.txt
dict_dir = os.path.join(os.getcwd(), 'data/dict.txt')
# hanzi.txt
hanzi_dir = os.path.join(os.getcwd(), 'data/hanzi.txt')
# mixdict.txt
mixdict_dir = os.path.join(os.getcwd(), 'data/mixdict.txt')

# 声学模型文件路径
AmModelFolder = os.path.join(os.getcwd(), 'data/output/am')
AmModelTensorBoard = os.path.join(os.getcwd(), 'data/output/am')
# 语言学模型文件路径
LmModelFolder = os.path.join(os.getcwd(), 'data/output/lm')
LmModelTensorboard = os.path.join(os.getcwd(), 'data/output/lm')

# 随机种子
seed = 42
# 是否打乱数据
shuffle = True

# 声学模型参数
am_batch_size = 5
am_epochs = 10

am_lr = 0.001
am_gpu_nums = 1
am_is_training = True
am_feature_dim = 200
am_feature_max_length = 1600  # 最大帧数

# 语音模型参数
lm_max_len = 48
lm_batch_size = 5
lm_epochs = 10
lm_feature_dim = 200
lm_num_heads = 8
lm_num_blocks = 6
lm_position_max_length = 100
lm_hidden_units = 512
lm_lr = 0.0003
lm_dropout_rate = 0.2
lm_is_training = True
lm_count = 5000

PAD = 0
SOS = 1
EOS = 2
PAD_FLAG = '<pad>'
SOS_FLAG = '<sos>'
EOS_FLAG = '</sos>'
