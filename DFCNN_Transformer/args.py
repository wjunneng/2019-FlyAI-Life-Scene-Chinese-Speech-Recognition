# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

# wav 路径
wav_dir = os.path.join(os.getcwd(), 'data/input')
# 日志路径
log_dir = os.path.join(os.getcwd(), 'data/log')
# 字典保存路径
vocab_dump_dir = os.path.join(os.getcwd(), 'data/vocab.pkl')
# 输出路径
output_dir = os.path.join(os.getcwd(), 'data/output')

EPOCHS = 10
BATCH = 10
model_name = 'Seq2Seq'
feature_type = 'mfcc'
# --------------------------------------------------Low Frame Rate (stacking and skipping frames)
# Low Frame Rate: number of frames to stack
LFR_m = 4
# Low Frame Rate: number of frames to skip
LFR_n = 3

# --------------------------------------------------encoder
# Dim of encoder input (before LFR)
d_input = 100
# Number of encoder stacks
n_layers_enc = 6
# Number of Multi Head Attention (MHA)
n_head = 8
# Dimension of key
d_k = 64
# Dimension of value
d_v = 64
# Dimension of model
d_model = 512
# Dimension of inner
d_inner = 2048
# Dropout rate
dropout = 0.1
# Positional Encoding max len
pe_maxlen = 5000

# --------------------------------------------------decoder
# Dim of decoder embedding
d_word_vec = 512
# Number of decoder stacks
n_layers_dec = 6
# share decoder embedding with decoder projection
tgt_emb_prj_weight_sharing = 1

# --------------------------------------------------Loss
# label smoothing
label_smoothing = 0.1

# --------------------------------------------------minibatch
# reshuffle the data at every epoch
shuffle = 1
# Batch frames. If this is not 0, batch size will make no sense
batch_frames = 0
# Batch size is reduced if the input sequence length > ML
maxlen_in = 800
# Batch size is reduced if the output sequence length > ML
maxlen_out = 150

# --------------------------------------------------optimizer
# learning rate
lr = 0.001
# tunable scalar multiply to learning rate
k = 0.2
# warmup steps
warmup_steps = 4000

# --------------------------------------------------Model parameters
# dimension of feature
input_dim = 128
# window size for FFT (ms)
window_size = 25
# window stride for FFT (ms)
stride = 10
hidden_size = 512
embedding_dim = 512
# apply CMVN on feature
cmvn = True
num_layers = 4
seed = 42  # 随机种子

# --------------------------------------------------Data parameters
IGNORE_ID = -1
sos_id = 0
eos_id = 1

# --------------------------------------------------Training parameters
# clip gradients at an absolute value of
grad_clip = 5.
# print training/validation stats  every __ batches
print_freq = 5

# --------------------------------------------------Prediction parameters
# Beam size
beam_size = 5
# Nbest size
nbest = 5
# Max output length. If ==0 (default), it uses a end-detect function to automatically find maximum hypothesis lengths
decode_max_len = 100
