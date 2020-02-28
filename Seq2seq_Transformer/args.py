import os
import sys

os.chdir(sys.path[0])
# -* model arguments *-
# 模型迭代次数
total_epochs = 60
# 模型维度
model_size = 320
# 注意力机制头数
n_heads = 4
# 编码器层数
num_enc_blocks = 6
# 解码器层数
num_dec_blocks = 6
# 残差连接丢弃率
residual_dropout_rate = 0.3
# 是否共享编码器词嵌入的权重
share_embedding = True
# 指定批大小 [batch:16->Global Step:1280]
batch_size = 4
# 热身步数
warmup_steps = 12000
# 学习率因子
lr_factor = 0.0005
# 梯度累计步数
accu_grads_steps = 8
# 输入特征维度
input_size = 64

vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
# 需要时刻注意是否更新
vocab_size = 766

# -----------ARGS---------------------
data_dir = os.path.join(os.getcwd(), "data")
input_dir = os.path.join(data_dir, 'input')
output_dir = os.path.join(data_dir, 'output')
data_model_dir = os.path.join(data_dir, 'model')
dev_csv_path = os.path.join(input_dir, 'dev.csv')
words_json_path = os.path.join(input_dir, 'words.json')
vocab_txt_path = os.path.join(input_dir, 'vocab.txt')

