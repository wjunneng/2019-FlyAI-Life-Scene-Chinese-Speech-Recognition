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
residual_dropout_rate = 0.5
# 是否共享编码器词嵌入的权重
share_embedding = True
# 指定批大小 [batch:16->Global Step:1280]
batch_size = 24
# 热身步数
warmup_steps = 12000
# 学习率因子
lr_factor = 1.0
# 梯度累计步数
accu_grads_steps = 8
# 输入特征维度
input_size = 40

vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
# 需要时刻注意是否更新
vocab_size = 3863