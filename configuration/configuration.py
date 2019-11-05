
class Configuration(object):
    # 音频序列最大长度
    max_audio_len = 900
    # 音频对应的中文序列最大长度
    max_tgt_len = 38
    # 嵌入的维度
    embedding_dim = 20
    # 隐藏层维度
    hidden_dim = 64
    # 输出的维度
    output_dim = 3507
    # 模型名称
    Torch_MODEL_NAME = "model.pkl"
