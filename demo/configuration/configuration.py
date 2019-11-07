import os, sys


class Configuration(object):
    # 音频序列最大长度
    max_audio_len = 1000
    # 音频对应的中文序列最大长度
    max_tgt_len = 40
    # 嵌入的维度
    embedding_dim = 81
    # 隐藏层维度
    hidden_dim = 128
    # 输出的维度
    output_dim = 3507

    # 模型名称
    Torch_MODEL_NAME = "model.pkl"
    # 训练数据的路径
    DATA_PATH = os.path.join('/'.join(sys.path[0].split('/')[:-1]), 'data', 'input')
    # 模型保存的路径
    MODEL_PATH = os.path.join('/'.join(sys.path[0].split('/')[:-1]), 'data', 'output', 'model')
    # 训练log的输出路径
    LOG_PATH = os.path.join('/'.join(sys.path[0].split('/')[:-1]), 'data', 'output', 'logs')
    # Words.json路径
    WORDS_PATH = os.path.join(DATA_PATH, 'words.json')
    # dev.csv路径
    DEV_PATH = os.path.join(DATA_PATH, 'dev.csv')

    # mfcc 特征
    sample_rate = 16000
    window_size = 0.01
    window_stride = 0.01
    n_fft = int(sample_rate * window_size)
    win_length = n_fft
    hop_length = int(sample_rate * window_stride)
    window = "hamming"
