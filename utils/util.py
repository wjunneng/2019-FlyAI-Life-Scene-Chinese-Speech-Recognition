# -*- coding: utf-8 -*
import os
import codecs
import pickle
import torch
import logging
import librosa
import random
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class Util(object):
    @staticmethod
    def write_pkl(path, data):
        """
        保存pkl模型
        :param path:
        :param data:
        :return:
        """
        with open(path, 'wb') as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_pkl(path):
        """
        加载pkl模型
        :param path:
        :return:
        """
        with open(path, 'rb') as file:
            data = pickle.load(file)

        return data

    @staticmethod
    def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, is_best):
        state = {'epoch': epoch,
                 'epochs_since_improvement': epochs_since_improvement,
                 'loss': loss,
                 'model': model,
                 'optimizer': optimizer}

        filename = 'checkpoint.tar'
        torch.save(state, filename)
        # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
        if is_best:
            torch.save(state, 'BEST_checkpoint.tar')

    @staticmethod
    def get_logger():
        logger = logging.getLogger()
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        return logger

    @staticmethod
    def normalize(yt):
        """
        # [-0.5, 0.5]
        :return:
        """
        yt_max = np.max(yt)
        yt_min = np.min(yt)
        a = 1.0 / (yt_max - yt_min)
        b = -(yt_max + yt_min) / (2 * (yt_max - yt_min))

        yt = yt * a + b
        return yt

    @staticmethod
    def extract_feature(input_file, sample_rate, feature='fbank', dim=80, cmvn=True, delta=False, delta_delta=False,
                        window_size=25, stride=10, save_feature=None):
        """
        # Acoustic Feature Extraction
        # Parameters
        #     - input file  : str, audio file path
        #     - feature     : str, fbank or mfcc
        #     - dim         : int, dimension of feature
        #     - cmvn        : bool, apply CMVN on feature
        #     - window_size : int, window size for FFT (ms)
        #     - stride      : int, window stride for FFT
        #     - save_feature: str, if given, store feature to the path and return len(feature)
        # Return
        #     acoustic features with shape (time step, dim)
        :param input_file:
        :param sample_rate:
        :param feature:
        :param dim:
        :param cmvn:
        :param delta:
        :param delta_delta:
        :param window_size:
        :param stride:
        :param save_feature:
        :return:
        """
        y, sr = librosa.load(input_file, sr=sample_rate)
        yt, _ = librosa.effects.trim(y, top_db=20)
        yt = Util.normalize(yt)
        ws = int(sr * 0.001 * window_size)
        st = int(sr * 0.001 * stride)
        if feature == 'fbank':  # log-scaled
            feat = librosa.feature.melspectrogram(y=yt, sr=sr, n_mels=dim,
                                                  n_fft=ws, hop_length=st)
            feat = np.log(feat + 1e-6)
        elif feature == 'mfcc':
            feat = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=dim, n_mels=26,
                                        n_fft=ws, hop_length=st)
            feat[0] = librosa.feature.rmse(yt, hop_length=st, frame_length=ws)

        else:
            raise ValueError('Unsupported Acoustic Feature: ' + feature)

        feat = [feat]
        if delta:
            feat.append(librosa.feature.delta(feat[0]))

        if delta_delta:
            feat.append(librosa.feature.delta(feat[0], order=2))
        feat = np.concatenate(feat, axis=0)
        if cmvn:
            feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]
        if save_feature is not None:
            tmp = np.swapaxes(feat, 0, 1).astype('float32')
            np.save(save_feature, tmp)
            return len(tmp)
        else:
            return np.swapaxes(feat, 0, 1).astype('float32')

    @staticmethod
    def spec_augment(spec: np.ndarray, num_mask=2, freq_masking=0.15, time_masking=0.20, value=0):
        """
        # Source: https://www.kaggle.com/davids1992/specaugment-quick-implementation
        :param spec:
        :param num_mask:
        :param freq_masking:
        :param time_masking:
        :param value:
        :return:
        """
        spec = spec.copy()
        num_mask = random.randint(1, num_mask)
        for i in range(num_mask):
            all_freqs_num, all_frames_num = spec.shape
            freq_percentage = random.uniform(0.0, freq_masking)

            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[f0:f0 + num_freqs_to_mask, :] = value

            time_percentage = random.uniform(0.0, time_masking)

            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[:, t0:t0 + num_frames_to_mask] = value
        return spec

    @staticmethod
    def build_LFR_features(inputs, m, n):
        """
        Actually, this implements stacking frames and skipping frames.
        if m = 1 and n = 1, just return the origin features.
        if m = 1 and n > 1, it works like skipping.
        if m > 1 and n = 1, it works like stacking but only support right frames.
        if m > 1 and n > 1, it works like LFR.
        Args:
            inputs_batch: inputs is T x D np.ndarray
            m: number of frames to stack
            n: number of frames to skip
        """
        # LFR_inputs_batch = []
        # for inputs in inputs_batch:
        LFR_inputs = []
        T = inputs.shape[0]
        T_lfr = int(np.ceil(T / n))
        for i in range(T_lfr):
            if m <= T - i * n:
                LFR_inputs.append(np.hstack(inputs[i * n:i * n + m]))
            else:  # process last LFR frame
                num_padding = m - (T - i * n)
                frame = np.hstack(inputs[i * n:])
                for _ in range(num_padding):
                    frame = np.hstack((frame, inputs[-1]))
                LFR_inputs.append(frame)
        return np.vstack(LFR_inputs)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AiShellDataset(Dataset):
    def __init__(self, samples, args, split):
        self.args = args
        self.samples = samples
        print('loading {}{} samples...'.format(len(self.samples), split))

    def __getitem__(self, i):
        sample = self.samples[i]
        wave = os.path.join('/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/data/input',
                            sample['audio_path'])
        trn = sample['label']

        feature = Util.extract_feature(input_file=wave, feature='fbank', dim=self.args.d_input, cmvn=True,
                                       sample_rate=self.args.sample_rate)
        # zero mean and unit variance
        feature = (feature - feature.mean()) / feature.std()
        feature = Util.spec_augment(feature)
        feature = Util.build_LFR_features(feature, m=self.args.LFR_m, n=self.args.LFR_n)

        return torch.from_numpy(feature), torch.from_numpy(np.array(trn))

    def __len__(self):
        return len(self.samples)


class PreProcess(object):
    def __init__(self, configuration):
        self.configuration = configuration
        # aishell_speech/libri_speech/life_speech
        self.datasource_type = configuration['datasource_type']
        # dev/train/test文件夹保存的路径 数据集的路径
        self.path = configuration[self.datasource_type]['path']
        # label保存的路径
        self.audio_label_path = configuration[self.datasource_type]['audio_label_path']
        # audio路径与label之间的分隔符
        self.audio_label_splitter = configuration[self.datasource_type]['audio_label_splitter']

        # 字典
        self.vocab_to_index = {configuration['token']['PAD_FLAG']: configuration['token']['PAD'],
                               configuration['token']['UNK_FLAG']: configuration['token']['UNK'],
                               configuration['token']['SOS_FLAG']: configuration['token']['SOS'],
                               configuration['token']['EOS_FLAG']: configuration['token']['EOS'],
                               configuration['token']['SPACE_FLAG']: configuration['token']['SPACE']}

    def get_data(self, data, type='train'):
        """
        获取数据
        :param data: {'train': samples, 'dev': samples, 'test': test}
        :param type:
        :return:
        """
        # key保存audio路径, value保存label
        audio_label = {}
        # token_index 和 wav文件的绝对路径
        samples = []

        # 遍历文件
        with codecs.open(os.path.join(self.path, self.audio_label_path)) as file:
            for line in dict.fromkeys(file.readlines(), True):
                audio_label_list = line.split(self.audio_label_splitter)
                audio_label[audio_label_list[0]] = self.audio_label_splitter.join(audio_label_list[1:]).strip('\n')

        if self.datasource_type == 'aishell_speech':
            # 遍历train dev test文件夹
            floder = os.path.join(self.path, 'wav', type)
        else:
            floder = os.path.join(self.path, type)
        assert (os.path.isdir(floder) is True)

        print('len:', len(os.listdir((floder))))
        for d in tqdm(os.listdir(floder)):
            dirs = os.path.join(floder, d)
            if os.path.isdir(dirs):
                files = [file for file in os.listdir(dirs) if file.endswith('.wav')]

                for file in dict.fromkeys(files, True):
                    file_path = os.path.join(dirs, file)
                    self.vocab_to_index, samples = PreProcess.add_token(vocab_to_index=self.vocab_to_index,
                                                                        samples=samples,
                                                                        eos_flag=self.configuration['token'][
                                                                            'EOS_FLAG'],
                                                                        audio_label=audio_label, file=file,
                                                                        file_path=file_path)

            elif os.path.isfile(dirs) and dirs.endswith('.wav'):
                self.vocab_to_index, samples = PreProcess.add_token(vocab_to_index=self.vocab_to_index, samples=samples,
                                                                    eos_flag=self.configuration['token']['EOS_FLAG'],
                                                                    audio_label=audio_label, file=d, file_path=dirs)
        data[type] = samples

        return data

    def get_index_to_vocab(self):
        """
        返回 index_to_vocab 字典
        :return:
        """
        # 反字典
        index_to_vocab = dict(zip(self.vocab_to_index.values(), self.vocab_to_index.keys()))

        return index_to_vocab

    def get_vocab_to_index(self):
        """
        返回 vocab_to_index 字典
        :return:
        """
        # 字典
        return self.vocab_to_index

    def save_pkl(self, types):
        """
        保存数据为 pkl 模型
        :param types:
        :return:
        """
        data = {}
        for type in types:
            data = self.get_data(type=type, data=data)
            print('%s' % type)
            print(len(data[type]))

        data['VOCAB'] = self.get_vocab_to_index()
        data['IVOCAB'] = self.get_index_to_vocab()
        print('VOCAB.size: %d' % len(self.get_vocab_to_index()))
        Util.write_pkl(data=data, path=os.path.join(self.configuration[self.configuration['datasource_type']]['path'],
                                                    self.configuration[self.configuration['datasource_type']][
                                                        'audio_index_pkl_path']))

        print('save success!')

    @staticmethod
    def add_token(vocab_to_index, samples, eos_flag, audio_label, file, file_path):
        """
        添加 token:<sos>
        :param vocab_to_index: key为vocab, value为index的字典
        :param samples: 包含token_index与wav_path
        :param eos_flag: <eos>
        :param audio_label: key为filenames, value为序列
        :param file: filename
        :param file_path: filepath
        :return:
        """
        token_to_index = []

        # 文件名
        if file in audio_label or file.split('.')[0] in audio_label:
            if file in audio_label:
                value = audio_label[file]
            elif file.split('.')[0] in audio_label:
                value = audio_label[file.split('.')[0]]

            # 添加<eos>标记
            value = list(value.strip()) + [eos_flag]

            # 遍历tokens
            for token in dict.fromkeys(value, True):
                if token not in vocab_to_index:
                    vocab_to_index[token] = len(vocab_to_index)

                token_to_index.append(vocab_to_index[token])
            samples.append({'token_index': token_to_index, 'wav_path': file_path})

        return vocab_to_index, samples

# if __name__ == '__main__':
#     configuration = Constant(type='seq2seq').get_configuration()
#
#     pre_process = PreProcess(configuration=configuration)
#     pre_process.save_pkl(types=['train', 'test', 'dev'])
