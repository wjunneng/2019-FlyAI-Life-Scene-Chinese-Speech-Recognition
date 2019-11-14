import numpy as np
import wave
import librosa

from configuration.constant import Constant


class Features(object):
    def __init__(self, wav_path, type):
        configuration = Constant(type=type).get_configuration()

        self.wav_path = wav_path
        self.win_length = configuration.win_length
        self.window_stride = configuration.window_stride
        self.window_size = configuration.window_size
        self.sample_rate = configuration.sample_rate
        self.window = configuration.window
        self.max_audio_len = configuration.max_audio_len
        self.embedding_dim = configuration.embedding_dim

        self.n_fft = int(self.sample_rate * self.window_size)
        self.hop_length = int(self.sample_rate * self.window_stride)

    @staticmethod
    def load_audio(wav_path, normalize=True):
        """
        加载 wav 文件
        :param normalize:
        :return:
        """

        with wave.open(wav_path) as wav_file:
            wav_data = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype='int32')
            wav_data = wav_data.astype("float")
        # 归一化
        if normalize is True:
            wav_data = (wav_data - wav_data.mean()) / wav_data.std()

        return wav_data

    def method_1(self, normalize=True):
        """
        方法一： 短时傅里叶变换[stft]
        :return:
        """
        spec = None
        try:
            wav = Features.load_audio(self.wav_path)
            D = librosa.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length,
                             win_length=self.win_length, window=self.window)

            spec, phase = librosa.magphase(D)
            spec = np.log1p(spec)

            spec = spec.transpose((1, 0))

            if normalize:
                spec = (spec - spec.mean()) / spec.std()
        except Exception as e:
            print('spec error %s' % e)

        try:
            if len(spec) >= self.max_audio_len:
                spec = spec[:self.max_audio_len]
                origanal_len = self.max_audio_len
            else:
                origanal_len = len(spec)
                spec = np.concatenate(
                    (spec, np.zeros([self.max_audio_len - origanal_len, self.embedding_dim])), 0
                )

            # 最后一行元素为句子实际长度
            spec = np.concatenate(
                (spec, np.array([origanal_len for _ in range(self.embedding_dim)]).reshape(
                    [1, self.embedding_dim])))
        except Exception as e:
            print('conc error %s' % e)

        return spec

    def method_2(self, normalize=True):
        """
        方法二： 获取梅尔倒谱系数
        :return:
        """
        mfcc = None
        try:
            wav, sr = librosa.load(self.wav_path, mono=True)
            mfcc = librosa.feature.mfcc(wav, sr, hop_length=int(self.window_stride * sr),
                                        n_fft=int(self.window_size * sr))
            mfcc = mfcc.transpose((1, 0))

            if normalize:
                mfcc = (mfcc - mfcc.mean()) / mfcc.std()
        except Exception as e:
            print('mfcc error %s' % e)

        try:
            if len(mfcc) >= self.max_audio_len:
                mfcc = mfcc[:self.max_audio_len]
                origanal_len = self.max_audio_len
            else:
                origanal_len = len(mfcc)
                mfcc = np.concatenate(
                    (mfcc, np.zeros([self.max_audio_len - origanal_len, self.embedding_dim])), 0)

            # 最后一行元素为句子实际长度
            mfcc = np.concatenate(
                (mfcc, np.array([origanal_len for _ in range(self.embedding_dim)]).reshape(
                    [1, self.embedding_dim])))
        except Exception as e:
            print('conc error %s' % e)

        return mfcc
