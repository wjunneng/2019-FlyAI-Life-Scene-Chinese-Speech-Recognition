import numpy as np
import wave
import librosa

from configuration.configuration import Configuration


class Features(object):
    def __init__(self, wav_path):
        self.wav_path = wav_path

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
            D = librosa.stft(wav, n_fft=Configuration.n_fft, hop_length=Configuration.hop_length,
                             win_length=Configuration.win_length, window=Configuration.window)

            spec, phase = librosa.magphase(D)
            spec = np.log1p(spec)

            spec = spec.transpose((1, 0))

            if normalize:
                spec = (spec - spec.mean()) / spec.std()
        except Exception as e:
            print('spec error %s' % e)

        try:
            if len(spec) >= Configuration.max_audio_len:
                spec = spec[:Configuration.max_audio_len]
                origanal_len = Configuration.max_audio_len
            else:
                origanal_len = len(spec)
                spec = np.concatenate(
                    (spec, np.zeros([Configuration.max_audio_len - origanal_len, Configuration.embedding_dim])), 0
                )

            # 最后一行元素为句子实际长度
            spec = np.concatenate(
                (spec, np.array([origanal_len for _ in range(Configuration.embedding_dim)]).reshape(
                    [1, Configuration.embedding_dim])))
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
            mfcc = librosa.feature.mfcc(wav, sr, hop_length=int(Configuration.window_stride * sr),
                                        n_fft=int(Configuration.window_size * sr))
            mfcc = mfcc.transpose((1, 0))

            if normalize:
                mfcc = (mfcc - mfcc.mean()) / mfcc.std()
        except Exception as e:
            print('mfcc error %s' % e)

        try:
            if len(mfcc) >= Configuration.max_audio_len:
                mfcc = mfcc[:Configuration.max_audio_len]
                origanal_len = Configuration.max_audio_len
            else:
                origanal_len = len(mfcc)
                mfcc = np.concatenate(
                    (mfcc, np.zeros([Configuration.max_audio_len - origanal_len, Configuration.embedding_dim])), 0)

            # 最后一行元素为句子实际长度
            mfcc = np.concatenate(
                (mfcc, np.array([origanal_len for _ in range(Configuration.embedding_dim)]).reshape(
                    [1, Configuration.embedding_dim])))
        except Exception as e:
            print('conc error %s' % e)

        return mfcc
