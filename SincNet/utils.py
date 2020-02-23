# -*- coding:utf-8 -*-
import os
import sys
import soundfile
import numpy as np

os.chdir(sys.path[0])


class Util(object):
    def __init__(self):
        pass

    @staticmethod
    def read_list(file):
        with open(file, 'r') as f:
            return [line.rstrip() for line in f.readlines()]

    @staticmethod
    def remove_start_and_end_silences(wav_file_list: list):
        """
        根据*.wrd文件中报告的信息删除开始和结束静音，并标准化每个句子的幅度
        :param wav_file_list:
        :return:
        """
        for wav_file in wav_file_list:
            [signal, fs] = soundfile.read(wav_file)
            signal = signal.astype(np.float)
            signal = signal / np.max(np.abs(signal))

            wav_file = wav_file.replace('.wav', '.wrd')
            wav_sig = Util.read_list(wav_file)

            # 去除开始和结束静音
            begin_signal = int(wav_sig[0].split(' ')[0])
            end_signal = int(wav_sig[-1].split(' ')[1])
            signal = signal[begin_signal, end_signal]

            # 标准化


if __name__ == '__main__':
    file = ['/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/SincNet/data/input/wav/common_voice_zh-CN_18531674.wav']
    Util.remove_start_and_end_silences(file)
