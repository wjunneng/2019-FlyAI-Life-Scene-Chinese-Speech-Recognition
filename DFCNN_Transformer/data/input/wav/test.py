import wave
import os

# ------------------- aishell-2
# _wave_params(nchannels=1, sampwidth=2, framerate=16000, nframes=68496, comptype='NONE', compname='not compressed')
# with wave.open(
#         '/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/DFCNN_Transformer/data/input/wav/BAC009S0724W0121.wav',
#         'rb') as file:
#     print(file.getparams())


files = os.listdir(
    '/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/DFCNN_Transformer/data/input/wav')

for file in files:
    if file.endswith('.wav'):
        with wave.open(file) as wav_file:
            print('file: {} / params: {}'.format(file, wav_file.getparams()))
