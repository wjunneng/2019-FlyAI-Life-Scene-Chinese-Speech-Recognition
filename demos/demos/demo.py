import torch
import argparse
import json
import os
import random
from shutil import copyfile
from demos.models.net_1 import Transformer
import numpy as np
from utils.util import Util
from configurations.constant import Constant


def parse_args():
    parser = argparse.ArgumentParser("End-to-End Automatic Speech Recognition Decoding.")
    # decode
    parser.add_argument('--beam_size', default=5, type=int, help='Beam size')
    parser.add_argument('--nbest', default=5, type=int, help='Nbest size')
    parser.add_argument('--decode_max_len', default=70, type=int,
                        help='Max output length. If ==0 (default), it uses a '
                             'end-detect function to automatically find maximum '
                             'hypothesis lengths')
    args = parser.parse_args()

    return args


def main(samples):
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    TYPE = 'transformer'
    configuration = Constant(type=TYPE).get_configuration()
    project_path = Constant(type=TYPE).get_project_path()
    WORDS_PATH = configuration.WORDS_PATH

    char_dict = {}
    char_dict_res = {}

    # 构建字典
    with open(os.path.join(project_path, WORDS_PATH)) as fin:
        words = json.loads(fin.read())
    words = list(words.keys())
    if TYPE == 'seq2seq':
        # 去除
        words = [" ", "<unk>"] + words
    elif TYPE == 'transformer':
        # 新增
        words = [configuration.PAD_FLAG,
                 configuration.UNK_FLAG,
                 configuration.SOS_FLAG,
                 configuration.EOS_FLAG,
                 configuration.SPACE_FLAG] + words

    for i, word in enumerate(words):
        char_dict[word] = i
        char_dict_res[i] = word
    print('vocab len: %d' % len(words))

    char_list = char_dict_res

    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model.eval()

    samples = random.sample(samples, 10)

    results = []

    for i, sample in enumerate(samples):
        wave = sample['audio_path']
        # trn = sample['trn']

        feature = Util.extract_feature(input_file=wave, feature='fbank', dim=configuration.input_dim, cmvn=True,
                                       sample_rate=configuration.sample_rate)
        feature = Util.build_LFR_features(feature, m=configuration.LFR_m, n=configuration.LFR_n)
        # feature = np.expand_dims(feature, axis=0)
        input = torch.from_numpy(feature).to(device)
        input_length = [input[0].shape[0]]
        input_length = torch.from_numpy(np.array(input_length)).to(device)
        print('start')
        nbest_hyps = model.recognize(input, input_length, char_list, args)
        out_list = []
        for hyp in nbest_hyps:
            out = hyp['yseq']
            out = [char_list[idx] for idx in out]
            out = ''.join(out)
            out_list.append(out)
        print('OUT_LIST: {}'.format(out_list))

        # gt = [char_list[idx] for idx in trn]
        # gt = ''.join(gt)
        # print('GT: {}\n'.format(gt))
        #
        # results.append({'out_list_{}'.format(i): out_list, 'gt_{}'.format(i): gt})

    with open('results.json', 'w') as file:
        json.dump(results, file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    datas = [
        # 以下列出列夫 托尔斯泰所着小说 战争与和平 中的人物 括号给出其首次出现章节
        {'audio_path': "/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/data/input/"
                       "wav/common_voice_zh-CN_18531674.wav"},
        # 从来都不是停止练习的借口
        {'audio_path': "/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/data/input/"
                       "wav/common_voice_zh-CN_18536155.wav"},
        # 比如生活的意义 上帝 真理等
        {'audio_path': "/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/data/input/"
                       "wav/common_voice_zh-CN_18536460.wav"},
        # 同时 他还担任过政治作战学校副教授 中国文化学院教授等
        {'audio_path': "/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/data/input/"
                       "wav/common_voice_zh-CN_18536592.wav"},
        # 保罗斯普尔是位于美国亚利桑那州科奇斯县的一个非建制地区
        {'audio_path': "/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/data/input/"
                       "wav/common_voice_zh-CN_18547568.wav"},
        # 武定州 中国唐朝时设置的州
        {'audio_path': "/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/data/input/"
                       "wav/common_voice_zh-CN_18567452.wav"},
        # 鲍曼普莱斯是位于美国加利福尼亚州门多西诺县的一个非建制地区
        {'audio_path': "/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/data/input/"
                       "wav/common_voice_zh-CN_18571522.wav"},
        # 明朝政治人物
        {'audio_path': "/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/data/input/"
                       "wav/common_voice_zh-CN_18571525.wav"},
        # 谷德刚 毕业于实践大学
        {'audio_path': "/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/data/input/"
                       "wav/common_voice_zh-CN_18571604.wav"},
        # 阿尔利河畔普拉人口变化图示
        {'audio_path': "/home/wjunneng/Ubuntu/2019-FlyAI-Life-Scene-Chinese-Speech-Recognition/data/input/"
                       "wav/common_voice_zh-CN_18585207.wav"},
    ]

    main(datas)
