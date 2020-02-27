# -*- coding: utf-8 -*
import os
import sys

os.chdir(sys.path[0])
from flyai.model.base import Base
import torch
from torch.utils.data import DataLoader

from Seq2seq_Transformer import args
from Seq2seq_Transformer.util import AudioDataset, Util
from Seq2seq_Transformer.module import Transformer, Recognizer

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Model(Base):
    def __init__(self, dataset):
        self.dataset = dataset
        self.args = args

    def predict(self, **data):
        audio_path = self.dataset.predict_data(**data)[0]

        # 定义评估模型
        eval_model = Transformer(input_size=self.args.input_size,
                                 vocab_size=self.args.vocab_size,
                                 d_model=self.args.model_size,
                                 n_heads=self.args.n_heads,
                                 d_ff=self.args.model_size * 4,
                                 num_enc_blocks=self.args.num_enc_blocks,
                                 num_dec_blocks=self.args.num_dec_blocks,
                                 residual_dropout_rate=0.0,
                                 share_embedding=self.args.share_embedding)

        if torch.cuda.is_available():
            eval_model.cuda()  # 将模型加载到GPU中

        # 将模型加载
        idx2unit = {}
        with open(self.args.vocab_txt_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                unit, idx = line.strip().split()
                idx2unit[int(idx)] = unit

        # 将模型加载
        dataset = AudioDataset(audios_list=[audio_path])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False,
                                collate_fn=Util.collate_fn)
        checkpoints = torch.load(os.path.join(self.args.data_model_dir, 'model.epoch.30.pt'))
        eval_model.load_state_dict(checkpoints)

        recognizer = Recognizer(eval_model, unit2char=idx2unit)

        print('Begin to decode test set!')
        for step, inputs in enumerate(dataloader):
            # 将输入加载到GPU中
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            preds = recognizer.recognize(inputs)

            return preds

    def predict_all(self, datas):
        labels = []
        for data in datas:
            predicts = self.predict(audio_path=data['audio_path'])

            labels.append(predicts)

        return labels
