# -*- coding: utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
import torch
from flyai.model.base import Base

from Seq2Seq import args
from Seq2Seq.utils.util import SortedByCountsDict
from Seq2Seq.utils.util import Util
from Seq2Seq.net import Net

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Model(Base):
    def __init__(self, dataset):
        self.dataset = dataset
        self.args = args

        sbcd = SortedByCountsDict(dump_dir=self.args.vocab_dump_dir)
        self.vocab = sbcd.get_vocab()
        self.i_vocab = sbcd.get_i_vocab()
        self.model = Net.load_model(os.path.join(self.args.output_dir, self.args.model_path))
        self.model.eval()
        self.model.to(DEVICE)

    def predict(self, **data):
        with torch.no_grad():
            audio_path = self.dataset.predict_data(**data)[0]
            feature = Util.compute_fbank_from_file(file=audio_path,
                                                   feature_dim=self.args.input_dim)
            input = Util.build_LFR_features(feature, self.args.LFR_m, self.args.LFR_n)
            input = torch.from_numpy(input).float()
            input_length = torch.tensor([input.size(0)], dtype=torch.int)
            input = input.to(DEVICE)
            nbest_hyps = self.model.recognize(input=input, input_length=input_length, char_list=self.vocab, args=args)
            pred_label = nbest_hyps[0]['yseq'][1:-1]
            pred_res = ''.join([self.i_vocab[index] for index in pred_label])
            print("pred :", pred_res)

        return pred_res

    def predict_all(self, datas):
        labels = []
        for data in datas:
            predicts = self.predict(audio_path=data['audio_path'])

            labels.append(predicts)

        return labels
