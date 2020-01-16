# -*- coding: utf-8 -*
import os
import sys

os.chdir(sys.path[0])
import torch
from flyai.model.base import Base

from Seq2Seq import args
from Seq2Seq.Utils.util import SortedByCountsDict
from Seq2Seq.Utils.util import Util

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Model(Base):
    def __init__(self, dataset):
        self.dataset = dataset
        self.args = args
        self.vocab = SortedByCountsDict(dump_dir=self.args.vocab_dump_dir).get_vocab()
        self.i_vocab = SortedByCountsDict(dump_dir=self.args.vocab_dump_dir).get_i_vocab()
        self.model = Util.load_checkpoint(os.path.join(self.args.output_dir, 'checkpoint.tar'))

    def predict(self, **data):
        audio_path = self.dataset.predict_data(**data)[0]
        feature = Util.extract_feature(input_file=audio_path, feature=self.args.feature_type, dim=self.args.input_dim,
                                       cmvn=True)
        feature = Util.build_LFR_features(feature, m=self.args.LFR_m, n=self.args.LFR_n)
        input = torch.from_numpy(feature).to(DEVICE)
        input_length = [input[0].shape[0]]
        input_length = torch.tensor(input_length, dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            nbest_hyps = self.model.recognize(input, input_length, self.i_vocab, args)

        hyp_list = []
        for hyp in nbest_hyps:
            out = hyp['yseq']
            out = [self.i_vocab[idx] for idx in out if idx not in (self.args.sos_id, self.args.eos_id)]
            out = ''.join(out)
            hyp_list.append(out)

        print(hyp_list)

    def predict_all(self, datas):
        labels = []
        for data in datas:
            predicts = self.predict(audio_path=data['audio_path'])

            labels.append(predicts)

        return labels
