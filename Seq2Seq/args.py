# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

# 日志路径
log_dir = os.path.join(os.getcwd(), 'data/log')

# 随机种子
seed = 42
# 模型类型
model_name = 'Seq2Seq'

