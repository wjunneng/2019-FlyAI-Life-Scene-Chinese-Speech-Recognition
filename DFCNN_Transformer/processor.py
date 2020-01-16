# -*- coding: utf-8 -*
from flyai.processor.base import Base


class Processor(Base):
    def __init__(self):
        pass

    def input_x(self, audio_path):
        """
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        """
        return audio_path

    def input_y(self, label):
        """
        该参数需要与app.yaml的Model的output-->columns->name 一一对应
        """
        return label

    def output_y(self, data):
        """
        验证时使用，把模型输出的y转为对应的结果
        """
        return data
