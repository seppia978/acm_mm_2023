import torch
import torch.nn as nn


class WTFLayer(nn.Module):

    def __init__(
        self, m:float = 5., classes_number:int = 1000
    ) -> None:
        super(WTFLayer,self).__init__()
        self.m,self.classes_number =\
            m,classes_number