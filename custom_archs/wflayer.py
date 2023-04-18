import torch
import torch.nn as nn

Tensor = torch.Tensor


class WFLayer(nn.Module):

    def __init__(
        self, m:float = 5., classes_number:int = 1000
    ) -> None:
        super(WFLayer,self).__init__()
        self.m,self.classes_number =\
            m,classes_number

        self.activations = []

    def save_activations(self, model:nn.Module, input:Tensor, output:Tensor) -> None:
        self.activations.append(output.data)
