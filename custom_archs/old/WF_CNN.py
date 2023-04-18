import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from collections.abc import Iterable
from typing import Optional, List, Tuple, Union

Tensor = torch.Tensor
amax,amin=1e1,-1e1
eps=1e-12


class WF-CNN()