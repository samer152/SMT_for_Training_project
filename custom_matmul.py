import math
from torch.autograd import Function
import torch
from torch.nn.utils import weight_norm

from ADQ_example import ADQ_Example
from BF16_matmul import BF16Matmul
from NormalMatmul import NormalMatmul
from utils import Dtype, Stream, load_kernel, Dtype_size


def custom_matmul(input, weight, compute_flavour):
    if compute_flavour == 1:
        # NORMAL matmul
        return NormalMatmul.apply(input, weight)
    elif compute_flavour == 2:
        # BF16
        return BF16Matmul.apply(input, weight)
    elif compute_flavour == 3:
        # ADQ example with zero 1 element
        return ADQ_Example.apply(input, weight)
    else:
        raise NotImplementedError




