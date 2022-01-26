import math
from torch.autograd import Function
import torch
from torch.nn.utils import weight_norm

from ADQ_example import ADQ_Example
from BF16_matmul import BF16Matmul
from BF9_matmul import BF9Matmul
from BF13_matmul import BF13Matmul
from BF11_matmul import BF11Matmul
from BF24_matmul import BF24Matmul
from NormalMatmul import NormalMatmul
from utils import Dtype, Stream, load_kernel, Dtype_size


def custom_matmul(input, weight, compute_flavour):

    experiment = 'normal'
    assert (experiment in ['normal', 'forward', 'backward', 'inference']), "Please provide valid experiment name."

    if compute_flavour == 1:
        # NORMAL matmul
        return NormalMatmul.apply(input, weight, experiment)
    elif compute_flavour == 2:
        # Mantissa 0
        return BF9Matmul.apply(input, weight, experiment)
    elif compute_flavour == 3:
        # Mantissa 2
        return BF11Matmul.apply(input, weight, experiment)
    elif compute_flavour == 4:
        # Mantissa 4
        return BF13Matmul.apply(input, weight, experiment)
    elif compute_flavour == 5:
        # BF16 - Mantissa 7
        return BF16Matmul.apply(input, weight, experiment)
    elif compute_flavour == 6:
        # Mantissa 15
        return BF24Matmul.apply(input, weight, experiment)
    elif compute_flavour == 7:
        # ADQ example with zero 1 element
        return ADQ_Example.apply(input, weight, experiment)
    else:
        raise NotImplementedError




