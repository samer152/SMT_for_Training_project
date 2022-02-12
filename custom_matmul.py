import math
from torch.autograd import Function
import torch
from torch.nn.utils import weight_norm

from ADQ_example import ADQ_Example
from SpecialMatmul import BF16Matmul, BF14Matmul, BF12Matmul, BF10Matmul, NVTFMatmul, PXR24Matmul, BF9Matmul
from NormalMatmul import NormalMatmul
from utils import Dtype, Stream, load_kernel, Dtype_size


def custom_matmul(input, weight, compute_flavour):
    if compute_flavour == 1:
        # NORMAL matmul
        return NormalMatmul.apply(input, weight)
    elif compute_flavour == 3:
        # ADQ example with zero 1 element
        return ADQ_Example.apply(input, weight)
    # Resiliency compute flavours
    elif compute_flavour == 2:
        # BF16
        return BF16Matmul.apply(input, weight)
    elif compute_flavour == 4:
        # BF14
        return BF14Matmul.apply(input, weight)        
    elif compute_flavour == 5:
        # BF12
        return BF12Matmul.apply(input, weight)        
    elif compute_flavour == 6:
        # BF10
        return BF10Matmul.apply(input, weight)        
    elif compute_flavour == 7:
        # NVTF
        return NVTFMatmul.apply(input, weight)        
    elif compute_flavour == 8:
        # PXR24
        return PXR24Matmul.apply(input, weight)        
    elif compute_flavour == 9:
        # BF9
        return BF9Matmul.apply(input, weight)        
    else:
        raise NotImplementedError




