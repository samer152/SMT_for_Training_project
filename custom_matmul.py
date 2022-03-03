import math
from torch.autograd import Function
import torch
from torch.nn.utils import weight_norm

from ADQ_example import ADQ_Example
from NormalMatmul import NormalMatmul
from SpecialMatmul import BF16Matmul, BF14Matmul, BF12Matmul, BF10Matmul, NVTFMatmul, PXR24Matmul, BF9Matmul
from SpecialMatmul_convertedSum import BF16MatMulSum, BF14MatMulSum, BF12MatMulSum, BF10MatMulSum, NVTFMatMulSum, PXR24MatMulSum, BF9MatMulSum
from SpecialMatmul_fwd import BF9Matmul_fwd, BF10Matmul_fwd
from SpecialMatmul_bwd_igrad import BF9Matmul_bwd_igrad, BF10Matmul_bwd_igrad
from SpecialMatmul_bwd_wgrad import BF9Matmul_bwd_wgrad, BF10Matmul_bwd_wgrad
from SpecialMatmul_ingradsBF14 import BF14Matmul_ingrad_sym, BF14Matmul_ingrad_asym_grad_output, BF14Matmul_ingrad_asym_weights

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
        # BF16 - NOTE: plot distributions of matrices
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
    elif compute_flavour == 10:
        # BF16_mul_sum
        return BF16MatMulSum.apply(input, weight)
    elif compute_flavour == 11:
        # BF14_mul_sum
        return BF14MatMulSum.apply(input, weight)
    elif compute_flavour == 12:
        # BF12_mul_sum
        return BF12MatMulSum.apply(input, weight)
    elif compute_flavour == 13:
        # BF10_mul_sum
        return BF10MatMulSum.apply(input, weight)
    elif compute_flavour == 14:
        # BF9_mul_sum
        return BF9MatMulSum.apply(input, weight)
    elif compute_flavour == 15:
        # NVTF_mul_sum
        return NVTFMatMulSum.apply(input, weight)
    elif compute_flavour == 16:
        # PXR24_mul_sum
        return PXR24MatMulSum.apply(input, weight)
    elif compute_flavour == 20:
        # fwd_BF9
        return BF9Matmul_fwd.apply(input, weight)
    elif compute_flavour == 21:
        # fwd_BF10
        return BF10Matmul_fwd.apply(input, weight)
    elif compute_flavour == 22:
        # BF9Matmul_bwd_wgrad
        return BF9Matmul_bwd_wgrad.apply(input, weight)
    elif compute_flavour == 23:
        # BF10Matmul_bwd_wgrad
        return BF10Matmul_bwd_wgrad.apply(input, weight)
    elif compute_flavour == 24:
        # BF9Matmul_bwd_igrad
        return BF9Matmul_bwd_igrad.apply(input, weight)
    elif compute_flavour == 25:
        # BF10Matmul_bwd_igrad
        return BF10Matmul_bwd_igrad.apply(input, weight)
    elif compute_flavour == 30:
        # BF14Matmul_ingrad_sym
        return BF14Matmul_ingrad_sym.apply(input, weight)
    elif compute_flavour == 31:
        # BF14Matmul_ingrad_asym_grad_output
        return BF14Matmul_ingrad_asym_grad_output.apply(input, weight)
    elif compute_flavour == 32:
        # BF14Matmul_ingrad_asym_weights
        return BF14Matmul_ingrad_asym_weights.apply(input, weight)
    else:
        raise NotImplementedError




