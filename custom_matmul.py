import math
from torch.autograd import Function
import torch
from utils import Dtype, Stream, load_kernel, Dtype_size
CUDA_NUM_THREADS = 1024

def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


def GET_BLOCKS_THREADS(N, T):
    return (N + T - 1) // T



class HalfMatmul(Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        ctx.save_for_backward(inputs, weights)
        weights = weights.t().contiguous()

        return inputs.matmul(weights)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():

            inputs, weights = ctx.saved_tensors

            #zero weights to calculate output gradients
            weights[:, 1::2] = 0
            inputs_gradients = grad_output.matmul(weights)
            grad_output_new = grad_output.transpose(1,2).contiguous()
            weights_gradients = grad_output_new.matmul(inputs)
            weights_gradients = weights_gradients.sum(0)
            return inputs_gradients, weights_gradients


class NormalMatmul(Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        ctx.save_for_backward(inputs, weights)
        weights = weights.t().contiguous()
        return inputs.matmul(weights)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():

            inputs, weights = ctx.saved_tensors
            inputs_gradients = grad_output.matmul(weights)
            grad_output_new = grad_output.transpose(1,2).contiguous()
            weights_gradients = grad_output_new.matmul(inputs)
            weights_gradients = weights_gradients.sum(0)
            return inputs_gradients, weights_gradients


def custom_matmul(input, weight, threads, muxing):
    if threads == 1 and muxing == 0:
        return NormalMatmul.apply(input, weight)
    elif threads == 2 and muxing == 1:
        return HalfMatmul.apply(input, weight)
    else:
        raise NotImplementedError

