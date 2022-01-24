import math
from torch.autograd import Function
import torch


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