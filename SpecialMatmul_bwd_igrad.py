import math
from torch.autograd import Function
import torch
from ConvertType import convert_to_bf12, convert_to_bf10, convert_to_bf9

class BF9Matmul_bwd_igrad(Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        ctx.save_for_backward(inputs, weights)

        weights_bf12 = convert_to_bf12(weights)
        inputs_bf12 = convert_to_bf12(inputs)

        weights_bf12 = weights_bf12.t().contiguous()
        return inputs_bf12.matmul(weights_bf12)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():

            inputs, weights = ctx.saved_tensors

            weights_bf12 = convert_to_bf12(weights)
            inputs_bf9 = convert_to_bf9(inputs)
            grad_output_bf12 = convert_to_bf12(grad_output.contiguous())

            inputs_gradients = grad_output_bf12.matmul(weights_bf12)
            grad_output_new = grad_output_bf12.transpose(1,2).contiguous()
            weights_gradients = grad_output_new.matmul(inputs_bf9)
            weights_gradients = weights_gradients.sum(0)
            return inputs_gradients, weights_gradients

class BF10Matmul_bwd_igrad(Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        ctx.save_for_backward(inputs, weights)

        weights_bf12 = convert_to_bf12(weights)
        inputs_bf12 = convert_to_bf12(inputs)

        weights_bf12 = weights_bf12.t().contiguous()
        return inputs_bf12.matmul(weights_bf12)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():

            inputs, weights = ctx.saved_tensors

            weights_bf12 = convert_to_bf12(weights)
            inputs_bf10 = convert_to_bf10(inputs)
            grad_output_bf12 = convert_to_bf12(grad_output.contiguous())

            inputs_gradients = grad_output_bf12.matmul(weights_bf12)
            grad_output_new = grad_output_bf12.transpose(1,2).contiguous()
            weights_gradients = grad_output_new.matmul(inputs_bf10)
            weights_gradients = weights_gradients.sum(0)
            return inputs_gradients, weights_gradients
