import math
from torch.autograd import Function
import torch
from ConvertType import convert_to_bf12, convert_to_bf10, convert_to_bf9

#NOTE: use BF9/BF10 in a specific pass and for the other two use BF12 (as been seen that it doesn't affe) 

class BF9Matmul_fwd(Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        ctx.save_for_backward(inputs, weights)

        weights_bf9 = convert_to_bf9(weights)
        inputs_bf9 = convert_to_bf9(inputs)

        weights_bf9 = weights_bf9.t().contiguous()
        return inputs_bf9.matmul(weights_bf9)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():

            inputs, weights = ctx.saved_tensors

            weights_bf12 = convert_to_bf12(weights)
            inputs_bf12 = convert_to_bf12(inputs)
            grad_output_bf12 = convert_to_bf12(grad_output.contiguous())

            inputs_gradients = grad_output_bf12.matmul(weights_bf12)
            grad_output_new = grad_output_bf12.transpose(1,2).contiguous()
            weights_gradients = grad_output_new.matmul(inputs_bf12)
            weights_gradients = weights_gradients.sum(0)
            return inputs_gradients, weights_gradients

class BF10Matmul_fwd(Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        ctx.save_for_backward(inputs, weights)

        weights_bf10 = convert_to_bf10(weights)
        inputs_bf10 = convert_to_bf10(inputs)

        weights_bf10 = weights_bf10.t().contiguous()
        return inputs_bf10.matmul(weights_bf10)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():

            inputs, weights = ctx.saved_tensors

            weights_bf12 = convert_to_bf12(weights)
            inputs_bf12 = convert_to_bf12(inputs)
            grad_output_bf12 = convert_to_bf12(grad_output.contiguous())

            inputs_gradients = grad_output_bf12.matmul(weights_bf12)
            grad_output_new = grad_output_bf12.transpose(1,2).contiguous()
            weights_gradients = grad_output_new.matmul(inputs_bf12)
            weights_gradients = weights_gradients.sum(0)
            return inputs_gradients, weights_gradients
