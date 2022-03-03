import math
from torch.autograd import Function
import torch
from ConvertType import convert_to_bf12, convert_to_bf14

class BF14Matmul_ingrad_sym(Function):
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

            weights_bf14 = convert_to_bf14(weights)
            inputs_bf12 = convert_to_bf12(inputs)
            grad_output_bf12 = convert_to_bf12(grad_output.contiguous())
            grad_output_bf14 = convert_to_bf14(grad_output.contiguous())

            inputs_gradients = grad_output_bf14.matmul(weights_bf14)
            grad_output_new = grad_output_bf12.transpose(1,2).contiguous()
            weights_gradients = grad_output_new.matmul(inputs_bf12)
            weights_gradients = weights_gradients.sum(0)
            return inputs_gradients, weights_gradients

class BF14Matmul_ingrad_asym_grad_output(Function):
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
            inputs_bf12 = convert_to_bf12(inputs)
            grad_output_bf12 = convert_to_bf12(grad_output.contiguous())
            grad_output_bf14 = convert_to_bf14(grad_output.contiguous())

            inputs_gradients = grad_output_bf14.matmul(weights_bf12)
            grad_output_new = grad_output_bf12.transpose(1,2).contiguous()
            weights_gradients = grad_output_new.matmul(inputs_bf12)
            weights_gradients = weights_gradients.sum(0)
            return inputs_gradients, weights_gradients

class BF14Matmul_ingrad_asym_weights(Function):
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

            weights_bf14 = convert_to_bf14(weights)
            inputs_bf12 = convert_to_bf12(inputs)
            grad_output_bf12 = convert_to_bf12(grad_output.contiguous())

            inputs_gradients = grad_output_bf12.matmul(weights_bf14)
            grad_output_new = grad_output_bf12.transpose(1,2).contiguous()
            weights_gradients = grad_output_new.matmul(inputs_bf12)
            weights_gradients = weights_gradients.sum(0)
            return inputs_gradients, weights_gradients
