import math
from torch.autograd import Function
import torch
from ConvertType import convert_to_bf9, convert_to_bf16

class BF16Matmul_ingrad_sym(Function):
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

            weights_bf16 = convert_to_bf16(weights)
            inputs_bf9 = convert_to_bf9(inputs)
            grad_output_bf9 = convert_to_bf9(grad_output.contiguous())
            grad_output_bf16 = convert_to_bf16(grad_output.contiguous())

            inputs_gradients = grad_output_bf16.matmul(weights_bf16)
            grad_output_new = grad_output_bf9.transpose(1,2).contiguous()
            weights_gradients = grad_output_new.matmul(inputs_bf9)
            weights_gradients = weights_gradients.sum(0)
            return inputs_gradients, weights_gradients

class BF16Matmul_ingrad_asym_grad_output(Function):
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

            weights_bf9 = convert_to_bf9(weights)
            inputs_bf9 = convert_to_bf9(inputs)
            grad_output_bf9 = convert_to_bf9(grad_output.contiguous())
            grad_output_bf16 = convert_to_bf16(grad_output.contiguous())

            inputs_gradients = grad_output_bf16.matmul(weights_bf9)
            grad_output_new = grad_output_bf9.transpose(1,2).contiguous()
            weights_gradients = grad_output_new.matmul(inputs_bf9)
            weights_gradients = weights_gradients.sum(0)
            return inputs_gradients, weights_gradients

class BF16Matmul_ingrad_asym_weights(Function):
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

            weights_bf16 = convert_to_bf16(weights)
            inputs_bf9 = convert_to_bf9(inputs)
            grad_output_bf9 = convert_to_bf9(grad_output.contiguous())

            inputs_gradients = grad_output_bf9.matmul(weights_bf16)
            grad_output_new = grad_output_bf9.transpose(1,2).contiguous()
            weights_gradients = grad_output_new.matmul(inputs_bf9)
            weights_gradients = weights_gradients.sum(0)
            return inputs_gradients, weights_gradients
