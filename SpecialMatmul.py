import math
from torch.autograd import Function
import torch
from ConvertType import convert_to_bf16, convert_to_nvtf, convert_to_bf14, convert_to_bf12, convert_to_bf10, convert_to_pxr24, convert_to_bf9

class BF16Matmul(Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        ctx.save_for_backward(inputs, weights)

        # convert to BF16
        weights_bf16 = convert_to_bf16(weights)
        inputs_bf16 = convert_to_bf16(inputs)

        weights_bf16 = weights_bf16.t().contiguous()
        return inputs_bf16.matmul(weights_bf16)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():

            inputs, weights = ctx.saved_tensors

            weights_bf16 = convert_to_bf16(weights)
            inputs_bf16 = convert_to_bf16(inputs)
            grad_output_bf16 = convert_to_bf16(grad_output.contiguous())

            inputs_gradients = grad_output_bf16.matmul(weights_bf16)
            grad_output_new = grad_output_bf16.transpose(1,2).contiguous()
            weights_gradients = grad_output_new.matmul(inputs_bf16)
            weights_gradients = weights_gradients.sum(0)
            return inputs_gradients, weights_gradients

class BF14Matmul(Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        ctx.save_for_backward(inputs, weights)

        # convert to BF14
        weights_bf14 = convert_to_bf14(weights)
        inputs_bf14 = convert_to_bf14(inputs)

        weights_bf14 = weights_bf14.t().contiguous()
        return inputs_bf14.matmul(weights_bf14)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():

            inputs, weights = ctx.saved_tensors

            weights_bf14 = convert_to_bf14(weights)
            inputs_bf14 = convert_to_bf14(inputs)
            grad_output_bf14 = convert_to_bf14(grad_output.contiguous())

            inputs_gradients = grad_output_bf14.matmul(weights_bf14)
            grad_output_new = grad_output_bf14.transpose(1,2).contiguous()
            weights_gradients = grad_output_new.matmul(inputs_bf14)
            weights_gradients = weights_gradients.sum(0)
            return inputs_gradients, weights_gradients

class BF12Matmul(Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        ctx.save_for_backward(inputs, weights)

        # convert to BF12
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

            inputs_gradients = grad_output_bf12.matmul(weights_bf12)
            grad_output_new = grad_output_bf12.transpose(1,2).contiguous()
            weights_gradients = grad_output_new.matmul(inputs_bf12)
            weights_gradients = weights_gradients.sum(0)
            return inputs_gradients, weights_gradients

class BF10Matmul(Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        ctx.save_for_backward(inputs, weights)

        # convert to BF10
        weights_bf10 = convert_to_bf10(weights)
        inputs_bf10 = convert_to_bf10(inputs)

        weights_bf10 = weights_bf10.t().contiguous()
        return inputs_bf10.matmul(weights_bf10)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():

            inputs, weights = ctx.saved_tensors

            weights_bf10 = convert_to_bf10(weights)
            inputs_bf10 = convert_to_bf10(inputs)
            grad_output_bf10 = convert_to_bf10(grad_output.contiguous())

            inputs_gradients = grad_output_bf10.matmul(weights_bf10)
            grad_output_new = grad_output_bf10.transpose(1,2).contiguous()
            weights_gradients = grad_output_new.matmul(inputs_bf10)
            weights_gradients = weights_gradients.sum(0)
            return inputs_gradients, weights_gradients

class NVTFMatmul(Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        ctx.save_for_backward(inputs, weights)

        # convert to NVTF
        weights_nvtf = convert_to_nvtf(weights)
        inputs_nvtf = convert_to_nvtf(inputs)

        weights_nvtf = weights_nvtf.t().contiguous()
        return inputs_nvtf.matmul(weights_nvtf)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():

            inputs, weights = ctx.saved_tensors

            weights_nvtf = convert_to_nvtf(weights)
            inputs_nvtf = convert_to_nvtf(inputs)
            grad_output_nvtf = convert_to_nvtf(grad_output.contiguous())

            inputs_gradients = grad_output_nvtf.matmul(weights_nvtf)
            grad_output_new = grad_output_nvtf.transpose(1,2).contiguous()
            weights_gradients = grad_output_new.matmul(inputs_nvtf)
            weights_gradients = weights_gradients.sum(0)
            return inputs_gradients, weights_gradients

class PXR24Matmul(Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        ctx.save_for_backward(inputs, weights)

        # convert to PXR24
        weights_pxr24 = convert_to_pxr24(weights)
        inputs_pxr24 = convert_to_pxr24(inputs)

        weights_pxr24 = weights_pxr24.t().contiguous()
        return inputs_pxr24.matmul(weights_pxr24)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():

            inputs, weights = ctx.saved_tensors

            weights_pxr24 = convert_to_pxr24(weights)
            inputs_pxr24 = convert_to_pxr24(inputs)
            grad_output_pxr24 = convert_to_pxr24(grad_output.contiguous())

            inputs_gradients = grad_output_pxr24.matmul(weights_pxr24)
            grad_output_new = grad_output_pxr24.transpose(1,2).contiguous()
            weights_gradients = grad_output_new.matmul(inputs_pxr24)
            weights_gradients = weights_gradients.sum(0)
            return inputs_gradients, weights_gradients

class BF9Matmul(Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        ctx.save_for_backward(inputs, weights)

        # convert to BF9
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

            inputs_gradients = grad_output_bf9.matmul(weights_bf9)
            grad_output_new = grad_output_bf9.transpose(1,2).contiguous()
            weights_gradients = grad_output_new.matmul(inputs_bf9)
            weights_gradients = weights_gradients.sum(0)
            return inputs_gradients, weights_gradients