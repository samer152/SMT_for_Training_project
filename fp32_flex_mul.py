import math
from torch.autograd import Function
import torch
import struct


def binary(num):
    return format(struct.unpack('!I',struct.pack('!f',num))[0], '032b')


def fp32(num):
    return struct.unpack('!f', struct.pack('!I', int(num, 2)))[0]


def squeeze_round(x):
    xn = binary(x)
    sign = xn[0]
    exp1, exp2 = xn[1:5], xn[5:9]
    man1, man2 = xn[9:21], xn[21:]
    # if exp1 != 0 and exp2 != 0:
    #     if exp2[0] == 1:
    #         exp1 = int(exp1, 2) + 1
    #         if exp1 > 15:
    #             exp1 = 15
    #         exp1 = format(exp1, '04b')
    #     exp2 = '0000'
    if man1 != 0 and man2 != 0:
        if man2[0] == 1:
            man1 = int(man1, 2) + 1
            if man1 > 4095:
                man1 = 4095
            man1 = format(man1, '012b')
        man2 = '00000000000'
    xn = sign + exp1 + exp2 + man1 + man2
    return fp32(xn)


def squeeze_values(x1, x2):
    if x1 == 0 or x2 == 0:
        return x1, x2
    x1n = squeeze_round(x1)
    x2n = squeeze_round(x2)
    return x1n, x2n


def squeeze(inputs, weights):
    len_weights = len(weights)
    for j in range(len(inputs)):
        for i in range(0, len(inputs[0])-1, 2):
            for k in range(len_weights):
                    inputs[j][i][k], inputs[j][i+1][k] = squeeze_values(inputs[j][i][k].item(), inputs[j][i+1][k].item())
    return inputs


class FP32FlexMatmul(Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        ctx.save_for_backward(inputs, weights)
        weights = weights.t().contiguous()
        squeezed_inputs = (squeeze(inputs, weights)).contiguous()

        return squeezed_inputs.matmul(weights)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():

            inputs, weights = ctx.saved_tensors

            squeezed_grad_output = (squeeze(grad_output, weights)).contiguous()
            inputs_gradients = squeezed_grad_output.matmul(weights)
            grad_output_new = grad_output.transpose(1,2)
            squeezed_grad_output_new = (squeeze(grad_output_new, inputs)).contiguous()
            weights_gradients = squeezed_grad_output_new.matmul(inputs)
            weights_gradients = weights_gradients.sum(0)
            return inputs_gradients, weights_gradients


