import math
from torch.autograd import Function
import torch
import Config as cfg
from utils import load_kernel, Stream, Dtype

CUDA_NUM_THREADS = 1024


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


def GET_BLOCKS_THREADS(N, T):
    return (N + T - 1) // T


_convertToBF9_cuda = '''
typedef union {
  ${Dtype} f;
  struct {
    unsigned int mantissa : 23;
    unsigned int exponent : 8;
    unsigned int sign : 1;
  } parts;
} float_cast;
extern "C"
#define BFLOAT9_MASK (0x00000000)
__global__ void convertToBF9_cuda(const ${Dtype}* input, ${Dtype}* data_output) {      
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < ${n}){
        float_cast a;
        a.f = input[index];
        a.parts.mantissa &= BFLOAT9_MASK;
        data_output[index] = a.f;
    }
}
'''


def _convertToBF9(input):

    # Each Cuda thread work on a single element
    n = input.numel()

    shape = torch.Size(input.size())
    data_output = input.new(*shape)

    # name of the CUDA function and the pointer to it
    func_name, func_ptr = 'convertToBF9_cuda', _convertToBF9_cuda

    with torch.cuda.device_of(input):
        f = load_kernel(func_name,
                        func_ptr, Dtype=Dtype(input), n=n)
        f(block=(CUDA_NUM_THREADS, 1, 1),
          grid=(GET_BLOCKS(n), 1, 1),
          args=[input.data_ptr(), data_output.data_ptr()],
          stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
    return data_output


class convertToBF9(Function):
    @staticmethod
    def forward(ctx, input):
        temp = _convertToBF9(input)
        return temp

    @staticmethod
    def backward(ctx, grad_output):
        # No need to do anything for Backward
        return grad_output


def convert_to_bf9(input):
    return convertToBF9.apply(input)


class BF9Matmul(Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        ctx.save_for_backward(inputs, weights)
        if cfg.EXPERIMENT == 'forward' or cfg.EXPERIMENT == 'normal':
            # convert to BF9
            weights_bf9 = convert_to_bf9(weights)
            inputs_bf9 = convert_to_bf9(inputs)

            weights_bf9 = weights_bf9.t().contiguous()
            return inputs_bf9.matmul(weights_bf9)
        else:
            # Normal MatMul
            weights = weights.t().contiguous()
            return inputs.matmul(weights)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():

            inputs, weights = ctx.saved_tensors
            if cfg.EXPERIMENT == 'backward' or cfg.EXPERIMENT == 'normal':
                weights_bf9 = convert_to_bf9(weights)
                inputs_bf9 = convert_to_bf9(inputs)
                grad_output_bf9 = convert_to_bf9(grad_output.contiguous())
                inputs_gradients = grad_output_bf9.matmul(weights_bf9)
                grad_output_new = grad_output_bf9.transpose(1,2).contiguous()
                weights_gradients = grad_output_new.matmul(inputs_bf9)
                weights_gradients = weights_gradients.sum(0)
            else:
                # Normal MatMul
                inputs, weights = ctx.saved_tensors
                inputs_gradients = grad_output.matmul(weights)
                grad_output_new = grad_output.transpose(1, 2).contiguous()
                weights_gradients = grad_output_new.matmul(inputs)
                weights_gradients = weights_gradients.sum(0)
            return inputs_gradients, weights_gradients