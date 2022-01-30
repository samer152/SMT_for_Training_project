import math
from torch.autograd import Function
import torch
from utils import load_kernel, Stream, Dtype

CUDA_NUM_THREADS = 1024


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


def GET_BLOCKS_THREADS(N, T):
    return (N + T - 1) // T


_convertToBF24_cuda = '''
typedef union {
  ${Dtype} f;
  struct {
    unsigned int mantissa : 23;
    unsigned int exponent : 8;
    unsigned int sign : 1;
  } parts;
} float_cast;
extern "C"
#define BFLOAT24_MASK (0x007fff00)
__global__ void convertToBF24_cuda(const ${Dtype}* input, ${Dtype}* data_output) {      
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < ${n}){
        float_cast a;
        a.f = input[index];
        a.parts.mantissa &= BFLOAT24_MASK;
        data_output[index] = a.f;
    }
}
'''


def _convertToBF24(input):

    # Each Cuda thread work on a single element
    n = input.numel()

    shape = torch.Size(input.size())
    data_output = input.new(*shape)

    # name of the CUDA function and the pointer to it
    func_name, func_ptr = 'convertToBF24_cuda', _convertToBF24_cuda

    with torch.cuda.device_of(input):
        f = load_kernel(func_name,
                        func_ptr, Dtype=Dtype(input), n=n)
        f(block=(CUDA_NUM_THREADS, 1, 1),
          grid=(GET_BLOCKS(n), 1, 1),
          args=[input.data_ptr(), data_output.data_ptr()],
          stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
    return data_output


class convertToBF24(Function):
    @staticmethod
    def forward(ctx, input):
        temp = _convertToBF24(input)
        return temp

    @staticmethod
    def backward(ctx, grad_output):
        # No need to do anything for Backward
        return grad_output


def convert_to_bf24(input):
    return convertToBF24.apply(input)



class BF24Matmul(Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        ctx.save_for_backward(inputs, weights)

        # convert to BF24
        weights_bf24 = convert_to_bf24(weights)
        inputs_bf24 = convert_to_bf24(inputs)

        weights_bf24 = weights_bf24.t().contiguous()
        return inputs_bf24.matmul(weights_bf24)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():

            inputs, weights = ctx.saved_tensors

            weights_bf24 = convert_to_bf24(weights)
            inputs_bf24 = convert_to_bf24(inputs)
            grad_output_bf24 = convert_to_bf24(grad_output.contiguous())

            inputs_gradients = grad_output_bf24.matmul(weights_bf24)
            grad_output_new = grad_output_bf24.transpose(1,2).contiguous()
            weights_gradients = grad_output_new.matmul(inputs_bf24)
            weights_gradients = weights_gradients.sum(0)
            return inputs_gradients, weights_gradients