import math
from torch.autograd import Function
import torch

from utils import load_kernel, Dtype, Stream

CUDA_NUM_THREADS = 1024


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


def GET_BLOCKS_THREADS(N, T):
    return (N + T - 1) // T


_zero_a_value_in_pair_cuda = '''
typedef union {
  ${Dtype} f;
  struct {
    unsigned int mantissa : 23;
    unsigned int exponent : 8;
    unsigned int sign : 1;
  } parts;
} float_cast;
extern "C"
__global__ void zero_a_value_in_pair_cuda(const ${Dtype}* input, ${Dtype}* data_output) {      
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    if(index < ${n}){

        // get pair number
        unsigned int cur_pair = index % ${pairs};

        // get pair's window in matrix (row)
        unsigned int cur_window = index / ${pairs};
        float_cast inp1, inp2;

        // calculate paris row pointer
        unsigned int cur_window_ptr =  cur_window * (2 * ${pairs} + ${remain});

        // calculate pairs pointer
        unsigned int cur_pair_ptr = 2 * cur_pair;

        // read pairs to floating variables
        inp1.f = input[cur_window_ptr + cur_pair_ptr];
        inp2.f = input[cur_window_ptr + cur_pair_ptr + 1];

        // read the exponents of the first in pair
        unsigned int inp1_e = inp1.parts.exponent;

        // read the mantissa of the first in pair
        unsigned int inp1_m = inp1.parts.mantissa;

        // do some tricks/conditions/control work over the values


        // zero the first number    
        // this is equivalent to inp1.f = 0;
        inp1.parts.exponent = 0;
        inp1.parts.mantissa = 0;

        // re-write the new values to the result vector
        data_output[cur_window_ptr + cur_pair_ptr] = inp1.f;
        data_output[cur_window_ptr + cur_pair_ptr + 1] = inp2.f;    
    }
}
'''


def _zero_a_value_in_pair(input, dim):
    # number of n is how many pairs are in the input vector
    if dim == 2:
        pairs = int(math.floor(input.size(1)/2))
        remain = input.size(1) % 2
        n = int(pairs * input.size(0))
    elif dim == 3:
        pairs = int(math.floor(input.size(2)/2))
        remain = input.size(2) % 2
        n = int(pairs * input.size(0) * input.size(1))
    else:
        raise NotImplementedError

    shape = torch.Size(input.size())
    data_output = input.new(*shape)
    data_output = torch.zeros_like(data_output)

    # name of CUDA function and pointer
    func_name, func_ptr = 'zero_a_value_in_pair_cuda', _zero_a_value_in_pair_cuda

    with torch.cuda.device_of(input):
        f = load_kernel(func_name,
                        func_ptr, Dtype=Dtype(input), n=n, pairs=pairs, windows=input.size(0), remain=remain)
        f(block=(CUDA_NUM_THREADS, 1, 1),
          grid=(GET_BLOCKS(n), 1, 1),
          args=[input.data_ptr(), data_output.data_ptr()],
          stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

    if remain:
        if dim == 2:
            data_output[:,input.size(1) - 1] = input[:, input.size(1)-1]
        elif dim == 3:
            data_output[:, :, input.size(2) - 1] = input[:, :, input.size(2) - 1]
        else:
            raise NotImplementedError
    return data_output

class ZeroAValueInPair(Function):
    @staticmethod
    def forward(ctx, input, dim):
        temp = _zero_a_value_in_pair(input, dim)
        return temp

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def zero_a_value_in_pair(input, dim = 2):
    return ZeroAValueInPair.apply(input, dim)

class ADQ_Example(Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        ctx.save_for_backward(inputs, weights)

        # Do ADQ
        weights = zero_a_value_in_pair(weights)

        # compute FWD
        weights = weights.t().contiguous()
        return inputs.matmul(weights)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            inputs, weights = ctx.saved_tensors

            # Do ADQ that zeros 1 element each time
            weights = zero_a_value_in_pair(weights.t().contiguous()).t().contiguous()

            # compute input gradients
            inputs_gradients = grad_output.matmul(weights)

            # Do ADQ
            grad_output_new = grad_output.transpose(1, 2).contiguous()
            grad_output_new = zero_a_value_in_pair(grad_output_new, dim=3)

            # compute weights gradients
            weights_gradients = grad_output_new.matmul(inputs)
            weights_gradients = weights_gradients.sum(0)

            return inputs_gradients, weights_gradients