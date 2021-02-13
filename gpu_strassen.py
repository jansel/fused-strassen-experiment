#!/usr/bin/env python
import pycuda.autoinit  # noqa
from functools import lru_cache
from pycuda.compiler import SourceModule
from pycuda import gpuarray, driver
from skcuda import cublas
from tabulate import tabulate
from torch import nn
import numpy as np
import pandas as pd
import timeit
import torch

BLOCKSIZE = 128
TRANSPOSE_WEIGHT = False
ALPHA = np.float32(1.0)
BETA = np.float32(0.0)
to_device = gpuarray.to_gpu
# Increase these for more stable measurement
TIMES, REPEAT = 1, 100

CUDA_INDEXING = """
    // common code for indexing into strassen 7-buffer format
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    const int idx0 = 0 * m * n + x * n + y;
    const int idx1 = 1 * m * n + x * n + y;
    const int idx2 = 2 * m * n + x * n + y;
    const int idx3 = 3 * m * n + x * n + y;
    const int idx4 = 4 * m * n + x * n + y;
    const int idx5 = 5 * m * n + x * n + y;
    const int idx6 = 6 * m * n + x * n + y;
"""

CUDA_INDEXING_QUADRANTS = """
    // common code for indexing into 4 quadrants of a input/output tensor
    const int q0 = x * n * 2 + y;
    const int q1 = x * n * 2 + (y + n);
    const int q2 = (x + m) * n * 2 + y;
    const int q3 = (x + m) * n * 2 + (y + n);
"""

CUDA_STRASSEN_SUFFIX_BIAS_RELU = """
    // strassen pointwise suffix (prior layer)
    float v0 = t[idx3] + t[idx4] + t[idx5] - t[idx1];
    float v1 = t[idx0] + t[idx1];
    float v2 = t[idx2] + t[idx3];
    float v3 = t[idx0] - t[idx2] + t[idx4] - t[idx6];

    // bias
    const float bias0 = bias[y];
    const float bias1 = bias[y + n];
    v0 += bias0;
    v1 += bias1;
    v2 += bias0;
    v3 += bias1;

    // relu (could easily swap in something else)
    v0 *= (v0 > 0);
    v1 *= (v1 > 0);
    v2 *= (v2 > 0);
    v3 *= (v3 > 0);
"""

CUDA_STRASSEN_PREFIX = """
    // strassen pointwise prefix (next layer)
    o[idx0] = v0;
    o[idx1] = v0 + v1;
    o[idx2] = v2 + v3;
    o[idx3] = v3;
    o[idx4] = v0 + v3;
    o[idx5] = v1 - v3;
    o[idx6] = v0 - v2;
"""

CUDA_CODE = f"""
__global__ void prefix(float* i, float* o, int m, int n)
{{
    {CUDA_INDEXING}
    {CUDA_INDEXING_QUADRANTS}
    float v0 = i[q0];
    float v1 = i[q1];
    float v2 = i[q2];
    float v3 = i[q3];
    {CUDA_STRASSEN_PREFIX}
}}
__global__ void chain(float* bias, float* t, float* o, int m, int n)
{{
    {CUDA_INDEXING}
    {CUDA_STRASSEN_SUFFIX_BIAS_RELU}
    {CUDA_STRASSEN_PREFIX}
}}
__global__ void suffix(float* bias, float* t, float* o, int m, int n)
{{
    {CUDA_INDEXING}
    {CUDA_STRASSEN_SUFFIX_BIAS_RELU}
    {CUDA_INDEXING_QUADRANTS}
    o[q0] = v0;
    o[q1] = v1;
    o[q2] = v2;
    o[q3] = v3;
}}
"""


@lru_cache(None)
def stream():
    return driver.Stream()


@lru_cache(None)
def cublas_handle():
    cublas_handle = cublas.cublasCreate()
    cublas.cublasSetStream(cublas_handle, stream().handle)
    return cublas_handle


@lru_cache(None)
def cuda_module():
    return SourceModule(CUDA_CODE)


@lru_cache(None)
def cuda_prefix():
    return cuda_module().get_function("prefix").prepare([np.uintp] * 2 + [np.int32] * 2)


@lru_cache(None)
def cuda_chain():
    return cuda_module().get_function("chain").prepare([np.uintp] * 3 + [np.int32] * 2)


@lru_cache(None)
def cuda_suffix():
    return cuda_module().get_function("suffix").prepare([np.uintp] * 3 + [np.int32] * 2)


def cuda_blocks(m, n):
    bs = min(BLOCKSIZE, n)
    assert n % bs == 0
    return (m, n // bs), (1, bs, 1)


def gpu_ptr(data):
    return np.uintp(data.ptr)


def gpu_empty(shape):
    return gpuarray.empty(shape, dtype=np.float32)


class S7(object):
    """ the 7 buffers used in strassen matmul """

    def __init__(self, *shape):
        self.shape = [7] + [s // 2 for s in shape]
        self.m, self.n = np.int32(self.shape[1]), np.int32(self.shape[2])
        self.grid, self.block = cuda_blocks(*self.shape[1:])
        self.set_buffer(gpu_empty(self.shape))

    def set_buffer(self, buf):
        x, y, z = buf.shape
        self.data = buf
        self.bptrs = gpuarray.arange(buf.ptr, buf.ptr + x * y * z * 4, y * z * 4, dtype=cublas.ctypes.c_void_p).gpudata
        self.data_ptr = gpu_ptr(buf)

    def call_cuda(self, fn, *args):
        fn.prepared_async_call(self.grid, self.block, stream(), *args, self.m, self.n)

    def import_weight(self, w):
        """ precompute the 7 weight buffers for strassen matmul """
        m, n = self.shape[1:]
        o = np.empty(self.shape, dtype=np.float32)
        w0, w1, w2, w3 = w[:m, :n], w[:m, n:], w[m:, :n], w[m:, n:]
        o[0, :, :] = w1 - w3
        o[1, :, :] = w3
        o[2, :, :] = w0
        o[3, :, :] = w2 - w0
        o[4, :, :] = w0 + w3
        o[5, :, :] = w2 + w3
        o[6, :, :] = w0 + w1
        if TRANSPOSE_WEIGHT:
            self.set_buffer(to_device(np.copy(o.transpose([0, 2, 1]), order="C")))
        else:
            self.set_buffer(to_device(o))
        return self

    def strassen_prefix(self, data):
        """ the initial pointwise part of strassen matmul """
        self.call_cuda(cuda_prefix(), gpu_ptr(data), self.data_ptr)
        return self

    @staticmethod
    def strassen_chain(bias, input, output):
        """
        fused version of:
            1) the final pointwise of a strassen matmul
            2) a bias+relu
            3) the initial pointwise of the next strassen matmul
        """
        output.call_cuda(cuda_chain(), gpu_ptr(bias), input.data_ptr, output.data_ptr)

    def strassen_suffix(self, bias, out):
        """
        fused version of:
            1) the final pointwise part of a strassen matmul
            2) a bias+relu
        """
        self.call_cuda(cuda_suffix(), gpu_ptr(bias), self.data_ptr, gpu_ptr(out))

    @staticmethod
    def matmul(i, w, o):
        l, m, n = o.shape
        k = i.shape[2]
        if TRANSPOSE_WEIGHT:
            cublas.cublasSgemmBatched(cublas_handle(),
                                      'T', 'N',
                                      n, m, k, ALPHA,
                                      w.bptrs, n,
                                      i.bptrs, k,
                                      BETA,
                                      o.bptrs, n,
                                      l)
        else:
            cublas.cublasSgemmBatched(cublas_handle(),
                                      'N', 'N',
                                      n, m, k, ALPHA,
                                      w.bptrs, n,
                                      i.bptrs, k,
                                      BETA,
                                      o.bptrs, n,
                                      l)


class FusedLinearReLU(object):
    """ Fused version of nn.Linear() and nn.ReLU() """

    def __init__(self, linear, batch_size):
        n = linear.weight.shape[0]
        m = linear.weight.shape[1]
        self.weights = S7(n, m).import_weight(linear.weight.transpose(0, 1))
        self.bias = to_device(np.copy(linear.bias).reshape([2, -1]))
        self.input_buffer = S7(batch_size, n)
        self.output_buffer = S7(batch_size, m)
        self.batch_size = batch_size
        self.hidden_size = linear.weight.shape[1]

    def matmul(self):
        S7.matmul(self.input_buffer, self.weights, self.output_buffer)

    def strassen_prefix(self, input):
        self.input_buffer.strassen_prefix(input)

    def strassen_chain(self, prior):
        S7.strassen_chain(prior.bias, prior.output_buffer, self.input_buffer)

    def strassen_suffix(self, tmp):
        self.output_buffer.strassen_suffix(self.bias, tmp)
        return tmp.get()


def measure(batch_size=128, sizes=[1024] * 9):
    input = torch.randn((batch_size, sizes[0]))
    input_cuda = input.clone().cuda()
    input_gpuarray = to_device(input.numpy())
    output_tmp = gpu_empty((batch_size, sizes[-1]))
    baseline_layers = []
    strassen_layers = []
    gflops = 0.0
    for input_size, output_size in zip(sizes, sizes[1:]):
        baseline_layers.append(nn.Linear(input_size, output_size))
        baseline_layers.append(nn.ReLU(inplace=True))
        strassen_layers.append(FusedLinearReLU(baseline_layers[-2], batch_size))
        gflops += batch_size * output_size * (2 * input_size + 3) / 1000000000.0 * TIMES

    baseline = nn.Sequential(*baseline_layers).eval()
    correct = baseline(input).numpy()
    baseline_cuda = baseline.cuda()

    def baseline():
        return baseline_cuda(input_cuda).cpu()

    def strassen():
        strassen_layers[0].strassen_prefix(input_gpuarray)
        strassen_layers[0].matmul()
        for i in range(1, len(strassen_layers)):
            strassen_layers[i].strassen_chain(strassen_layers[i - 1])
            strassen_layers[i].matmul()
        return strassen_layers[-1].strassen_suffix(output_tmp)

    np.testing.assert_allclose(correct, baseline().numpy(), atol=1e-4)
    np.testing.assert_allclose(correct, strassen(), atol=1e-4)

    algos = (baseline, strassen)
    results = np.zeros([len(algos), REPEAT])
    for rep in range(REPEAT):
        # interleave timing runs to deal with frequency scaling
        for alg in range(len(algos)):
            results[alg, rep] = timeit.timeit(algos[alg], number=TIMES)

    results = np.median(results, axis=1)

    # print(tabulate([gflops / results], headers=[x.__name__ for x in algos]))
    return results[0] / results[1]


def main():
    layers = (8, 4, 2, 1)
    batch_sizes = (128, 64, 32, 16, 8, 4, 2)
    hidden_sizes = (128, 256, 512, 1024, 2048, 4096)
    results = np.zeros([len(layers), len(batch_sizes), 2 + len(hidden_sizes)])
    for l in range(len(layers)):
        for bs in range(len(batch_sizes)):
            results[l, bs, 0] = layers[l]
            results[l, bs, 1] = batch_sizes[bs]
            for h in range(len(hidden_sizes)):
                results[l, bs, 2 + h] = measure(
                    batch_sizes[bs],
                    [hidden_sizes[h]] * (layers[l] + 1))
    rows = results.reshape([-1, 2 + len(hidden_sizes)])
    headers = ['layers', 'batch_size'] + [str(x) for x in hidden_sizes]
    print(tabulate(rows, headers=headers, floatfmt=".2f"))
    pd.DataFrame(rows, columns=headers).to_csv('strassen.csv', index=False)


if __name__ == '__main__':
    with torch.no_grad():
        main()
