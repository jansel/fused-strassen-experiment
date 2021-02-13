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
TRANSPOSE_WEIGHT = True
ALPHA = np.float32(1.0)
BETA = np.float32(0.0)
to_device = gpuarray.to_gpu
# Increase these for more stable measurement
TIMES, REPEAT = 1, 100


def gen_cuda_pointwise(pointwise):
    return f"""
    {{
        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;
        float v = o[x * stride + y];
        {pointwise};
        o[x * stride + y] = v;
    }} 
    """


CUDA_CODE = f"""
__global__ void relu(float* o, int stride)
    {gen_cuda_pointwise("v *= (v > 0)")}
__global__ void bias(float* bias, float* o, int stride) 
    {gen_cuda_pointwise("v += bias[y]")}
__global__ void bias_relu(float* bias, float* o, int stride)
    {gen_cuda_pointwise("v += bias[y]; v *= (v > 0)")}
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
def cuda_relu():
    return cuda_module().get_function("relu").prepare([np.uintp] + [np.int32])


@lru_cache(None)
def cuda_bias():
    return cuda_module().get_function("bias").prepare([np.uintp] * 2 + [np.int32])


@lru_cache(None)
def cuda_bias_relu():
    return cuda_module().get_function("bias_relu").prepare([np.uintp] * 2 + [np.int32])


def cuda_blocks(m, n):
    bs = min(BLOCKSIZE, n)
    assert n % bs == 0
    return (m, n // bs), (1, bs, 1)


def gpu_ptr(data):
    return np.uintp(data.ptr)


class Buffer(object):
    def __init__(self, *shape):
        self.shape = list(shape)
        self.m, self.n = np.int32(self.shape[0]), np.int32(self.shape[1])
        self.grid, self.block = cuda_blocks(*self.shape)
        self.set_buffer(torch.empty(shape))

    def set_buffer(self, buf):
        buf = to_device(np.copy(buf.numpy(), order="C"))
        self.data = buf
        self.data_ptr = gpu_ptr(buf)
        return self

    def call_cuda(self, fn, *args):
        fn.prepared_async_call(self.grid, self.block, stream(), *args, self.data_ptr, self.n)

    @staticmethod
    def matmul(i, w, o):
        m, n = o.shape
        k = i.shape[1]
        if TRANSPOSE_WEIGHT:
            cublas.cublasSgemm(cublas_handle(),
                               'T', 'N',
                               n, m, k, ALPHA,
                               w.data.gpudata, n,
                               i.data.gpudata, k,
                               BETA,
                               o.data.gpudata, n)
        else:
            cublas.cublasSgemm(cublas_handle(),
                               'N', 'N',
                               n, m, k, ALPHA,
                               w.data.gpudata, n,
                               i.data.gpudata, k,
                               BETA,
                               o.data.gpudata, n)


class LinearReLU(object):
    def __init__(self, linear, batch_size):
        n = linear.weight.shape[0]
        m = linear.weight.shape[1]
        if TRANSPOSE_WEIGHT:
            self.weights = Buffer(n, m).set_buffer(linear.weight)
        else:
            self.weights = Buffer(n, m).set_buffer(linear.weight.transpose(0, 1))
        self.bias = to_device(np.copy(linear.bias))
        self.output_buffer = Buffer(batch_size, m)
        self.batch_size = batch_size
        self.hidden_size = linear.weight.shape[1]

    def unfused(self, input_buffer):
        Buffer.matmul(input_buffer, self.weights, self.output_buffer)
        self.output_buffer.call_cuda(cuda_bias(), gpu_ptr(self.bias))
        self.output_buffer.call_cuda(cuda_relu())

    def fused(self, input_buffer):
        Buffer.matmul(input_buffer, self.weights, self.output_buffer)
        self.output_buffer.call_cuda(cuda_bias_relu(), gpu_ptr(self.bias))


def measure(batch_size=128, sizes=[1024] * 9):
    input_cpu = torch.randn((batch_size, sizes[0]))
    input_cuda = input_cpu.clone().cuda()
    input_gpuarray = Buffer(*input_cpu.shape).set_buffer(input_cpu)

    baseline_layers = []
    experiment_layers = []
    gflops = 0.0
    for input_size, output_size in zip(sizes, sizes[1:]):
        baseline_layers.append(nn.Linear(input_size, output_size))
        baseline_layers.append(nn.ReLU(inplace=True))
        experiment_layers.append(LinearReLU(baseline_layers[-2], batch_size))
        gflops += batch_size * output_size * (2 * input_size + 3) / 1000000000.0 * TIMES

    baseline = nn.Sequential(*baseline_layers).eval()
    correct = np.copy(baseline(input_cpu).numpy())
    baseline_cuda = baseline.cuda()

    def baseline():
        return baseline_cuda(input_cuda).cpu()

    def unfused():
        prior = input_gpuarray
        for layer in experiment_layers:
            layer.unfused(prior)
            prior = layer.output_buffer
        return prior.data.get()

    def fused():
        prior = input_gpuarray
        for layer in experiment_layers:
            layer.fused(prior)
            prior = layer.output_buffer
        return prior.data.get()

    np.testing.assert_allclose(correct, baseline().numpy(), atol=1e-5)
    np.testing.assert_allclose(correct, unfused(), atol=1e-5)
    np.testing.assert_allclose(correct, fused(), atol=1e-5)

    algos = (baseline, unfused, fused)
    results = np.zeros([len(algos), REPEAT])
    for rep in range(REPEAT):
        # interleave timing runs to deal with frequency scaling
        for alg in range(len(algos)):
            results[alg, rep] = timeit.timeit(algos[alg], number=TIMES)

    results = np.median(results, axis=1)
    # print(tabulate([gflops / results], headers=[x.__name__ for x in algos]))
    return results[0] / results[1], results[0] / results[2]


def main():
    layers = (8, 4, 2, 1)
    batch_sizes = (128, 64, 32, 16, 8, 4, 2)
    hidden_sizes = (128, 256, 512, 1024, 2048, 4096)
    results = np.zeros([len(layers), len(batch_sizes), 2 + 2 * len(hidden_sizes)])
    for l in range(len(layers)):
        for bs in range(len(batch_sizes)):
            results[l, bs, 0] = layers[l]
            results[l, bs, 1] = batch_sizes[bs]
            for h in range(len(hidden_sizes)):
                unfused, fused = measure(
                    batch_sizes[bs],
                    [hidden_sizes[h]] * (layers[l] + 1))
                results[l, bs, 2 + h] = unfused
                results[l, bs, 2 + len(hidden_sizes) + h] = fused
    rows = results.reshape([-1, 2 + 2 * len(hidden_sizes)])
    headers = ['layers', 'batch_size'] + [str(x) for x in hidden_sizes] + [str(x) for x in hidden_sizes]
    print("overhead reduction".rjust(53).ljust(80) + "pointwise fusion")
    print(tabulate(rows, headers=headers, floatfmt=".2f"))
    pd.DataFrame(rows, columns=headers).to_csv('overhead.csv', index=False)


if __name__ == '__main__':
    with torch.no_grad():
        main()
