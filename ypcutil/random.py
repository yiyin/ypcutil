#!/usr/bin/env python

from datetime import datetime

import numpy as np

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.tools import dtype_to_ctype, context_dependent_memoize
import parray
import curand

def randn(shape, dtype = np.double, mean = 0.0, std = 1.0, seed = None):
    grid = (8*cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1)
    block = (128, 1, 1)
    state = curand.curand_setup(
            grid[0]*block[0],
            seed = seed if seed is not None else \
            int(''.join([str(i) for i in datetime.now().timetuple()[:-1]])))
    func = get_randn_func(dtype)
    result = parray.empty(shape, dtype = dtype)
    func.prepared_call(grid, block, state.gpudata,
                       result.gpudata, result.mem_size,
                       mean, std)
    del func, state
    curand.reset_stack_limit()
    return result

@context_dependent_memoize
def get_randn_func(dtype):
    template = """
#include "curand_kernel.h"
extern "C" {
__global__ void
generate_randn(curandStateXORWOW_t* state, %(type)s* result,
               int size, %(type)s mean, %(type)s std)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = blockDim.x * gridDim.x;
    
    curandStateXORWOW_t local_state;
    if(tid < size)
    {
        local_state = state[tid];
    }
    for(int i = tid; i < size; i+= total_threads)
    {
        result[i] = curand_normal%(double)s(&local_state)*std + mean;
    }
    if(tid < size)
    {
        state[tid] = local_state;
    }
}
}
"""
    mod = SourceModule(template % {
                       "type": dtype_to_ctype(dtype),
                       "double": "_double" if dtype == np.double else ''},
                       no_extern_c = True)
    func = mod.get_function("generate_randn")
    func.prepare('PPi'+np.dtype(dtype).char*2)
    return func

def rand(shape, dtype = np.double, min = 0.0, max = 1.0, seed = None):
    grid = (8*cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1)
    block = (128, 1, 1)
    state = curand.curand_setup(
            grid[0]*block[0],
            seed = seed if seed is not None else \
            int(''.join([str(i) for i in datetime.now().timetuple()[:-1]])))
    func = get_rand_func(dtype)
    result = parray.empty(shape, dtype = dtype)
    func.prepared_call(grid, block, state.gpudata,
                       result.gpudata, result.mem_size,
                       min, max-min)
    del state, func
    curand.reset_stack_limit()
    return result

@context_dependent_memoize
def get_rand_func(dtype):
    template = """
#include "curand_kernel.h"
extern "C" {
__global__ void
generate_rand(curandStateXORWOW_t* state, %(type)s* result,
              int size, %(type)s minimum, %(type)s range)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = blockDim.x * gridDim.x;
    
    curandStateXORWOW_t local_state;
    if(tid < size)
    {
        local_state = state[tid];
    }
    for(int i = tid; i < size; i+= total_threads)
    {
        result[i] = curand_uniform%(double)s(&local_state)*range + minimum;
    }
    if(tid < size)
    {
        state[tid] = local_state;
    }
}
}
"""
    mod = SourceModule(template % {
                       "type": dtype_to_ctype(dtype),
                       "double": "_double" if dtype == np.double else ''},
                       no_extern_c = True)
    func = mod.get_function("generate_rand")
    func.prepare('PPi'+np.dtype(dtype).char*2)
    return func
