#!/usr/bin/env python

import numpy as np

import pycuda.driver as cuda
from pycuda.tools import dtype_to_ctype
from pycuda.compiler import SourceModule
import scikits.cuda.cufft as cufft

import ypcutil.parray as parray

class cufft(object):
    """
    This class is to facilitate taking fft
    of the same type for multiple times 
    """
    def __init__(self, shape, dtype, in_ld, out_ld,
                 forward = True, econ = False, batch_size = 1):
        self.shape = shape
        self.ndim = len(shape)
        self.forward = forward
        self.in_ld = in_ld
        self.out_ld = out_ld
        self.batch_size = batch_size
        (self.intype, self.outtype,
         self.ffttype, self.fftfunc, self.fftdir) = self.gettypes()
        self.setup_dim()
        self.create_plan(self.batch_size)
        
    def transform(self, d_in, d_out):
        assert d_in.dtype == self.intype
        assert d_out.dtype == self.outtype
        
        if self.fftdir is None:
            self.fftfunc(self.plan, int(d_in.gpudata), int(d_out.gpudata))
        else:
            self.fftfunc(slef.plan, int(d_in.gpudata),
                         int(d_out.gpudata), self.fftdir)
    
    def __del__(self):
        if self.planned:
            self.destroy_plan()
        
    def setup_dim(self):
        self.n = np.asarray(self.shape ,np.int32)
        if self.forward:
            self.inembed = np.asarray(self.shape, np.int32)
            self.onembed = np.asarray(self.shape ,np.int32)
            if self.econ:
                self.onembed[-1] = self.onembed[-1]/2+1
        else:
            self.inembed = np.asarray(self.shape, np.int32)
            self.onembed = np.asarray(self.shape, np.int32)
            if self.econ:
                self.inembed[-1] = self.inembed[-1]/2+1
        
    def destroy_plan(self):
        cufft.cufftDestroy(self.plan)
        self.planned = False

    def create_plan(self, batchsize):
        self.plan = cufft.cufftPlanMany(
            self.ndim, self.n.ctypes.data,
            self.inembed.ctypes.data, 1, self.in_ld,
            self.onembed.ctypes.data, 1, self.out_ld,
            self.ffttype, batch_size)
        self.planned = True

    def gettypes(self):
        dtype = self.dtype
        forward = self.forward
        econ = self.econ
        single = parray.issingle(dtype)
        if issubclass(dtype, np.complexfloating):
            intype = dtype
            if forward:
                outtype = dtype
                if single:
                    ffttype = cufft.CUFFT_C2C
                    fftfunc = cufft.cufftExecC2C
                else:
                    ffttype = cufft.CUFFT_Z2Z
                    fftfunc = cufft.cufftExecZ2Z
                fftdir = cufft.CUFFT_FORWARD
            else:
                if econ:
                    outtype = parray.complextofloat(dtype)
                    if single:
                        ffttype = cufft.CUFFT_C2R
                        fftfunc = cufft.cufftExecC2R
                    else:
                        ffttype = cufft.CUFFT_Z2D
                        fftfunc = cufft.cufftExecZ2D
                    fftdir = None
                else:
                    outtype = dtype
                    if single:
                        ffttype = cufft.CUFFT_C2C
                        fftfunc = cufft.cufftExecC2C
                    else:
                        ffttype = cufft.CUFFT_Z2Z
                        fftfunc = cufft.cufftExecZ2Z
                    fftdir = cufft.CUFFT_INVERSE
        else:
            intype = dtype
            if not forward:
                forward = True
                from warnings import warn
                warn("real input will be forward transform by default")
            outtype = parray.floattocomplex(dtype)
            if single:
                ffttype = cufft.CUFFT_R2C
                fftfunc = cufft.cufftExecR2C
            else:
                ffttype = cufft.CUFFT_D2Z
                fftfunc = cufft.cufftExecD2Z
            fftdir = None
        return intype, outtype, ffttype, fftfunc, fftdir
        
    
def fft(d_A):
    """
    can accept only 2D array
    1D FFT for each row
    """
    assert len(d_A.shape) <= 2
    A = d_A
    reshaped = False
    if any([b == 1 for b in A.shape]):
        total_inputs = 1
        size = max(A.shape)
        if A.shape[1] == 1:
            A = d_A.reshape((1, size))
            reshaped = True
    else:
        total_inputs = shape[0]
        size = A.shape[1]
    
    outdtype = parray.floattocomplex(A.dtype)        
    d_output = parray.empty((total_inputs, size), outdtype)
    
    batch_size = min(total_inputs, 128)
    # TODO: check if d_output.ld is correct for vectors
    plan = cufft(size, A.dtype, A.ld, d_output.ld,
                 forward = True, econ = False,
                 batch_size = batch_size)
    for i in range(0, total_inputs, batch_size):
        ntransform = min(batch_size, total_inputs-i)
        if ntransform != batch_size:
            del plan
            plan = cufft(size, A.dtype, A.ld, d_output.ld,
                         forward = True, econ = False,
                         batch_size = ntransform)
        plan.transform(A[i:i+ntransform], d_output[i:i+ntransform])
    del plan
    if parray.isrealobj(A):
        pad_func = get_1d_pad_func(outdtype)
        pad_func.prepared_call(
            (6*cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1),
            (256, 1, 1), d_output.gpudata, d_output.ld,
            size, total_inputs)
    return d_output.reshape(d_A.shape) if reshaped else d_output

def ifft(d_A, econ = False, even_size = None):
    """
    can accept only 2D array
    1D Inverse FFT for each row
    If econ is True, consider the input A has only half of the fft coeff.
    """
    assert len(d_A.shape) <= 2
    A = d_A
    reshaped = False
    if any([b == 1 for b in A.shape]):
        total_inputs = 1
        if econ:
            if even_size is None:
                even_size = check_even_econ_1d(a, max(A.shape))
            size = (max(A.shape)-1)*2 if even_size else (max(A.shape)-1)*2+1
        else:
            size = max(A.shape)
        if A.shape[1] == 1:
            A = d_A.reshape((1, max(A.shape)))
            reshaped = True
    else:
        total_inputs = max(A.shape)
        if econ:
            if even_size is None:
                even_size = check_even_econ_1d(a, A.shape[1])
            size = (A.shape[1]-1)*2 if even_size else (A.shape[1]-1)*2+1
        else:
            size = A.shape[1]
    
    outdtype = parray.complextofloat(A.dtype) if econ else A.dtype        
    d_output = parray.empty((total_inputs, size), outdtype)
    
    batch_size = min(total_inputs, 128)
    # TODO: check if d_output.ld is correct for vectors
    plan = cufft(size, A.dtype, A.ld, d_output.ld,
                 forward = False, econ = econ,
                 batch_size = batch_size)
    for i in range(0, total_inputs, batch_size):
        ntransform = min(batch_size, total_inputs-i)
        if ntransform != batch_size:
            del plan
            plan = cufft(size, A.dtype, A.ld, d_output.ld,
                         forward = False, econ = econ,
                         batch_size = ntransform)
        plan.transform(A[i:i+ntransform], d_output[i:i+ntransform])
    del plan
    return d_output.reshape(d_A.shape) if reshaped else d_output

def fft2(A):
    pass

def ifft2(A):
    pass
    

def check_even_econ_1d(A, size):
    """
    Check whether the fft size of A is even or odd.
    
    Parameters
    ----------
    A: parray.PitchArray
       The array A contains the reduced storage
       of fft for some real sequences
    size: int
          The size of reduced storage
          
    Returns
    -------
    out: bool
         True if the size is even, i.e. the last entry of fft is real
         False if the size is odd, i.e. the last entry of fft is complex
    """
    a = np.empty(1, A.dtype)
    cuda.memcpy_dtoh(a, int(int(A.gpudata) + A.dtype.itemsize*(size-1)))
    return a.imag == 0.0


def get_1d_pad_func(dtype):
    """
    Assumes that the array is already allocated and the half of
    the entry is filled with half of the fft results
    """
    template = """
__global__ void
get_1d_pad_kernel(%(type)s input, int ld, int fftsize, int batch)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = blockDim.x * gridDim.x;
    int entry_per_row = (fftsize>>1);
    int total_entry = entry_per_row*batch;
    int row, col;
    
    for(int i = tid; i < total_entry; i += total_threads)
    {
        row = i / entry_per_row;
        col = i %% entry_per_row;
        input[row*ld + fftsize-col] = conj(input[row*ld + col+1]);
    }
}

    """
    mod = SourceModule(template % {"type": dtype_to_ctype(dtype)},
                       options = ["--ptxas-options=-v"])
    func = mod.get_function('get_1d_pad_kernel')
    func.prepare([np.intp, np.int32, np.int32, np.int32])
    #grid = (6*cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1)    
    #block = (256, 1, 1)
    return func
    

