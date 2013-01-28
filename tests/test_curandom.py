from ypcutil.importcuda import *
import ypcutil.curand as cr
from pycuda.compiler import SourceModule

M = 1600
state = cr.curand_setup(M, 1000)


code = """

#include "curand_kernel.h"
#include <math.h>

extern "C" {
__global__ void rand_gen(double* output, curandState* devState, int N, int M)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(tid < M)
    {
        curandState localState = devState[tid];
        
        for(int i = 0; i < N; ++i)
        {
            output[M*i + tid] = curand_normal_double(&localState);
        }
    
    }
    

}
}
"""

mod = SourceModule(code, options = ["--ptxas-options=-v"],no_extern_c=True)
func = mod.get_function("rand_gen")
func.prepare([np.intp, np.intp, np.int32, np.int32])

N = 200
A = garray.empty((N,M),np.double)

func.prepared_call(((M-1)/128+1,1),(128,1,1), A.gpudata, state.gpudata, N, M)