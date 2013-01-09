#!/usr/bin/env python

import pycuda.driver as cuda
import parray
import numpy as np
import linalg as la
from kernel_utils import *
import time
from pycuda.tools import dtype_to_ctype
from scikits.cuda.cublas import cublasDnrm2, cublasSnrm2, cublasInit, cublasShutdown

def rnn1(G, q, dt = 1e-6, alpha = 5000, steps = 4000, XOUTPUTSTEPS = None, lamb = 0.0):
    if G.dtype != q.dtype:
        raise TypeError("matrix multiplication must have same dtype")
    
    if np.iscomplexobj(G):
        raise TypeError("RNN currently only solves real types")
    
    if (len(G.shape) != 2) | (len(q.shape) != 2):
        raise TypeError("G, q must both be matrices")
    
    if XOUTPUTSTEPS is None:
        XOUTPUTSTEPS = min(20, steps)
        x_steps = steps / XOUTPUTSTEPS
        fullout = False
    else:
        fullout = True
        x_steps = steps / int(XOUTPUTSTEPS)
        output = parray.empty((XOUTPUTSTEPS, q.size), q.dtype)
    
    dt = float(dt)
    alpha = float(alpha)
    
    
    c = parray.zeros_like(q)
    y = parray.zeros_like(q)
    err_func = get_rnn1_err_func(G.dtype)
    update_func = get_rnn1_update_func(G.dtype)
    
    
    
    if y.dtype == np.float64:
        normfunc = cublasDnrm2
    else:
        normfunc = cublasSnrm2
    
    grid = (6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1)
    
    start = time.time()
    
    for i in range(1,steps+1):
        Gc = la.dot(G, c)
        launch_kernel(err_func, (256,1,1), grid, [y, q, Gc, c, lamb, c.size, 1], prepared = True)
        
        Gc = la.dot(G,y,opa='t')
        launch_kernel(update_func, (256,1,1), grid, [c, Gc, c.size, 1, dt*alpha], prepared = True)
        
        
        if i%x_steps == 0:
            ynorm = normfunc(y.size, y.gpudata, 1);
            print "%d, norm = %.10f, time=%f(ms)" % (i / x_steps, ynorm, (time.time()-start)*1000);
            if fullout:
                cuda.memcpy_dtod(int(output.gpudata)+output.dtype.itemsize*output.ld*int(i/x_steps-1), c.gpudata, c.dtype.itemsize * c.size)
    
    #cuda.memcpy_dtod(q.gpudata, c.gpudata, c.dtype.itemsize*c.size)
    
    if fullout:
        return c,output
    else:
        return c


def rnn1_mpi(G, q, rank, diag, groupsize, col_comm, row_comm, diag_comm, dt = 1e-6, alpha = 5000, steps = 4000, lamb = 0.0):
    from mpi4py import MPI
    from scikits.cuda.cublas import cublasDnrm2, cublasSnrm2
    
    if G.dtype != q.dtype:
        raise TypeError("matrix multiplication must have same dtype")

    if np.iscomplexobj(G):
        raise TypeError("RNN currently only solves real types")

    if (len(G.shape) != 2) | (len(q.shape) != 2):
        raise TypeError("G, q must both be matrices")


    col_rank = col_comm.Get_rank()
    row_rank = row_comm.Get_rank()
    
    XOUTPUTSTEPS = min(20, steps)
    x_steps = steps / XOUTPUTSTEPS
    

    c = np.zeros((G.shape[1], 1), G.dtype)
    d_c = parray.empty((G.shape[1], 1), G.dtype)
    

    dt = float(dt)
    alpha = float(alpha)

    err = np.zeros((G.shape[0],1), G.dtype)
    d_err = parray.empty_like(q)
    

    if d_err.dtype == np.float64:
        normfunc = cublasDnrm2
        mpitype = MPI.DOUBLE
    else:
        normfunc = cublasSnrm2
        mpitype = MPI.FLOAT

    grid = (6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1)

    if diag:
        temp_all = np.zeros((q.size*groupsize, 1), G.dtype)
        d_temp_all = parray.empty_like(temp_all)
        update_func = get_rnn1_err_func(G.dtype)
        update_c_func = get_rnn1_update_func(G.dtype)
        
        ynorm = np.zeros(1, G.dtype)
        recv = np.zeros(1, G.dtype)

    if rank == 0:
        start = MPI.Wtime()
    
    for i in range(1,steps+1):
        col_comm.Bcast(c, root = row_rank)
        d_c.set(c)
        
        d_temp = la.dot(G, d_c)
        temp = d_temp.get()
        
        row_comm.Gather([temp, temp.size, mpitype], [temp_all if diag else None, temp.size, mpitype], root= col_rank)

        if diag:
            d_temp_all.set(temp_all)
            launch_kernel(update_func, (256,1,1), grid, [d_err, q, d_temp_all, d_c, lamb, q.size, groupsize], prepared = True)
            err = d_err.get()
            
            ynorm[0] = normfunc(d_err.size, d_err.gpudata, 1)**2
        
        row_comm.Bcast(err, root = col_rank)
        d_err.set(err)
        d_temp = la.dot(G, d_err, opa='t')
        temp = d_temp.get()
        col_comm.Gather([temp, temp.size, mpitype], [temp_all if diag else None, temp.size, mpitype], root = row_rank)
        
        if diag:
            d_temp_all.set(temp_all)
            launch_kernel(update_c_func, (256,1,1), grid, [d_c, d_temp_all, q.size, groupsize, dt*alpha], prepared = True)
            c = d_c.get()
        
        
            if i%x_steps == 0:
            
                diag_comm.Reduce(ynorm, recv)
                if rank == 0:
                    print "%d, norm = %.10f, time=%f(ms)" % (i / x_steps, np.sqrt(recv[0]), (MPI.Wtime()-start)*1000);
            

    #if diag:
        #cuda.memcpy_dtod(q.gpudata, d_c.gpudata, d_c.dtype.itemsize*c.size)

    return d_c



def rnn1_mpi_tri(G, q, rank, diag, blockrow, blockcol, groupsize, col_comm, row_comm, diag_comm, dt = 1e-6, alpha = 5000, steps = 4000, lamb = 0.0):
    """
    For symmetric matrix using only the upper diagonal blocks.
    """
    
    from mpi4py import MPI
    from scikits.cuda.cublas import cublasDnrm2, cublasSnrm2
    
    if G.dtype != q.dtype:
        raise TypeError("matrix multiplication must have same dtype")
    
    if np.iscomplexobj(G):
        raise TypeError("RNN currently only solves real types")
    
    if (len(G.shape) != 2) | (len(q.shape) != 2):
        raise TypeError("G, q must both be matrices")
    
    col_rank = col_comm.Get_rank()
    row_rank = row_comm.Get_rank()
    
    XOUTPUTSTEPS = min(20, steps)
    x_steps = steps / XOUTPUTSTEPS
    
    c1 = np.zeros((G.shape[1],1), G.dtype)
    c2 = np.zeros((G.shape[0],1), G.dtype)
    
    dt = float(dt)
    alpha = float(alpha)
    
    
    err = np.zeros((G.shape[0],1), G.dtype)
    d_err = parray.empty_like(q)
    
    if d_err.dtype == np.float64:
        normfunc = cublasDnrm2
        mpitype = MPI.DOUBLE
    else:
        normfunc = cublasSnrm2
        mpitype = MPI.FLOAT
    
    grid = (6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1)
    d_c = parray.to_gpu(c1)
    if diag:
        
        Gc_all = np.zeros((q.size*groupsize, 1), G.dtype)
        err_func = get_rnn1_err_func(G.dtype)
        update_func = get_rnn1_update_func(G.dtype)
        if G.dtype == np.float64:
            
            nrmfunc = cublasDnrm2
        
        else:
            nrmfunc = cublasSnrm2
        ynorm = np.zeros(1, G.dtype)
        recv = np.zeros(1, G.dtype)
    
    
    if rank == 0:
        start = MPI.Wtime()
    
    for j in range(1,steps+1):
        MPI.COMM_WORLD.Barrier()
        if diag:
            for i in range(groupsize):
                if i != blockrow:
                    if i < blockrow:
                        col_comm.Send(c1, i, i+200)
                    else:
                        row_comm.Send(c1, i, i+200)
        else:
            row_comm.Recv(c2, col_rank, row_rank + 200)
            col_comm.Recv(c1, row_rank, col_rank + 200)
        
        d_c1 = parray.to_gpu(c1)
        d_c2 = parray.to_gpu(c2)
        
        if diag:
            Gc1 = la.dot(G, d_c1).get()
        else:
            Gc1 = la.dot(G, d_c1).get()
            Gc2 = la.dot(G, d_c2, opa='t').get()
        
        MPI.COMM_WORLD.Barrier()
        
        if diag:
            Gc_all[blockrow * G.shape[0]:(blockrow+1)*G.shape[0]] = Gc1
            for i in range(groupsize):
                
                if i != blockrow:
                    if i < blockrow:
                        col_comm.Recv(Gc_all[i*G.shape[0]:(i+1)*G.shape[0]], i, i+100)
                    else:
                        row_comm.Recv(Gc_all[i*G.shape[0]:(i+1)*G.shape[0]], i, i + 100)
        
        else:
            col_comm.Send(Gc2, row_rank, col_rank+100)
            row_comm.Send(Gc1, col_rank, row_rank+100)
        
        if col_rank == row_rank:
            d_Gc_all = parray.to_gpu(Gc_all)
            launch_kernel(err_func, (256,1,1), grid, [d_err, q, d_Gc_all, d_c, lamb, q.size, groupsize], prepared = True)
            err = d_err.get()
            
            ynorm[0] = nrmfunc(d_err.size, d_err.gpudata, 1)**2
        
        MPI.COMM_WORLD.Barrier()
        
        if diag:
            for i in range(groupsize):
                if i != blockrow:
                    if i < blockrow:
                        col_comm.Send(err, i, i+400)
                    else:
                        row_comm.Send(err, i, i+400)
        else:
            row_comm.Recv(c2, col_rank, row_rank + 400)
            col_comm.Recv(c1, row_rank, col_rank + 400)
        
        
        d_c1 = parray.to_gpu(c1)
        d_c2 = parray.to_gpu(c2)
        
        if diag:
            Gc1 = la.dot(G, d_err).get()
        else:
            Gc1 = la.dot(G, d_c1).get()
            Gc2 = la.dot(G, d_c2,opa='t').get()
        
        MPI.COMM_WORLD.Barrier()
        
        
        if diag:
            Gc_all[blockrow * G.shape[0]:(blockrow+1)*G.shape[0]] = Gc1
            for i in range(groupsize):
                
                if i != blockrow:
                    if i > blockrow:
                        row_comm.Recv(Gc_all[i*G.shape[0]:(i+1)*G.shape[0]], i, i+300)
                    else:
                        col_comm.Recv(Gc_all[i*G.shape[0]:(i+1)*G.shape[0]], i, i + 300)
        
        else:
            col_comm.Send(Gc2, row_rank, col_rank+300)
            row_comm.Send(Gc1, col_rank, row_rank+300)
        
        
        if col_rank == row_rank:
            d_Gc_all = parray.to_gpu(Gc_all)
            launch_kernel(update_func, (256,1,1), grid, [d_c, d_Gc_all, q.size, groupsize, dt*alpha], prepared = True)
            c1 = d_c.get()
            
            if j%x_steps == 0:
                if diag:
                    diag_comm.Reduce(ynorm, recv)
                if rank == 0:
                    print "%d, norm = %.10f, time=%f(ms)" % (j / x_steps, np.sqrt(recv[0]), (MPI.Wtime()-start)*1000);
    
    return d_c



def get_rnn1_update_func(dtype):
    template = """
    __global__ void update( %(type)s* xk, %(type)s* temp, int size, int groupsize, %(type)s L)
    {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        int total_threads = blockDim.x * gridDim.x;
    
        for(int i = tid; i < size; i+=total_threads)
        {
            float tmp = temp[i];
            for(int j = 1; j < groupsize; ++j)
            {
                tmp += temp[i + j * size];
            }
            xk[i] += tmp*L;
        }
    }
    
    """
    
    func = func_compile("update", template % {"type": dtype_to_ctype(dtype)})
    func.prepare([np.intp, np.intp, np.int32, np.int32, dtype.type])
    
    return func

"""
def get_rnn1_err_func(dtype):
    template = ""
    __global__ void err_update( %(type)s* err, %(type)s* b, %(type)s* Ax, int size, int groupsize)
    {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        int total_threads = blockDim.x * gridDim.x;
    
        for(int i = tid; i < size; i+=total_threads)
        {
            float tmp = Ax[i];
            for(int j = 1; j < groupsize; ++j)
            {
                tmp += Ax[i + j * size];
            }
            err[i] = ( b[i]-tmp );
        }

    }
    
    ""
    
    func = func_compile("err_update", template % {"type": dtype_to_ctype(dtype)})
    func.prepare([np.intp, np.intp, np.intp, np.int32, np.int32], (256,1,1))

    return func
"""

def get_rnn1_err_func(dtype):
    template = """
        __global__ void err_update( %(type)s* err, %(type)s* b, %(type)s* Ax, %(type)s* c, %(type)s lambda, int size, int groupsize)
        {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        int total_threads = blockDim.x * gridDim.x;
        
        for(int i = tid; i < size; i+=total_threads)
        {
        float tmp = Ax[i];
        for(int j = 1; j < groupsize; ++j)
        {
        tmp += Ax[i + j * size];
        }
        err[i] = ( b[i]-tmp - lambda*c[i] );
        }
        
        }
        
        """
    
    func = func_compile("err_update", template % {"type": dtype_to_ctype(dtype)})
    func.prepare([np.intp, np.intp, np.intp, np.intp, dtype.type, np.int32, np.int32])
    
    return func


def rnn2(G, q, dt = 1e-6, alpha = 200, lamb = 10, steps = 4000, XOUTPUTSTEPS = None):
    if G.dtype != q.dtype:
        raise TypeError("matrix multiplication must have same dtype")
        
    if np.iscomplexobj(G):
        raise TypeError("RNN currently only solves real types")

    if (len(G.shape) != 2) | (len(q.shape) != 2):
        raise TypeError("G, q must both be matrices")

    if XOUTPUTSTEPS is None:
        XOUTPUTSTEPS = min(20, steps)
        x_steps = steps / XOUTPUTSTEPS
        fullout = False
    else:
        fullout = True
        x_steps = steps / int(XOUTPUTSTEPS)
        output = parray.empty((XOUTPUTSTEPS, q.size), q.dtype)
    
    dt = float(dt)
    alpha = float(alpha)
    lamb = float(lamb)
    
    
    c = parray.zeros_like(q)
    y = parray.zeros_like(q)
    xp = parray.zeros_like(q)
    xm = parray.zeros_like(q)
    
    y_update_func = get_rnn2_y_update_func(G.dtype)
    c_update_func = get_rnn2_c_update_func(G.dtype)
    
    
    
    
    r = parray.empty_like(q)
    
    if r.dtype == np.float64:
        normfunc = cublasDnrm2
    else:
        normfunc = cublasSnrm2

    grid = (6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1)
    
    start = time.time()

    for i in range(1,steps+1):
        Gc = la.dot(G, c)
        launch_kernel(y_update_func, (256,1,1), grid, [y, dt*alpha*lamb, q, Gc, r, c.size, 1], prepared = True)
        
        Gc = alpha*(Gc - la.dot(G,y))
        launch_kernel(c_update_func, (256,1,1), grid, [xp, xm, c, Gc, dt*lamb, c.size, 1], prepared = True)
        
        
        if i%x_steps == 0:
            ynorm = normfunc(r.size, r.gpudata, 1);
            print "%d, norm = %.10f, time=%f(ms)" % (i / x_steps, ynorm, (time.time()-start)*1000);
            if fullout:
                cuda.memcpy_dtod(int(output.gpudata)+output.dtype.itemsize*output.ld*int(i/x_steps-1), c.gpudata, c.dtype.itemsize * c.size)

    #cuda.memcpy_dtod(q.gpudata, c.gpudata, c.dtype.itemsize*c.size)

    if fullout:
        return c,output
    else:
        return c


def rnn2_mpi(G, q, rank, diag, groupsize, col_comm, row_comm, diag_comm, dt = 1e-6, alpha = 200, lamb = 10, steps = 4000):
    from mpi4py import MPI
    from scikits.cuda.cublas import cublasDnrm2, cublasSnrm2
    
    if G.dtype != q.dtype:
        raise TypeError("matrix multiplication must have same dtype")

    if np.iscomplexobj(G):
        raise TypeError("RNN currently only solves real types")

    if (len(G.shape) != 2) | (len(q.shape) != 2):
        raise TypeError("G, q must both be matrices")


    col_rank = col_comm.Get_rank()
    row_rank = row_comm.Get_rank()
    
    XOUTPUTSTEPS = min(20, steps)
    x_steps = steps / XOUTPUTSTEPS
    
    dt = float(dt)
    alpha = float(alpha)
    lamb = float(lamb)
    
    y_update_func = get_rnn2_y_update_func(G.dtype)
    c_update_func = get_rnn2_c_update_func(G.dtype)
    
    c = np.zeros((G.shape[1], 1), G.dtype)
    d_c = parray.empty((G.shape[1], 1), G.dtype)
    
    
    d_y = parray.zeros_like(c)
    y = np.zeros_like(c)
    
    
    xp = parray.zeros_like(q)
    xm = parray.zeros_like(q)
    
    grid = (6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1)
    
    if G.dtype == np.float64:
        nrmfunc = cublasDnrm2
        mpitype = MPI.DOUBLE
        
    else:
        nrmfunc = cublasSnrm2
        mpitype = MPI.FLOAT
    

    if diag:
        Gc_all = np.zeros((q.size*groupsize, 1), G.dtype)
        d_Gc_all = parray.empty_like(Gc_all)
        d_r = parray.empty_like(q)
        
        ynorm = np.zeros(1, G.dtype)
        recv = np.zeros(1, G.dtype)

    if rank == 0:
        start = MPI.Wtime()
        
    for i in range(1,steps+1):
        col_comm.Bcast(c, root = row_rank)
        d_c.set(c)
        
        d_Gc = la.dot(G, d_c)
        Gc = d_Gc.get()
        
        row_comm.Gather([Gc, Gc.size, mpitype], [Gc_all if diag else None, Gc.size, mpitype], root= col_rank)

        if diag:
            d_Gc_all.set(Gc_all)
            launch_kernel(y_update_func, (256,1,1), grid, [d_y, dt*alpha*lamb, q, d_Gc_all, d_r, q.size, groupsize], prepared = True)
            y = d_y.get()
            
        col_comm.Bcast(y, root = row_rank)
        d_y.set(y)
        
        d_Gc = alpha*(d_Gc - la.dot(G,d_y))
        Gc = d_Gc.get()
        row_comm.Gather([Gc, Gc.size, mpitype], [Gc_all if diag else None, Gc.size, mpitype], root= col_rank)
        
        if diag:
            d_Gc_all.set(Gc_all)
            launch_kernel(c_update_func, (256,1,1), grid, [xp, xm, d_c, d_Gc_all, dt*lamb, d_c.size, groupsize], prepared = True)
            c = d_c.get()
            
            ynorm[0] = nrmfunc(d_r.size, d_r.gpudata, 1)**2
        
        if i%x_steps == 0:
            if diag:
                diag_comm.Reduce(ynorm, recv)
            if rank == 0:
                print "%d, norm = %.10f, time=%f(ms)" % (i / x_steps, np.sqrt(recv[0]), (MPI.Wtime()-start)*1000);
            

    #if diag:
        #cuda.memcpy_dtod(q.gpudata, d_c.gpudata, d_c.dtype.itemsize*c.size)

    return d_c




def get_rnn2_y_update_func(dtype):
    rnn2_template = """
    __global__ void rnn2_y_update(%(type)s* d_y, double dt_alpha_lambda, %(type)s* d_q, %(type)s* d_Gc, %(type)s* d_ynorm, int size, int groupsize)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int total_threads = blockDim.x * gridDim.x;
    %(type)s dy;
	for(int i = tid; i < size; i+=total_threads)
	{
        %(type)s tmp = d_Gc[i];
        for(int j = 1; j < groupsize; ++j)
        {
            tmp += d_Gc[i + j * size];
        }
        dy = ( d_q[i] - tmp );
    
		d_y[i] += dt_alpha_lambda * dy;
        d_ynorm[i] = dy;
	}

}
    """
    func = func_compile("rnn2_y_update", rnn2_template % {"type": dtype_to_ctype(dtype)})
    func.prepare([np.intp, np.float64, np.intp, np.intp, np.intp, np.int32, np.int32])
    return func


def get_rnn2_c_update_func(dtype):
    rnn2_template = """
    __global__ void rnn2_c_update(%(type)s* d_xp, %(type)s* d_xm, %(type)s* d_c, %(type)s* d_Gc, double dt_lambda, int size, int groupsize)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int total_threads = blockDim.x * gridDim.x;

	%(type)s xp, xm, Gc;

	for(int i = tid; i < size; i+=total_threads)
	{
		xp = d_xp[i];
		xm = d_xm[i];
        Gc = d_Gc[i];
        for(int j = 1; j < groupsize; ++j)
        {
            Gc += d_Gc[i + j * size];
        }

		xp += dt_lambda * ( fmax(0, xp-Gc) - xp  );
		d_xp[i] = xp;

		xm += dt_lambda * ( fmax(0, xm + Gc) - xm);
		d_xm[i] = xm;
		d_c[i] = xp - xm;
	}
}

    """
    func = func_compile("rnn2_c_update", rnn2_template % {"type": dtype_to_ctype(dtype)})
    func.prepare([np.intp, np.intp, np.intp, np.intp, np.float64, np.int32, np.int32])
    return func



def rnn3(G, q, dt = 1e-6, alpha = 5000, steps = 4000, XOUTPUTSTEPS = None):
    if G.dtype != q.dtype:
        raise TypeError("matrix multiplication must have same dtype")

    if np.iscomplexobj(G):
        raise TypeError("RNN currently only solves real types")

    if (len(G.shape) != 2) | (len(q.shape) != 2):
        raise TypeError("G, q must both be matrices")

    if XOUTPUTSTEPS is None:
        XOUTPUTSTEPS = min(20, steps)
        x_steps = steps / XOUTPUTSTEPS
        fullout = False
    else:
        fullout = True
        x_steps = steps / int(XOUTPUTSTEPS)
        output = parray.empty((XOUTPUTSTEPS, q.size), q.dtype)
        
    cublasInit()

    c = parray.zeros_like(q)
    update_func = get_rnn3_update_func(G.dtype)

    dt = float(dt)
    alpha = float(alpha)

    y = parray.empty_like(q)

    if y.dtype == np.float64:
        normfunc = cublasDnrm2
    else:
        normfunc = cublasSnrm2

    grid = (6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1)
    
    start = time.time()
    for i in range(1,steps+1):
        Gc = la.dot(G, c)
        launch_kernel(update_func, (256,1,1), grid, [c, dt*alpha, q, Gc, y, c.size, 1], prepared = True)
        
        if i%x_steps == 0:
            ynorm = normfunc(y.size, y.gpudata, 1);
            print "%d, norm = %.10f, time=%f(ms)" % (i / x_steps, ynorm, (time.time()-start)*1000);
            if fullout:
                cuda.memcpy_dtod(int(output.gpudata)+output.dtype.itemsize*output.ld*int(i/x_steps-1), c.gpudata, c.dtype.itemsize * c.size)

    #cuda.memcpy_dtod(q.gpudata, c.gpudata, c.dtype.itemsize*c.size)

    cublasShutdown()
    
    if fullout:
        return c,output
    else:
		return c


def rnn3_mpi(G, q, rank, diag, groupsize, col_comm, row_comm, diag_comm, dt = 1e-6, alpha = 5000, steps = 4000):
    from mpi4py import MPI
    from scikits.cuda.cublas import cublasDnrm2, cublasSnrm2
    
    if G.dtype != q.dtype:
        raise TypeError("matrix multiplication must have same dtype")

    if np.iscomplexobj(G):
        raise TypeError("RNN currently only solves real types")

    if (len(G.shape) != 2) | (len(q.shape) != 2):
        raise TypeError("G, q must both be matrices")


    col_rank = col_comm.Get_rank()
    row_rank = row_comm.Get_rank()
    
    XOUTPUTSTEPS = min(20, steps)
    x_steps = steps / XOUTPUTSTEPS
    

    c = np.zeros((G.shape[1], 1), G.dtype)
    d_c = parray.empty((G.shape[1], 1), G.dtype)
    

    dt = float(dt)
    alpha = float(alpha)

    d_y = parray.empty_like(q)

    if d_y.dtype == np.float64:
        normfunc = cublasDnrm2
        mpitype = MPI.DOUBLE
    else:
        normfunc = cublasSnrm2
        mpitype = MPI.FLOAT

    grid = (6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1)

    if diag:
        Gc_all = np.zeros((q.size*groupsize, 1), G.dtype)
        d_Gc_all = parray.empty_like(Gc_all)
        update_func = get_rnn3_update_func(G.dtype)
        if G.dtype == np.float64:
            
            nrmfunc = cublasDnrm2
            
        else:
            nrmfunc = cublasSnrm2
        ynorm = np.zeros(1, G.dtype)
        recv = np.zeros(1, G.dtype)

    if rank == 0:
        start = MPI.Wtime()
    
    for i in range(1,steps+1):
        col_comm.Bcast(c, root = row_rank)
        d_c.set(c)
        
        d_Gc = la.dot(G, d_c)
        Gc = d_Gc.get()
        
        row_comm.Gather([Gc, Gc.size, mpitype], [Gc_all if diag else None, Gc.size, mpitype], root= col_rank)

        if diag:
            d_Gc_all.set(Gc_all)
            launch_kernel(update_func, (256,1,1), grid, [d_c, dt*alpha, q, d_Gc_all, d_y, q.size, groupsize], prepared = True)
            c = d_c.get()
            
            ynorm[0] = nrmfunc(d_y.size, d_y.gpudata, 1)**2
        
        if i%x_steps == 0:
            if diag:
                diag_comm.Reduce(ynorm, recv)
            if rank == 0:
                print "%d, norm = %.10f, time=%f(ms)" % (i / x_steps, np.sqrt(recv[0]), (MPI.Wtime()-start)*1000);
            

    #if diag:
        #cuda.memcpy_dtod(q.gpudata, d_c.gpudata, d_c.dtype.itemsize*c.size)

    return d_c



def rnn3_mpi_tri(G, q, rank, diag, blockrow, blockcol, groupsize, col_comm, row_comm, diag_comm, dt = 1e-6, alpha = 5000, steps = 4000, d_ck = None):
    """
    For symmetric matrix using only the upper diagonal blocks
    """
    
    from mpi4py import MPI
    from scikits.cuda.cublas import cublasDnrm2, cublasSnrm2, cublasInit, cublasShutdown
    
    cublasInit()
    
    if G.dtype != q.dtype:
        raise TypeError("matrix multiplication must have same dtype")
    
    if np.iscomplexobj(G):
        raise TypeError("RNN currently only solves real types")
    
    if (len(G.shape) != 2) | (len(q.shape) != 2):
        raise TypeError("G, q must both be matrices")
    
    
    col_rank = col_comm.Get_rank()
    row_rank = row_comm.Get_rank()
    
    XOUTPUTSTEPS = min(20, steps)
    x_steps = steps / XOUTPUTSTEPS
    
    
    
    if diag:
        if d_ck is None:
            c1 = np.zeros((G.shape[1],1), G.dtype)
            c2 = np.zeros((G.shape[0],1), G.dtype)
        else:
            c1 = d_ck.get()
            c2 = c1.copy()
    else:
        c1 = np.zeros((G.shape[1],1), G.dtype)
        c2 = np.zeros((G.shape[0],1), G.dtype)
    
    
    dt = float(dt)
    alpha = float(alpha)
    
    d_y = parray.empty_like(q)
    
    if d_y.dtype == np.float64:
        normfunc = cublasDnrm2
        mpitype = MPI.DOUBLE
    else:
        normfunc = cublasSnrm2
        mpitype = MPI.FLOAT
    
    grid = (6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1)
    
    if diag:
        Gc_all = np.zeros((q.size*groupsize, 1), G.dtype)
        update_func = get_rnn3_update_func(G.dtype)
        if G.dtype == np.float64:
            
            nrmfunc = cublasDnrm2
        
        else:
            nrmfunc = cublasSnrm2
        ynorm = np.zeros(1, G.dtype)
    recv = np.zeros(1, G.dtype)
    
    
    if rank == 0:
        start = MPI.Wtime()
    
    for j in range(1,steps+1):
        MPI.COMM_WORLD.Barrier()
        if diag:
            for i in range(groupsize):
                if i != blockrow:
                    if i < blockrow:
                        col_comm.Send(c1, i, i+200)
                    else:
                        row_comm.Send(c1, i, i+200)
        else:
            row_comm.Recv(c2, col_rank, row_rank + 200)
            col_comm.Recv(c1, row_rank, col_rank + 200)
        
        d_c1 = parray.to_gpu(c1)
        d_c2 = parray.to_gpu(c2)
        
        if diag:
            Gc1 = la.dot(G, d_c1).get()
        else:
            Gc1 = la.dot(G, d_c1).get()
            Gc2 = la.dot(G, d_c2, opa='t').get()
        
        MPI.COMM_WORLD.Barrier()
        
        if diag:
            Gc_all[blockrow * G.shape[0]:(blockrow+1)*G.shape[0]] = Gc1
            for i in range(groupsize):
                
                if i != blockrow:
                    if i < blockrow:
                        col_comm.Recv(Gc_all[i*G.shape[0]:(i+1)*G.shape[0]], i, i+100)
                    else:
                        row_comm.Recv(Gc_all[i*G.shape[0]:(i+1)*G.shape[0]], i, i + 100)
        
        else:
            row_comm.Send(Gc1, col_rank, row_rank+100)
            col_comm.Send(Gc2, row_rank, col_rank+100)
        
        if col_rank == row_rank:
            d_Gc_all = parray.to_gpu(Gc_all)
            update_func.prepared_call(grid, (256,1,1), d_c1.gpudata, dt*alpha, q.gpudata, d_Gc_all.gpudata, d_y.gpudata, q.size, groupsize)
            c1 = d_c1.get()
            
            ynorm[0] = nrmfunc(d_y.size, d_y.gpudata, 1)**2
        
        if j%x_steps == 0:
            if diag:
                diag_comm.Reduce(ynorm, recv)
            if rank == 0:
                print "%d, norm = %.10f, time=%f(ms)" % (j / x_steps, np.sqrt(recv[0]), (MPI.Wtime()-start)*1000);
        MPI.COMM_WORLD.Bcast(recv, root = 0)
        if np.isnan(recv)[0]:
            return d_c1, True

    cublasShutdown()
    return d_c1, np.isnan(recv)[0]


def get_rnn3_update_func(dtype):
    rnn3_template = """
    __global__ void rnn3_update(%(type)s* d_c, double dt_alpha, %(type)s* d_q, %(type)s* d_Gc, %(type)s* d_ynorm, int size, int groupsize)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int total_threads = blockDim.x * gridDim.x;
        %(type)s dy;
        for(int i = tid; i < size; i+=total_threads)
	{
        %(type)s tmp = d_Gc[i];
        for(int j = 1; j < groupsize; ++j)
        {
            tmp += d_Gc[i + j * size];
        }
        dy = ( d_q[i] - tmp );
        d_c[i] += dt_alpha * dy;
        d_ynorm[i] = dy;
	}
}
    """
    func = func_compile("rnn3_update", rnn3_template % {"type": dtype_to_ctype(dtype)})
    func.prepare([np.intp, np.float64, np.intp, np.intp, np.intp, np.int32, np.int32])
    return func





