#!/usr/bin/env python


import scikits.cuda.cula as cula
import numpy as np
import parray
import pycuda.driver as cuda
import pycuda.gpuarray as garray
import scikits.cuda.cublas as cublas
from kernel_utils import *
from pycuda.tools import dtype_to_ctype

""" assuming row major storage as in PitchArray """


def dot(A, B, opa = 'n', opb = 'n', C = None, Cstart = None):
    """
    returns multiplication of two PitchArray A and B
    if C is specified, use the memory in C.
    Specified C must have the same leading dimension as that of the result and
    the other dimension must be bigger or equal to that of the result.

    opa: operation on A
         'n' or 'N': use A itself
         't' or 'T': use transpose of A
         'c' or 'C': use conjugate transpose of A
         
    opb: operation on B
         'n' or 'N': use B itself
         't' or 'T': use transpose of B
         'c' or 'C': use conjugate transpose of B

    """
    
    if A.dtype != B.dtype:
        raise TypeError("matrix multiplication must have same dtype")

    if (len(A.shape) != 2) | (len(B.shape) != 2):
        raise TypeError("A, B must both be matrices")


    if opa in ['n', 'N']:
        m,n = A.shape
    elif opa in ['t','T', 'c','C']:
        n,m = A.shape
    else:
        raise ValueError("unknown value assigned to opa")

    if opb in ['n', 'N']:
        k,l = B.shape
    elif opb in ['t','T', 'c','C']:
        l,k = B.shape
    else:
        raise ValueError("unknown value assigned to opa")

    if (k != n) | (0 in [m,n,l]):
        raise ValueError("matrix dimension mismatch, (%d,%d) with (%d,%d)" % (m,n,k,l))

    dtype = A.dtype
    if dtype in [np.float32, np.float64]:
        if opb in ['c', 'C']:
            opb = 't'

        if opa in ['c', 'C']:
            opa = 't'
        


    
    if dtype == np.float64:
        tp = 'cublas.cublasD'
        complex_type = False
    elif dtype == np.complex128:
        tp = 'cublas.cublasZ'
        complex_type = True
    elif dtype == np.float32:
        tp = 'cublas.cublasS'
        complex_type = False
    elif dtype == np.complex64:
        tp = 'cublas.cublasC'
        complex_type = True

    if C is None:
        C = parray.empty((m,l), dtype)
        Cstart = 0
    else:
        if Cstart is None:
            Cstart = 0
        
        if C.shape[1] != l:
            raise AttributeError("shape of the provided result array " + C.shape.__str__() + " does not match intended result " + (m,l).__str__())
        if C.shape[0] < m + Cstart:
            raise AttributeError("shape of the provided result array " + C.shape.__str__() + " does not match intended result " + (m,l).__str__())
            
    conjA = False
    conjB = False
    conjC = False
    
    itemsize = C.dtype.itemsize
    
    if m == 1:
        if n == 1:
            alpha = A.get()[0,0]
            cuda.memcpy_dtod(int(C.gpudata) + Cstart * itemsize , B.gpudata, l*dtype.itemsize)
            if opa in ['c','C']:
                alpha = np.conj(alpha)
            if opb in ['c', 'C']:
                C.conj()
            
            func = tp+"scal(l, alpha, int(C.gpudata) + Cstart * itemsize, 1)"
        else:
            if l > 1:
                alpha = 1.0
                beta = 0.0
                if opa in ['c','C']:
                    A.conj()
                    conjA = True

                func = tp+"gemv('"+opb+"',B.shape[1], B.shape[0], alpha, B.gpudata, B.ld, A.gpudata, 1, beta, int(C.gpudata) + Cstart * itemsize * C.ld, 1)"
            else:
                if opa in ['c','C']:
                    if opb in ['c', 'C']:
                        func = "C.set(np.array(" + tp + "dotu(n, A.gpudata, 1, B.gpudata, 1)" +").conj())"
                    else:
                        func = "C.set(np.array(" + tp + "dotc(n, A.gpudata, 1, B.gpudata, 1)" +"))"
                elif opb in ['c', 'C']:
                    func = "C.set(np.array(" + tp + "dotc(n, B.gpudata, 1, A.gpudata, 1)" +"))"
                else:
                    if complex_type:
                        func = "C.set(np.array(" + tp + "dotu(n, A.gpudata, 1, B.gpudata, 1)" +"))"
                    else:
                        func = "C.set(np.array(" + tp + "dot(n, A.gpudata, 1, B.gpudata, 1)" +"))"
    else:#m!=1
        if n == 1:
            if l == 1:
                alpha = B.get()[0,0]
                cuda.memcpy_dtod(int(C.gpudata) + Cstart * itemsize, B.gpudata, l*dtype.itemsize)
                if opa in ['c','C']:
                    alpha = np.conj(alpha)
                if opb in ['c', 'C']:
                    C.conj()
                func = tp+"scal(l, alpha, int(C.gpudata) + Cstart * itemsize,1)"
            else:
                C.fill(0)
                if opa in ['c','C']:
                    if opb in ['c', 'C']:
                        B.conj()
                        conjB = True
                        func =tp + "gerc(l, m, 1.0, B.gpudata, 1, A.gpudata, 1, int(C.gpudata) + Cstart * itemsize * C.ld, C.ld)"
                    else:
                        func = tp + "gerc(l, m, 1.0, B.gpudata, 1, A.gpudata, 1, int(C.gpudata) + Cstart * itemsize * C.ld, C.ld)"
                elif opb in ['c', 'C']:
                    B.conj()
                    conjB = True
                    func =tp + "geru(l, m, 1.0, B.gpudata, 1, A.gpudata, 1, int(C.gpudata) + Cstart * itemsize * C.ld, C.ld)"
                else:
                    if complex_type:
                        func = tp + "geru(l, m, 1.0, B.gpudata, 1, A.gpudata, 1, int(C.gpudata) + Cstart * itemsize * C.ld, C.ld)" 
                    else:
                        func = tp + "ger(l, m, 1.0, B.gpudata, 1, A.gpudata, 1, int(C.gpudata) + Cstart * itemsize * C.ld, C.ld)" 
        else:
            if l == 1:
                if opb in ['c', 'C']:
                    if opa in ['c', 'C']:
                        conjC = True
                        func = tp + "gemv('n', A.shape[1], A.shape[0], 1.0, A.gpudata, A.ld, B.gpudata, 1, 0.0, int(C.gpudata) + Cstart * itemsize * C.ld, 1)"
                    else:
                        B.conj()
                        conjB = True
                        if opa in ['t', 'T']:
                            opa = 'n'
                        else:
                            opa = 't'
                        
                        func = tp + "gemv('" + opa + "', A.shape[1], A.shape[0], 1.0, A.gpudata, A.ld, B.gpudata, 1, 0.0, int(C.gpudata) + Cstart * itemsize * C.ld, 1)"
                        
                else:
                    if opa in ['c', 'C']:
                        B.conj()
                        conjB = True
                        conjC = True
                        func = tp + "gemv('n', A.shape[1], A.shape[0], 1.0, A.gpudata, A.ld, B.gpudata, 1, 0.0, int(C.gpudata) + Cstart * itemsize * C.ld, 1)"
                    else:
                        if opa in ['t', 'T']:
                            opa = 'n'
                        else:
                            opa = 't'
                        
                        func = tp + "gemv('" + opa + "', A.shape[1], A.shape[0], 1.0, A.gpudata, A.ld, B.gpudata, 1, 0.0, int(C.gpudata) + Cstart * itemsize * C.ld, 1)"
            else:
                func = tp+"gemm('" + opb + "','" + opa + "', l, m, k, 1.0, B.gpudata, B.ld, A.gpudata, A.ld, 0.0, int(C.gpudata) + Cstart * itemsize * C.ld, C.ld)"
         

        
    try:
        eval(func)
    except cublas.cublasNotInitialized:
        cublas.cublasInit()
        eval(func)

    
    if conjC:
        C.conj()

    if conjA:
        A.conj()

    if conjB:
        B.conj()
    
    return C
        

def svd(G, compute_u = 1, compute_v = 1, econ = False):
    """
    compute Singular Value Decompositon of G
    G = U*(diag(S))*V

    arguments
    G:  PitchArray, GPUArray or numpy.ndarray of shape (m,n)
        if G is GPUArray or PitchArray, its gpudata will be destroyed after calling the function
    compute_u: whether return U matrix or not
    compute_v: whether return V matrix or not
    econ:   return economical matrix

    output:
    U: as U in G = U*(diag(S))*V, if econ, returns the first min(m,n) columns of U
    S: a row vector containing all singular values
        with descending order
    V: as V in G = U*(diag(S))*V, if econ, returns the first min(m,n) rows of V

    order of output:
    always obeys the order U,S,V
    e.g.
    S = svd(G, compute_u = 0, compute_v = 0)
    U,S = svd(G, compute_u = 1, compute_v = 0)
    S,V = svd(G, compute_u = 0, compute_v = 1)
    U,S,V = svd(G, compute_u = 1, compute_v = 1)
    
    """
    
    if G.__class__ is not parray.PitchArray:
        if G.__class__ is garray.GPUArray:
            h_G = G.get()
            del G.gpudata
            A= parray.to_gpu(h_G)
        elif G.__class__ is np.ndarray:
            A = parray.to_gpu(G)
        else:
            raise TypeError("G must be either parray, or GPUArray or ndarray")
    else:
        A = G
    
    real_dtype = np.dtype(np.float32)
    if A.dtype == np.complex64:
        svd_func = cula.culaDeviceCgesvd        
    elif A.dtype == np.float32:
        svd_func = cula.culaDeviceSgesvd
    else:
        if cula._libcula_toolkit == 'premium':
            if A.dtype == np.complex128:
                svd_func = cula.culaDeviceZgesvd
            elif A.dtype == np.float64:
                svd_func = cula.culaDeviceDgesvd
            else:
                raise ValueError('unsupported type')
            real_dtype = np.dtype(np.float64)
        else:
            raise TypeError('does not support premium double precision svd')
    
    if len(A.shape) != 2:
        raise TypeError("svd only works on 2D matrix")

    
    S = parray.empty(min(A.shape), real_dtype)

    cula.culaInitialize()
    
    if compute_u:
        if compute_v:
            if econ:
                if A.shape[1] <= A.shape[0]:
                    jobu = 'A'
                    jobvt = 'O'
                    V = parray.empty((A.shape[1], A.shape[1]), A.dtype)
                    svd_func(jobu, jobvt, A.shape[1], A.shape[0], A.gpudata, A.ld, S.gpudata, V.gpudata, V.ld, 1, 1)
                    cula.culaShutdown()
                    return A,S,V
                else:
                    jobu = 'O'
                    jobvt = 'A'
                    U = parray.empty((A.shape[0], A.shape[0]), A.dtype)
                    svd_func(jobu, jobvt, A.shape[1], A.shape[0], A.gpudata, A.ld, S.gpudata, 1, 1, U.gpudata, U.ld)
                    cula.culaShutdown()
                    return U,S,A
            else:
                if A.shape[1] <= A.shape[0]:
                    jobu = 'O'
                    jobvt = 'A'
                    U = parray.empty((A.shape[0], A.shape[0]), A.dtype)
                    svd_func(jobu, jobvt, A.shape[1], A.shape[0], A.gpudata, A.ld, S.gpudata, 1, 1, U.gpudata, U.ld)
                    cula.culaShutdown()
                    A.shape = (A.shape[1],A.shape[1])
                    return U,S,A
                else:
                    jobu = 'A'
                    jobvt = 'O'
                    V = parray.empty((A.shape[1], A.shape[1]), A.dtype)
                    svd_func(jobu, jobvt, A.shape[1], A.shape[0], A.gpudata, A.ld, S.gpudata, V.gpudata, V.ld, 1, 1)
                    A.shape = (A.shape[0], A.shape[0])
                    cula.culaShutdown()
                    return A,S,V
        
        else:
            if econ | (A.shape[1] >= A.shape[0]):
                jobu = 'N'
                jobvt = 'O'
                svd_func(jobu, jobvt, A.shape[1], A.shape[0], A.gpudata, A.ld, S.gpudata, 1, 1, 1, 1)
                if (A.shape[1] > A.shape[0]):
                    A.shape = (A.shape[0], A.shape[0])
                cula.culaShutdown()
                return A,S
            else:
                jobu = 'N'
                jobvt = 'A'
                U = parray.empty((A.shape[0],A.shape[0]),A.dtype)
                svd_func(jobu, jobvt, A.shape[1], A.shape[0], A.gpudata, A.ld, S.gpudata, 1, 1, U.gpudata, U.ld)
                cula.culaShutdown()
                return U,S
    else:
        if compute_v:
            if econ | (A.shape[1] <= A.shape[0]):
                jobu = 'O'
                jobvt = 'N'
                svd_func(jobu, jobvt, A.shape[1], A.shape[0], A.gpudata, A.ld, S.gpudata, 1, 1, 1, 1)
                if (A.shape[1] < A.shape[0]):
                    A.shape = (A.shape[1], A.shape[1])
                cula.culaShutdown()
                return S,A
            else:
                jobu = 'A'
                jobvt = 'N'
                V = parray.empty((A.shape[1],A.shape[1]),A.dtype)
                svd_func(jobu, jobvt, A.shape[1], A.shape[0], A.gpudata, A.ld, S.gpudata, V.gpudata, V.ld, 1, 1)
                cula.culaShutdown()
                return S,V
        
        else:
            jobu = 'N'
            jobvt = 'N'
            svd_func(jobu, jobvt, A.shape[1], A.shape[0], A.gpudata, A.ld, S.gpudata, 1, 1, 1, 1)
            cula.culaShutdown()
            return S


def pinv(G, rcond = 1e-4):
    """
    computes the Moore-Penrose pseudo-inversion using SVD method

    input:
    G: PitchArray, GPUArray or numpy.ndarray of shape (m,n)
        if G is GPUArray or PitchArray, its gpudata will be destroyed after calling the function
    rcond:  Cutoff for small singular values.
            Singular values smaller (in modulus) than
            `rcond` * largest_singular_value (again, in modulus)
            are set to zero
    """
    
    U,S,V = svd(G, econ=1)
    rcond = S.dtype.type(rcond)

    sv_func = get_svinv_kernel(S.dtype, V.dtype)
    launch_kernel(sv_func, (256,1,1), (S.size, 1), [S, V, V.ld, V.shape[1], rcond])
    return dot(V, U, opa='c', opb='c')

def solve_eq(G, q, rcond = 1e-4):
    """
    solves Gc = q using pseudo-inversion

    input:
    G: PitchArray
       Its gpudata will be destroyed after calling the function
    q: PitchArray
    dtype of G and q must b the same

    rcond:  Cutoff for small singular values.
            Singular values smaller (in modulus) than
            `rcond` * largest_singular_value (again, in modulus)
            are set to zero

    return:
    solution c
    """
    if G.dtype != q.dtype:
        raise TypeError("G,q must be of the same dtype")

    if G.shape[0] != q.shape[0]:
        raise ValueError("number of columns of G must be the same of size of q")


    U,S,V = svd(G, econ=1)
    
    qq = dot(U, q, opa='c')
    rcond = S.dtype.type(rcond)
    
    sq_func = get_sq_kernel(S.dtype, qq.dtype)
    launch_kernel(sq_func, (256,1,1), (6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1), [S, qq, rcond, S.size])
    
    result = dot(V, qq, opa='c')
    return result

    
    
def solve_eq_sym(G, q, rcond = 1e-4):
    """
    solves Gc = q using pseudo-inversion via SVD, with G a self-adjoint matrix

    input:
    G: PitchArray, a self-adjoint matrix
       Its gpudata will be destroyed after calling the function
    q: PitchArray
    dtype of G and q must b the same

    rcond:  Cutoff for small singular values.
            Singular values smaller (in modulus) than
            `rcond` * largest_singular_value (again, in modulus)
            are set to zero

    return:
    solution c
    """
    cublas.cublasInit()
    if G.dtype != q.dtype:
        raise TypeError("G,q must be of the same dtype")
    
    if G.shape[0] != G.shape[1]:
        raise ValueError("G must be square matrix")

    if G.shape[1] != q.size:
        raise ValueError("number of columns of G must be the same of size of q")

    U,S = svd(G, compute_v=0)
    qq = dot(U, q, opa='c')
    rcond = S.dtype.type(rcond)
    
    sq_func = get_sq_kernel(S.dtype, qq.dtype)
    launch_kernel(sq_func, (256,1,1), (6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1), [S, qq, rcond, S.size])
    
    result = dot(U, qq)
    cublas.cublasShutdown()
    
    return result

    
def eig_sym(G, compute_z = 1, uplo = 'U'):
    """
    compute Eigenvalue Decompositon of a symmetric or Hermitian matrix G
    G = V D V^{*}

    arguments
    G:  PitchArray, GPUArray or numpy.ndarray
        if G is GPUArray or PitchArray, its gpudata will be destroyed after calling the function
    compute_z: whether return eigenvectors (=1) or not (=0)
    uplo: 'U' or 'u' assumes the entries of G are stored in upper triangular part,
          lower off diagonal triangular part is not referenced
          'L' or 'l' assumes the entries of G are stored in lower triangular part,
          upper off diagonal triangular part is not referenced

    output:
    D: a row vector containing all eigenvalues with ascending order
    V: if compute_z = 1, jth column of V contains orthonormal eigenvector associated
       with jth eigenvalue

    examples:.
    D = eig_sym(G, compute_z = 0)
    D,V = eig_sym(G, compute_z = 1)
    
    """
    if cula._libcula_toolkit != 'premium':
        raise ValueError("eigenvalue decomposition is only supported in premium version of CULA")

    if G.__class__ is not parray.PitchArray:
        if G.__class__ is garray.GPUArray:
            h_G = G.get()
            del G.gpudata
            A= parray.to_gpu(h_G)
        elif G.__class__ is np.ndarray:
            A = parray.to_gpu(G)
        else:
            raise TypeError("G must be either parray, or GPUArray or ndarray")
    else:
        A = G
    
    if len(A.shape) != 2:
        raise TypeError("eig only works on 2D matrix")
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("G must be square matrix")

    if uplo in ['u', 'U']:
        uplo = 'L'
    elif uplo in ['l', 'L']:
        uplo = 'U'
    else:
        raise ValueError("uplo must be 'U' or 'L'")
    
    
    
    real_dtype = np.dtype(np.float32)
    if A.dtype == np.complex64:
        eig_func = cula.culaDeviceCheev        
    elif A.dtype == np.float32:
        eig_func = cula.culaDeviceSsyev
    else:
        if A.dtype == np.complex128:
            eig_func = cula.culaDeviceZheev
        elif A.dtype == np.float64:
            eig_func = cula.culaDeviceDsyev
        else:
            raise ValueError('unsupported type')
        real_dtype = np.dtype(np.float64)
    
    
    
    D = parray.empty(A.shape[0], real_dtype)
    
    cula.culaInitialize()

    
    if compute_z:
        jobz = 'V'
    else:
        jobz = 'N'
    
    
    eig_func(jobz, uplo, A.shape[0], A.gpudata, A.ld, D.gpudata)

    cula.culaShutdown()
    
    if compute_z:
        return D, A.conj().T()
    else:
        return D
        
    


def FISTA_l2(A, b, L = 1, steps=5000):
    import time
    
    xk = parray.zeros_like(b)
    t_k = 1
    t_km1 = 1
    xkm1 = xk.copy()
    
    #c = dot(A,b,opa='t')
    if b.dtype == np.float64:
        normfunc = cublas.cublasDnrm2
    else:
        normfunc = cublas.cublasSnrm2
    
    x_steps = steps/20
    
    start = time.time()
    
    for i in range(1,steps+1):
        yk = xk + ((t_km1-1)/t_k)*(xk-xkm1)
        err = dot(A,yk) - b
        temp = dot(A,err, opa='t')
        xkm1 = xk.copy()
        xk = yk - temp / L
        
        t_kp1 = 0.5*(1+np.sqrt(1+4*t_k**2))
        t_km1 = t_k
        t_k = t_kp1
        
        
        if i%x_steps == 0:
            ynorm = normfunc(err.size, err.gpudata, 1)
            print "%d, norm = %.10f, time=%f(ms)" % (i / x_steps, ynorm, (time.time()-start)*1000)
    
    return xk
        
    
def fista_l2_mpi(A, b, rank, diag, groupsize, col_comm, row_comm, diag_comm, L = 1, steps = 4000):
    from mpi4py import MPI
    
    if A.dtype != b.dtype:
        raise TypeError("matrix multiplication must have same dtype")


    if (len(A.shape) != 2) | (len(b.shape) != 2):
        raise TypeError("G, q must both be matrices")


    col_rank = col_comm.Get_rank()
    row_rank = row_comm.Get_rank()
    
    XOUTPUTSTEPS = min(20, steps)
    x_steps = steps / XOUTPUTSTEPS
    
    L = float(L)
    
    d_xk = parray.zeros_like(b)
    t_k = 1.
    t_km1 = 1.
    d_xkm1 = d_xk.copy()
    
    grid = (6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1)
    
    d_err = parray.empty_like(b)

    if d_err.dtype == np.float64:
        normfunc = cublas.cublasDnrm2
        mpitype = MPI.DOUBLE
    else:
        normfunc = cublas.cublasSnrm2
        mpitype = MPI.FLOAT
    
    
    yk = np.zeros((A.shape[1], 1), A.dtype)
    d_yk = parray.empty_like(yk)
    
    err = np.zeros((A.shape[0],1), A.dtype)
    d_err = parray.empty_like(err)
    
    if diag:
        temp_all = np.zeros((b.size*groupsize, 1), A.dtype)
        d_temp_all = parray.empty_like(temp_all)
        update_err_func = get_fista_err_func(A.dtype)
        update_xk_func = get_fista_update_func(A.dtype)
        
        ynorm = np.zeros(1, A.dtype)
        recv = np.zeros(1, A.dtype)

    if rank == 0:
        start = MPI.Wtime()
        
    for i in range(1,steps+1):
        if diag:
            d_yk = d_xk + ((t_km1-1)/t_k)*(d_xk-d_xkm1)
            yk = d_yk.get()
        
        col_comm.Bcast(yk, root = row_rank)
        d_yk.set(yk)
        d_temp = dot(A, d_yk)
        temp = d_temp.get()
        
        row_comm.Gather([temp, temp.size, mpitype], [temp_all if diag else None, temp.size, mpitype], root = col_rank)
        
        if diag:
            d_temp_all.set(temp_all)
            launch_kernel(update_err_func, (256,1,1), grid, [d_err, b, d_temp_all, b.size, groupsize], prepared = True)
            err = d_err.get()
            ynorm[0] = normfunc(d_err.size, d_err.gpudata, 1)**2
        
        row_comm.Bcast(err, root = col_rank)
        d_err.set(err)
        d_temp = dot(A, d_err, opa='t')
        temp = d_temp.get()
        col_comm.Gather([temp, temp.size, mpitype], [temp_all if diag else None, temp.size, mpitype], root = row_rank)
        
        if diag:
            d_temp_all.set(temp_all)
            #d_xkm1 = d_xk.copy()
            cuda.memcpy_dtod(d_xkm1.gpudata, d_xk.gpudata, d_xk.size*d_xk.dtype.itemsize)
            launch_kernel(update_xk_func, (256,1,1), grid, [d_xk, d_yk, d_temp_all, b.size, groupsize, L], prepared = True)
            t_kp1 = 0.5*(1+np.sqrt(1+4*t_k**2))
            t_km1 = t_k
            t_k = t_kp1
        
            if i % x_steps == 0:
                diag_comm.Reduce(ynorm, recv)
                if rank == 0:
                    print "%d, norm = %.10f, time=%f(ms)" % (i / x_steps, np.sqrt(recv[0]), (MPI.Wtime()-start)*1000)
        
    return d_xk
        
def get_fista_err_func(dtype):
    template = """
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
            err[i] = ( tmp - b[i] );
        }

    }
    
    """
    
    func = func_compile("err_update", template % {"type": dtype_to_ctype(dtype)})
    func.prepare([np.intp, np.intp, np.intp, np.int32, np.int32])

    return func


def get_fista_update_func(dtype):
    template = """
    __global__ void update( %(type)s* xk, %(type)s* yk, %(type)s* temp, int size, int groupsize, %(type)s L)
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
            xk[i] = ( yk[i] - tmp/L );
        }
    }
    
    """
    
    func = func_compile("update", template % {"type": dtype_to_ctype(dtype)})
    func.prepare([np.intp, np.intp, np.intp, np.int32, np.int32, dtype.type])
    
    return func


def get_sq_kernel(dtype_s, dtype_q):
    template = """
        #include <pycuda/pycuda-complex.hpp>
        
        __global__ void
        sq_Kernel(%(types)s* d_S, %(typeq)s* d_q, %(types)s rcond, int size)
        {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            int total = blockDim.x * gridDim.x;

            __shared__ %(types)s max[1];
        
            if(threadIdx.x == 0)
            {
                max[0] = d_S[0] * rcond;
            }
            __syncthreads();

            for(int i = tid; i < size; i += total)
            {
                %(types)s s = d_S[i];
                %(typeq)s q = d_q[i];

                if(s > max[0])
                {
                    d_q[i] = q / s;
                }else
                {
                    d_q[i] = 0.0;
                }
            }
        }
        
        """
    func = func_compile("sq_Kernel", template % {"types": dtype_to_ctype(dtype_s), "typeq": dtype_to_ctype(dtype_q)})
    return func

def get_eigsq_kernel(dtype_s, dtype_q):
       
    template = """
        #include <pycuda/pycuda-complex.hpp>
        
        __global__ void
        eigsq_Kernel(%(types)s* d_S, %(typeq)s* d_q, %(types)s thres, int size)
        {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            int total = blockDim.x * gridDim.x;

            for(int i = tid; i < size; i += total)
            {
                %(types)s s = d_S[i];
                %(typeq)s q = d_q[i];

                if(fabs%(iff)s(s) > thres)
                {
                    d_q[i] = q / s;
                }else
                {
                    d_q[i] = 0.0;
                }
            }
        }
        
        """
    func = func_compile("eigsq_Kernel", template % {"types": dtype_to_ctype(dtype_s), "typeq": dtype_to_ctype(dtype_q), "iff": "f" if dtype_q == np.float32 else ""})
    return func


def get_sinv_kernel(dtype):
    template = """
        #include <pycuda/pycuda-complex.hpp>
        
        __global__ void
        sinv_Kernel(%(types)s* d_S, %(types)s rcond, int size)
        {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            int total = blockDim.x * gridDim.x;

            __shared__ %(types)s max[1];
        
            if(threadIdx.x == 0)
            {
                max[0] = d_S[0] * rcond;
            }
            __syncthreads();

            for(int i = tid; i < size; i += total)
            {
                %(types)s s = d_S[i];

                if(s > max[0])
                {
                    d_S[i] = 1.0 / s;
                }else
                {
                    d_S[i] = 0.0;
                }
            }
        }
        
        """
    func = func_compile("sinv_Kernel", template % {"types": dtype_to_ctype(dtype_s)})
    return func


def get_svinv_kernel(dtype_s, dtype_v):
    template = """
        #include <pycuda/pycuda-complex.hpp>
        
        __global__ void
        svinv_Kernel(%(types)s* d_S, %(typev)s* d_V, int ld, int size, %(types)s rcond)
        {
            int tid = threadIdx.x;
            int bid = blockIdx.x;
            int bdim = blockDim.x;

            __shared__ %(types)s s[0];

            if(threadIdx.x == 0)
            {
                %(types)s max = d_S[0] * rcond;
                s[0] = d_S[bid];
                if(s[0] > max)
                {
                    s[0] = 1/s[0];
                }else
                {
                    s[0] = 0.0;
                }
            }
            __syncthreads();

            for(int i = tid; i < size; i+=bdim)
            {
                d_V[bid * ld + i] *= s[0];
            }

        }
        
        """
    func = func_compile("svinv_Kernel", template % {"types": dtype_to_ctype(dtype_s), "typev": dtype_to_ctype(dtype_v)})
    return func