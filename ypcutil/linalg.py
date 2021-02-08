#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype, context_dependent_memoize
from pycuda.compiler import SourceModule

import skcuda.cublas as cublas
import skcuda.cula as cula
import skcuda.cusolver as cusolver
from . import parray
from . import simpleio as si


""" assuming row major storage as in PitchArray """
class cublashandle(object):
    """ Create a cublas handle """
    def __init__(self):
        self.handle = None
        self.create()

    def create(self):
        if self.handle is None:
            self.handle = cublas.cublasCreate()

    def destroy(self):
        if self.handle is not None:
            cublas.cublasDestroy(self.handle)
            self.handle = None

    def __del__(self):
        self.destroy()


def dot(A, B, opa = 'n', opb = 'n',
        C = None, Cstart = None,
        scale = 1.0, Cscale = 0.0, handle = None):
    """
    Multiplication of two matrices A and B in PitchArray format
    if C is specified, use the memory in C.
    Specified C must have the same leading dimension as that of the result and
    the other dimension must be bigger or equal to that of the result.

    Parameters:
    -----------
    A: parray.PitchArray
    B: parray.PitchArray
    opa: str
         operation on A
         'n' or 'N': use A itself
         't' or 'T': use transpose of A
         'c' or 'C': use conjugate transpose of A
    opb: str
         operation on B
         'n' or 'N': use B itself
         't' or 'T': use transpose of B
         'c' or 'C': use conjugate transpose of B
    C: parray.PitchArray
       if specified, the result will be stored in C
    Cstart: int
            the offset start of C array
    scale: float
            scaling factor for A*B
            see Cscale
    Cscale: float
            scaling factor for C
            result will be C = C*Cscale + scale*A*B

    Note:
    -----
    works only for CUDA VERSION > 4.0 where handle is introduced.

    Will NOT work for complex case when A and B shares overlapping
    memory, but should work if A==B.
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
        raise ValueError("matrix dimension mismatch, "
                         "(%d,%d) with (%d,%d)" % (m,n,k,l))

    dtype = A.dtype
    if dtype in [np.float32, np.float64]:
        if opb in ['c', 'C']:
            opb = 't'

        if opa in ['c', 'C']:
            opa = 't'

    scale = dtype.type(scale)
    Cscale = dtype.type(Cscale)

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
        Cempty = True
        Cscale = dtype.type(0)
    else:
        Cempty = False
        if Cstart is None:
            Cstart = 0
        if C.shape[1] != l:
            raise AttributeError("shape of the provided result array "
                                 + C.shape.__str__()
                                 + " does not match intended result "
                                 + (m,l).__str__())
        if C.shape[0] < m + Cstart:
            raise AttributeError("shape of the provided result array "
                                 + C.shape.__str__()
                                 + " does not match intended result "
                                + (m,l).__str__())
        if C.dtype != dtype:
            raise TypeError("Result array C provided must have "
                            "the same dtype as inputs")

    conjA = False
    conjB = False
    conjC = False

    sameflag = (A==B)

    itemsize = C.dtype.itemsize
    handlestr = "handle.handle"
    if m == 1:
        if n == 1:
            alpha = A.get()[0,0]
            if opa in ['c','C']:
                alpha = np.conj(alpha)
            C*=Cscale
            if opb in ['c','C']:
                func = (tp+"axpy(handle.handle, l, alpha*scale, "
                        + "parray.conj(B).gpudata, 1,"
                        + "int(C.gpudata)+Cstart*itemsize, 1)")
            else:
                func = (tp+"axpy(handle.handle, l, alpha*scale, "
                        + "B.gpudata, 1, "
                        + "int(C.gpudata)+Cstart*itemsize, 1)")
        else:
            if l > 1:
                alpha = scale
                beta = Cscale
                if opa in ['c','C']:
                    A.conj()
                    conjA = True
                func = (tp+"gemv(handle.handle, '"+opb+"',B.shape[1], "
                        + "B.shape[0], alpha, B.gpudata, B.ld, A.gpudata, "
                        + "1, beta, int(C.gpudata)+Cstart*itemsize*C.ld, 1)")
            else:
                if opa in ['c','C']:
                    if opb in ['c', 'C']:
                        func = ("C.set(np.array(scale*" + tp
                                + "dotu(handle.handle, n, A.gpudata, "
                                + "1, B.gpudata, 1)"
                                +").conj()+C.get()*Cscale)")
                    else:
                        func = ("C.set(np.array(scale*" + tp
                                + "dotc(handle.handle, n, A.gpudata, "
                                + "1, B.gpudata, 1)) + C.get()*Cscale)")
                elif opb in ['c', 'C']:
                    func = ("C.set(np.array(scale*" + tp
                            + "dotc(handle.handle, n, B.gpudata, 1, "
                            + "A.gpudata, 1)) + C.get()*Cscale)")
                else:
                    if complex_type:
                        func = ("C.set(np.array(scale*" + tp
                                + "dotu(handle.handle, n, A.gpudata, 1, "
                                + "B.gpudata, 1)) + C.get()*Cscale)")
                    else:
                        func = ("C.set(np.array(scale*" + tp
                                + "dot(handle.handle, n, A.gpudata, 1, "
                                + "B.gpudata, 1)) + C.get()*Cscale)")
    else:#m!=1
        if n == 1:
            if l == 1:
                alpha = B.get()[0,0]
                if opb in ['c','C']:
                    alpha = np.conj(alpha)
                C*=Cscale
                if opa in ['c','C']:
                    func = (tp+"axpy(handle.handle, m, alpha*scale, "
                            + "parray.conj(A).gpudata, 1, "
                            + "int(C.gpudata)+Cstart*itemsize, 1)")
                else:
                    func = (tp+"axpy(handle.handle, m, alpha*scale, "
                            + "A.gpudata, 1, "
                            + "int(C.gpudata)+Cstart*itemsize, 1)")
            else:
                if Cempty:
                    C.fill(0)
                else:
                    C*=Cscale
                if opa in ['c','C']:
                    if opb in ['c', 'C']:
                        B.conj()
                        conjB = True
                        func = (tp + "gerc(handle.handle, l, m, scale, "
                                + "B.gpudata, 1, A.gpudata, 1, "
                                + "int(C.gpudata)+Cstart*itemsize*C.ld, "
                                + "C.ld)")
                    else:
                        func = (tp + "gerc(handle.handle, l, m, scale, "
                                + "B.gpudata, 1, A.gpudata, 1, "
                                + "int(C.gpudata)+Cstart*itemsize*C.ld, "
                                + "C.ld)")
                elif opb in ['c', 'C']:
                    if sameflag:
                        B.conj()
                        conjB = True
                        func = (tp + "gerc(handle.handle, l, m, scale, "
                                + "B.gpudata, 1, A.gpudata, 1, "
                                + "int(C.gpudata)+Cstart*itemsize*C.ld, C.ld)")

                    else:
                        B.conj()
                        conjB = True
                        func = (tp + "geru(handle.handle, l, m, scale, "
                                + "B.gpudata, 1, A.gpudata, 1, "
                                + "int(C.gpudata)+Cstart*itemsize*C.ld, C.ld)")
                else:
                    if complex_type:
                        func = (tp + "geru(handle.handle, l, m, scale, "
                                + "B.gpudata, 1,  A.gpudata, 1, "
                                + "int(C.gpudata)+Cstart*itemsize*C.ld, C.ld)")
                    else:
                        func = (tp + "ger(handle.handle, l, m, scale, "
                                + "B.gpudata, 1, A.gpudata, 1, "
                                + "int(C.gpudata)+Cstart*itemsize*C.ld, C.ld)")
        else:
            if l == 1:
                if opb in ['c', 'C']:
                    if opa in ['c', 'C']:
                        conjC = True
                        if not Cempty:
                            C.conj()
                            Cscale = Cscale.conj()
                        func = (tp + "gemv(handle.handle, 'n', A.shape[1], "
                                + "A.shape[0], scale, A.gpudata, A.ld, "
                                + "B.gpudata, 1, Cscale, int(C.gpudata) + "
                                + "Cstart * itemsize * C.ld, 1)")
                    else:
                        B.conj()
                        conjB = True
                        if opa in ['t', 'T']:
                            opa = 'n'
                        else:
                            opa = 't'

                        func = (tp + "gemv(handle.handle, '" + opa + "', "
                                + "A.shape[1], A.shape[0], scale, A.gpudata, "
                                + "A.ld, B.gpudata, 1, Cscale, "
                                + "int(C.gpudata)+Cstart*itemsize*C.ld, 1)")
                else:
                    if opa in ['c', 'C']:
                        B.conj()
                        conjB = True
                        conjC = True
                        if not Cempty:
                            C.conj()
                            Cscale = Cscale.conj()
                        func = (tp + "gemv(handle.handle, 'n', A.shape[1], "
                                + "A.shape[0], scale, A.gpudata, A.ld, "
                                + "B.gpudata, 1, Cscale, int(C.gpudata) + "
                                + "Cstart * itemsize * C.ld, 1)")
                    else:
                        if opa in ['t', 'T']:
                            opa = 'n'
                        else:
                            opa = 't'
                        func = (tp + "gemv(handle.handle, '" + opa + "', "
                                + "A.shape[1],  A.shape[0], scale, A.gpudata, "
                                + "A.ld, B.gpudata, 1, Cscale, int(C.gpudata) "
                                + "+ Cstart * itemsize * C.ld, 1)")
            else:
                func = (tp+"gemm(handle.handle, '" + opb + "','" + opa + "', "
                        + "l, m, k, scale, B.gpudata, B.ld, A.gpudata, A.ld, "
                        + "Cscale, int(C.gpudata) + "
                        + "Cstart * itemsize * C.ld, C.ld)")

    if handle is None:
        handle = cublashandle()
    eval(func)

    if conjC:
        C.conj()

    if conjA:
        A.conj()

    if conjB:
        B.conj()
    return C


def norm(A, handle = None):
    """
    computes the l2 norm of a vector A

    Parameters
    ----------
    A : parray.PitchArray
        a one dimensional vector
    handle : cublashandle, optional
        handle to cublas
    """
    if handle is None:
        handle = cublashandle()
    dtype = A.dtype
    if dtype == np.float64:
        nrmfunc = cublas.cublasDnrm2
    elif dtype == np.complex128:
        nrmfunc = cublas.cublasDznrm2
    elif dtype == np.float32:
        nrmfunc = cublas.cublasSnrm2
    elif dtype == np.complex64:
        nrmfunc = cublas.cublasScnrm2
    result = nrmfunc(handle.handle, A.size, A.gpudata, 1)
    return result

def svd(G, compute_u = True, compute_v = True, econ = False):
    """
    compute Singular Value Decompositon of G
    G = U*(diag(S))*V

    Parameters
    ----------------------------------------
    G:  PitchArray, GPUArray or numpy.ndarray of shape (m,n)
        if G is GPUArray or PitchArray, its gpudata will be
        destroyed after calling the function
    compute_u: bool
               whether return U matrix or not
    compute_v: bool
               whether return V matrix or not
    econ: bool
          return economical matrix

    Returns:
    U:  parray.PitchArray matrix
        as U in G = U*(diag(S))*V,
        if econ, returns the first min(m,n) columns of U
    S:  parray.PitchArray vector
        a row vector containing all singular values
        with descending order
    V:  parray.PitchArray matrix
        as V in G = U*(diag(S))*V,
        if econ, returns the first min(m,n) rows of V

    order of output:
    always obeys the order U,S,V
    e.g.
    S = svd(G, compute_u = False, compute_v = False)
    U,S = svd(G, compute_u = True, compute_v = False)
    S,V = svd(G, compute_u = False, compute_v = True)
    U,S,V = svd(G, compute_u = True, compute_v = True)
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
        if cula._libcula_toolkit == 'standard':
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
                    svd_func(jobu, jobvt, A.shape[1], A.shape[0],
                             A.gpudata, A.ld, S.gpudata, V.gpudata,
                            V.ld, 1, 1)
                    #cula.culaShutdown()
                    return A,S,V
                else:
                    jobu = 'O'
                    jobvt = 'A'
                    U = parray.empty((A.shape[0], A.shape[0]), A.dtype)
                    svd_func(jobu, jobvt, A.shape[1], A.shape[0],
                             A.gpudata, A.ld, S.gpudata, 1, 1,
                             U.gpudata, U.ld)
                    #cula.culaShutdown()
                    return U,S,A
            else:
                if A.shape[1] <= A.shape[0]:
                    jobu = 'O'
                    jobvt = 'A'
                    U = parray.empty((A.shape[0], A.shape[0]), A.dtype)
                    svd_func(jobu, jobvt, A.shape[1], A.shape[0],
                             A.gpudata, A.ld, S.gpudata, 1, 1,
                             U.gpudata, U.ld)
                    #cula.culaShutdown()
                    A.shape = (A.shape[1],A.shape[1])
                    return U,S,A
                else:
                    jobu = 'A'
                    jobvt = 'O'
                    V = parray.empty((A.shape[1], A.shape[1]), A.dtype)
                    svd_func(jobu, jobvt, A.shape[1], A.shape[0],
                             A.gpudata, A.ld, S.gpudata, V.gpudata,
                             V.ld, 1, 1)
                    A.shape = (A.shape[0], A.shape[0])
                    #cula.culaShutdown()
                    return A,S,V
        else:
            if econ | (A.shape[1] >= A.shape[0]):
                jobu = 'N'
                jobvt = 'O'
                svd_func(jobu, jobvt, A.shape[1], A.shape[0],
                         A.gpudata, A.ld, S.gpudata, 1, 1, 1, 1)
                if (A.shape[1] > A.shape[0]):
                    A.shape = (A.shape[0], A.shape[0])
                #cula.culaShutdown()
                return A,S
            else:
                jobu = 'N'
                jobvt = 'A'
                U = parray.empty((A.shape[0],A.shape[0]),A.dtype)
                svd_func(jobu, jobvt, A.shape[1], A.shape[0],
                         A.gpudata, A.ld, S.gpudata, 1, 1, U.gpudata, U.ld)
                #cula.culaShutdown()
                return U,S
    else:
        if compute_v:
            if econ | (A.shape[1] <= A.shape[0]):
                jobu = 'O'
                jobvt = 'N'
                svd_func(jobu, jobvt, A.shape[1], A.shape[0],
                         A.gpudata, A.ld, S.gpudata, 1, 1, 1, 1)
                if (A.shape[1] < A.shape[0]):
                    A.shape = (A.shape[1], A.shape[1])
                #cula.culaShutdown()
                return S,A
            else:
                jobu = 'A'
                jobvt = 'N'
                V = parray.empty((A.shape[1],A.shape[1]),A.dtype)
                svd_func(jobu, jobvt, A.shape[1], A.shape[0],
                         A.gpudata, A.ld, S.gpudata, V.gpudata, V.ld, 1, 1)
                #cula.culaShutdown()
                return S,V
        else:
            jobu = 'N'
            jobvt = 'N'
            svd_func(jobu, jobvt, A.shape[1], A.shape[0], A.gpudata,
                     A.ld, S.gpudata, 1, 1, 1, 1)
            #cula.culaShutdown()
            return S


def pinv(G, rcond = 1e-8):
    """
    Computes the Moore-Penrose pseudo-inversion using SVD method

    Parameters
    -----------------------------------
    G:  PitchArray, GPUArray or numpy.ndarray of shape (m,n)
        if G is GPUArray or PitchArray,
        its gpudata will be destroyed after calling the function
    rcond:  float
            Cutoff for small singular values.
            Singular values smaller (in modulus) than
            `rcond` * largest_singular_value (again, in modulus)
            are set to zero

    Returns
    -----------------------------------
    out: PitchArray
         pseudo-inverse of G matrix
    """
    U,S,V = svd(G, econ=1)
    print("using absolute rcond:", S[0].get()*rcond, "in inversion")
    sv_func = _get_svinv_kernel(S.dtype, V.dtype)
    sv_func.prepared_call(
        (S.size, 1), (256,1,1), S.gpudata, V.gpudata,
        V.ld, V.shape[1], rcond)
    return dot(V, U, opa='c', opb='c')

def pinv_sym(G, rcond = 1e-8):
    """
    Computes the Moore-Penrose pseudo-inversion using SVD method

    Parameters
    -----------------------------------
    G:  PitchArray, GPUArray or numpy.ndarray of shape (m,m)
        symmetric matrix
        if G is GPUArray or PitchArray,
        its gpudata will be destroyed after calling the function
    rcond:  float
            Cutoff for small singular values.
            Singular values smaller (in modulus) than
            `rcond` * largest_singular_value (again, in modulus)
            are set to zero

    Returns
    -----------------------------------
    out: PitchArray
         pseudo-inverse of G matrix
    """
    S,V = svd(G, compute_u=0)
    print("using absolute rcond:", S[0].get()*rcond, "in inversion")
    V_2 = V.copy()
    sv_func = _get_svinv_kernel(S.dtype, V.dtype)
    sv_func.prepared_call(
        (S.size, 1), (256,1,1), S.gpudata, V.gpudata,
        V.ld, V.shape[1], rcond)
    return dot(V, V_2, opa='c')

def solve_eq(G, q, rcond = 1e-8):
    """
    Solves Gc = q using pseudo-inversion

    Parameters
    ------------------------------
    G: PitchArray
       Its gpudata will be destroyed after calling the function
    q: PitchArray
        dtype of G and q must b the same
    rcond:  Cutoff for small singular values.
            Singular values smaller (in modulus) than
            `rcond` * largest_singular_value (again, in modulus)
            are set to zero

    Returns
    ------------------------------
    out: PitchArray vector
         solution c
    """
    if G.dtype != q.dtype:
        raise TypeError("G,q must be of the same dtype")

    if G.shape[0] != q.shape[0]:
        raise ValueError("number of columns of G must be "
                         "the same of size of q")
    U,S,V = svd(G, econ=1)
    qq = dot(U, q, opa='c')
    print("using absolute rcond:", S[0].get()*rcond, "in inversion")
    sq_func = _get_sq_kernel(S.dtype, qq.dtype)
    sq_func.prepared_call(
        (6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1),
        (256,1,1), S.gpudata, qq.gpudata, rcond, S.size)
    result = dot(V, qq, opa='c')
    return result


def solve_eq_sym(G, q, rcond = 1e-8, save_singular = None, cut_num = None, disp_rcond=False):
    """
    solves Gc = q using pseudo-inversion via SVD,
    with G a self-adjoint matrix

    Parameters
    -------------------------------------
    G: PitchArray, a self-adjoint matrix
       Its gpudata will be destroyed after calling the function
    q: PitchArray
       dtype of G and q must b the same
    rcond:  float
            Cutoff for small singular values.
            Singular values smaller (in modulus) than
            `rcond` * largest_singular_value (again, in modulus)
            are set to zero

    Returns
    --------------------------------------
    out: PitchArray vector
         solution c
    """
    if rcond is None:
        rcond = 1e-8

    if G.dtype != q.dtype:
        raise TypeError("G,q must be of the same dtype")

    if G.shape[0] != G.shape[1]:
        raise ValueError("G must be square matrix")

    if G.shape[1] != q.size:
        raise ValueError("number of columns of G must be "
                         "the same of size of q")
    U,S = svd(G, compute_v=0)
    if save_singular is not None:
        si.write_memory_to_file(S, save_singular)

    if cut_num is not None:
        if cut_num >= S.size:
            rcond = 1e-7
        else:
            rcond = (S[cut_num].get()+S[cut_num-1].get())/2/S[0].get()

    qq = dot(U, q, opa='c')
    if disp_rcond:
        print("using absolute rcond:", S[0].get()*rcond, "in inversion")

    sq_func = _get_sq_kernel(S.dtype, qq.dtype)
    sq_func.prepared_call(
        (6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1),
        (256,1,1), S.gpudata, qq.gpudata, rcond, S.size)
    result = dot(U, qq)
    return result

def solve_eq_sym_mixed(G, qr, qi, rcond = 1e-8, save_singular = None, cut_num = None):
    """
    solves Gc = qr+1j*qi using pseudo-inversion via SVD,
    with G a real symmetric matrix

    Parameters
    -------------------------------------
    G: PitchArray, a self-adjoint matrix
       Its gpudata will be destroyed after calling the function
    q: PitchArray
       dtype of G and q must b the same
    rcond:  float
            Cutoff for small singular values.
            Singular values smaller (in modulus) than
            `rcond` * largest_singular_value (again, in modulus)
            are set to zero

    Returns
    --------------------------------------
    out: PitchArray vector
         solution c
    """
    if G.dtype != qr.dtype:
        raise TypeError("G,qr must be of the same dtype")
    if G.dtype != qi.dtype:
        raise TypeError("G,qr must be of the same dtype")

    if G.shape[0] != G.shape[1]:
        raise ValueError("G must be square matrix")

    if G.shape[1] != qr.size:
        raise ValueError("number of columns of G must be "
                         "the same of size of q")
    U,S = svd(G, compute_v=0)
    if save_singular is not None:
        si.write_memory_to_file(S, save_singular)

    if cut_num is not None:
        if cut_num >= S.size:
            rcond = 1e-7
        else:
            rcond = (S[cut_num].get()+S[cut_num-1].get())/2/S[0].get()

    qqr = dot(U, qr, opa='c')
    print("using absolute rcond:", S[0].get()*rcond, "in inversion")

    sq_func = _get_sq_kernel(S.dtype, qqr.dtype)
    sq_func.prepared_call(
        (6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1),
        (256,1,1), S.gpudata, qqr.gpudata, rcond, S.size)
    resultr = dot(U, qqr)

    qqi = dot(U, qi, opa='c')

    sq_func.prepared_call(
        (6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1),
        (256,1,1), S.gpudata, qqi.gpudata, rcond, S.size)
    resulti = dot(U, qqi)

    return parray.make_complex(resultr, resulti)

def eig_sym_cusolver(x, max_sweep = 15, sort_eig = 1, tolerance = 1e-7,
                     get_stats = False):
    n, m = x.shape
    w = parray.empty((n,1), dtype = parray.complextofloat(x.dtype))

    handle = cusolver.cusolverDnCreate()
    params = cusolver.cusolverDnCreateSyevjInfo()
    cusolver.cusolverDnXsyevjSetTolerance(params, tolerance)
    cusolver.cusolverDnXsyevjSetMaxSweeps(params, max_sweep)
    cusolver.cusolverDnXsyevjSetSortEig(params, sort_eig)

    if x.dtype == np.float32:
        buffer_func = cusolver.cusolverDnSsyevj_bufferSize
        solver_func = cusolver.cusolverDnSsyevj
    elif x.dtype == np.double:
        buffer_func = cusolver.cusolverDnDsyevj_bufferSize
        solver_func = cusolver.cusolverDnDsyevj
    elif x.dtype == np.complex64:
        #raise ValueError("current implementation of eigendecomposition of\
        #                  Hermitian matrix does not work for single precision")
        x.conj()
        buffer_func = cusolver.cusolverDnCheevj_bufferSize
        solver_func = cusolver.cusolverDnCheevj
    elif x.dtype == np.complex128:
        x.conj()
        buffer_func = cusolver.cusolverDnZheevj_bufferSize
        solver_func = cusolver.cusolverDnZheevj

    lwork = buffer_func(handle, 'CUSOLVER_EIG_MODE_VECTOR',
                        'u', n, x.gpudata, x.ld,
                        w.gpudata, params)

    workspace_gpu = parray.zeros(lwork, dtype = x.dtype)
    info = parray.zeros(1, dtype = np.int32)
    # Compute:
    solver_func(handle, 'CUSOLVER_EIG_MODE_VECTOR',
                'u', n, x.gpudata, x.ld,
                w.gpudata, workspace_gpu.gpudata,
                lwork, info.gpudata, params)
    # print info
    if get_stats:
        sweeps = cusolver.cusolverDnXsyevjGetSweeps(handle, params)
        residual = cusolver.cusolverDnXsyevjGetResidual(handle, params)

    cusolver.cusolverDnDestroySyevjInfo(params)
    cusolver.cusolverDnDestroy(handle)

    if get_stats:
        return w, x.T(), sweeps, residual
    else:
        return w, x.T()

def eig_sym(G, compute_z = True, uplo = 'U'):
    """
    compute Eigenvalue Decompositon of a symmetric or Hermitian matrix G
    G = V D V^{*}

    Parameters
    -------------------------------------
    G:  PitchArray, GPUArray or numpy.ndarray
        if G is GPUArray or PitchArray, its gpudata will be destroyed
        after calling the function
    compute_z: bool
               whether return eigenvectors
    uplo: str
          'U' or 'u' assumes the entries of G are stored
          in upper triangular part,
          lower off diagonal triangular part is not referenced
          'L' or 'l' assumes the entries of G are stored
          in lower triangular part,
          upper off diagonal triangular part is not referenced

    Returns
    -------------------------------------
    D:  PitchArray
        a row vector containing all eigenvalues with ascending order
    V:  PitchArray
        if compute_z, jth column of V contains orthonormal
        eigenvector associated with jth eigenvalue

    Examples
    D = eig_sym(G, compute_z = False)
    D,V = eig_sym(G, compute_z = True)
    """
    if cula._libcula_toolkit != 'premium':
        raise ValueError("eigenvalue decomposition is only supported "
                         "in premium version of CULA")

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
    handle = cublashandle()
    if compute_z:
        jobz = 'V'
    else:
        jobz = 'N'
    eig_func(handle.handle, jobz, uplo, A.shape[0], A.gpudata, A.ld, D.gpudata)
    #cula.culaShutdown()
    if compute_z:
        return D, A.conj().T()
    else:
        return D


def add_eye(A, scale):
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square matrix")
    func = _get_add_eye_func(A.dtype)
    func.prepared_call(
        (6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1),
        (256,1,1), A.gpudata, A.ld, scale, A.shape[0])
    return A


def FISTA_l2(A, b, L = 1, steps=5000):
    import time

    xk = parray.zeros_like(b)
    t_k = 1
    t_km1 = 1
    xkm1 = xk.copy()
    x_steps = steps//20
    handle = cublashandle()
    start = time.time()

    for i in range(1,steps+1):
        yk = xk + ((t_km1-1)/t_k)*(xk-xkm1)
        err = dot(A, yk, handle = handle) - b
        temp = dot(A, err, handle = handle, opa='t')
        xkm1 = xk.copy()
        xk = yk - temp / L

        t_kp1 = 0.5*(1+np.sqrt(1+4*t_k**2))
        t_km1 = t_k
        t_k = t_kp1

        if i%x_steps == 0:
            ynorm = norm(err, handle = handle)
            print("%d, norm = %.10f, time=%f(ms)" % (
                  i / x_steps, ynorm, (time.time()-start)*1000))
    return xk


def fista_l2_mpi(A, b, rank, diag, groupsize, col_comm,
                 row_comm, diag_comm, L = 1, steps = 4000):
    from mpi4py import MPI

    if A.dtype != b.dtype:
        raise TypeError("matrix multiplication must have same dtype")
    if (len(A.shape) != 2) | (len(b.shape) != 2):
        raise TypeError("G, q must both be matrices")

    col_rank = col_comm.Get_rank()
    row_rank = row_comm.Get_rank()

    XOUTPUTSTEPS = min(20, steps)
    x_steps = steps // XOUTPUTSTEPS

    L = float(L)
    d_xk = parray.zeros_like(b)
    t_k = 1.
    t_km1 = 1.
    d_xkm1 = d_xk.copy()

    grid = (6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1)
    handle = cublashandle()
    d_err = parray.empty_like(b)

    if d_err.dtype == np.float64:
        mpitype = MPI.DOUBLE
    else:
        mpitype = MPI.FLOAT
    yk = np.zeros((A.shape[1], 1), A.dtype)
    d_yk = parray.empty_like(yk)
    err = np.zeros((A.shape[0],1), A.dtype)
    d_err = parray.empty_like(err)

    if diag:
        temp_all = np.zeros((b.size*groupsize, 1), A.dtype)
        d_temp_all = parray.empty_like(temp_all)
        update_err_func = _get_fista_err_func(A.dtype)
        update_xk_func = _get_fista_update_func(A.dtype)
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
        d_temp = dot(A, d_yk, handle = handle)
        temp = d_temp.get()

        row_comm.Gather(
            [temp, temp.size, mpitype],
            [temp_all if diag else None, temp.size, mpitype],
            root = col_rank)

        if diag:
            d_temp_all.set(temp_all)
            update_err_func.prepared_call(
                grid, (256,1,1), d_err.gpudata, b.gpudata,
                d_temp_all.gpudata, b.size, groupsize)
            err = d_err.get()
            ynorm[0] = norm(d_err, handle = handle)**2
        row_comm.Bcast(err, root = col_rank)
        d_err.set(err)
        d_temp = dot(A, d_err, opa='t', handle = handle)
        temp = d_temp.get()
        col_comm.Gather(
            [temp, temp.size, mpitype],
            [temp_all if diag else None, temp.size, mpitype],
            root = row_rank)

        if diag:
            d_temp_all.set(temp_all)
            cuda.memcpy_dtod(d_xkm1.gpudata, d_xk.gpudata,
                             d_xk.size*d_xk.dtype.itemsize)
            update_xk_func.prepared_call(
                grid, (256,1,1), d_xk.gpudata, d_yk.gpudata,
                d_temp_all.gpudata, b.size, groupsize, L)
            t_kp1 = 0.5*(1+np.sqrt(1+4*t_k**2))
            t_km1 = t_k
            t_k = t_kp1

            if i % x_steps == 0:
                diag_comm.Reduce(ynorm, recv)
                if rank == 0:
                    print("%d, norm = %.10f, time=%f(ms)" % (
                          i / x_steps, np.sqrt(recv[0]),
                          (MPI.Wtime()-start)*1000))
    return d_xk


@context_dependent_memoize
def _get_fista_err_func(dtype):
    template = """
__global__ void err_update( %(type)s* err, %(type)s* b,
                            %(type)s* Ax, int size, int groupsize)
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
    mod = SourceModule(template % {"type": dtype_to_ctype(dtype)})
    func = mod.get_function("err_update")
    func.prepare('PPPii')#[np.intp, np.intp, np.intp, np.int32, np.int32])
    return func


@context_dependent_memoize
def _get_fista_update_func(dtype):
    template = """
__global__ void update(%(type)s* xk, %(type)s* yk, %(type)s* temp,
                       int size, int groupsize, %(type)s L)
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
    mod = SourceModule(template % {"type": dtype_to_ctype(dtype)})
    func = mod.get_function("update")
    func.prepare('PPPii'+np.dtype(dtype).char)#[np.intp, np.intp, np.intp, np.int32, np.int32,
    #             np.double if dtype == np.double else np.float32])
    return func


@context_dependent_memoize
def _get_sq_kernel(dtype_s, dtype_q):
    template = """
#include <pycuda-complex.hpp>

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
    mod = SourceModule(template % {"types": dtype_to_ctype(dtype_s),
                       "typeq": dtype_to_ctype(dtype_q)})
    func = mod.get_function("sq_Kernel")
    func.prepare('PP'+np.dtype(dtype_s).char+'i')#[np.intp, np.intp,
    #             np.double if dtype_s == np.double else np.float32,
    #             np.int32])
    return func


@context_dependent_memoize
def _get_eigsq_kernel(dtype_s, dtype_q):
    template = """
#include <pycuda-complex.hpp>

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
    mod = SourceModule(template % {
                       "types": dtype_to_ctype(dtype_s),
                       "typeq": dtype_to_ctype(dtype_q),
                       "iff": "f" if dtype_q == np.float32 else ""})
    func = mod.get_function("eigsq_Kernel")
    func.prepare('PP'+np.dtype(dtype_s).char+'i')#[np.intp, np.intp,
    #              np.double if dtype_s == np.double else np.float32,
    #              np.int32])
    return func


@context_dependent_memoize
def _get_sinv_kernel(dtype):
    template = """
#include <pycuda-complex.hpp>

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
    mod = SourceModule(template % {"types": dtype_to_ctype(dtype_s)})
    func = mod.get_function("sinv_Kernel")
    func.prepare('P'++np.dtype(dtype).char+'i')#[np.intp, np.double if dtype == np.double else np.float32,
    #            np.int32])
    return func


@context_dependent_memoize
def _get_svinv_kernel(dtype_s, dtype_v):
    template = """
#include <pycuda-complex.hpp>

__global__ void
svinv_Kernel(%(types)s* d_S, %(typev)s* d_V, int ld,
             int size, %(types)s rcond)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bdim = blockDim.x;

    __shared__ %(types)s s[1];

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
    mod = SourceModule(template % {
                       "types": dtype_to_ctype(dtype_s),
                       "typev": dtype_to_ctype(dtype_v)})
    func = mod.get_function("svinv_Kernel")
    func.prepare('PPii'+np.dtype(dtype_s).char)#[np.intp, np.intp, np.int32, np.int32,
    #             np.double if dtype_s == np.double else np.float32])
    return func


@context_dependent_memoize
def _get_add_eye_func(dtype):
    template = """
__global__ void adddiag(%(type)s* A, int ld, %(type)s value, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = blockDim.x * gridDim.x;

    for(int i = tid; i < N; i+=total_threads)
    {
        A[tid * ld + tid] += value;
    }
}

    """
    mod = SourceModule(template % {"type": dtype_to_ctype(dtype)})
    func = mod.get_function("adddiag")
    func.prepare('Pi'+np.dtype(dtype).char+'i')
    #    [np.intp, np.int32,
    #    dtype.type if isinstance(dtype, np.dtype) else dtype,
    #    np.int32])
    return func
