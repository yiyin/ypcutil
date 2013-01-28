from ypcutil.importcuda import *
import ypcutil.linalg as sl
import scikits.cuda.cublas as cublas

print "test 1"
A=np.random.rand(100,1)
B=np.random.rand(100,100)
C=np.dot(A.T, B)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='t')
print np.abs(C-d_C.get()).max()

print "test 2"
A=np.random.rand(1,100)
B=np.random.rand(100,100)
C=np.dot(A,B)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B)
print np.abs(C-d_C.get()).max()

print "test 3"
A=np.random.rand(1,100)
B=np.random.rand(100,100)
C=np.dot(A,B.T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B,opb='t')
print np.abs(C-d_C.get()).max()

print "test 4"
A=np.random.rand(100,1)
B=np.random.rand(100,100)
C=np.dot(A.T, B.T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='t', opb='t')
print np.abs(C-d_C.get()).max()

print "test 5"
A=np.random.rand(100,1) + np.random.rand(100,1)*1j
B=np.random.rand(100,100) + np.random.rand(100,100)*1j
C=np.dot(A.T, B)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='t')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 6"
A=np.random.rand(1,100) + np.random.rand(1,100)*1j
B=np.random.rand(100,100) + np.random.rand(100,100)*1j
C=np.dot(A, B)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B)
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 7"
A=np.random.rand(1,100) + np.random.rand(1,100)*1j
B=np.random.rand(100,100) + np.random.rand(100,100)*1j
C=np.dot(A, B.T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opb='t')
print np.abs(C-d_C.get()).max()

print "test 8"
A=np.random.rand(100,1) + np.random.rand(100,1)*1j
B=np.random.rand(100,100) + np.random.rand(100,100)*1j
C=np.dot(A.T, B.T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa = 't', opb='t')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

A=np.random.rand(1,100) + np.random.rand(1,100)*1j
B=np.random.rand(100,100) + np.random.rand(100,100)*1j
C=np.dot(A, B.conj().T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opb='c')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 9"
A=np.random.rand(100,1) + np.random.rand(100,1)*1j
B=np.random.rand(100,100) + np.random.rand(100,100)*1j
C=np.dot(A.T, B.conj().T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa = 't', opb='c')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 10"
A=np.random.rand(100,1) + np.random.rand(100,1)*1j
B=np.random.rand(100,100) + np.random.rand(100,100)*1j
C=np.dot(A.conj().T, B)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='c')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 11"
A=np.random.rand(100,1) + np.random.rand(100,1)*1j
B=np.random.rand(100,100) + np.random.rand(100,100)*1j
C=np.dot(A.conj().T, B.T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='c', opb='t')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 12"
A=np.random.rand(100,1) + np.random.rand(100,1)*1j
B=np.random.rand(100,100) + np.random.rand(100,100)*1j
C=np.dot(A.conj().T, B.conj().T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='c', opb='c')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()


print "test 13"
A=np.random.rand(1,100) + np.random.rand(1,100)*1j
B=np.random.rand(100,1) + np.random.rand(100,1)*1j
C=np.dot(A, B)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B)
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 14"
A=np.random.rand(100,1) + np.random.rand(100,1)*1j
B=np.random.rand(100,1) + np.random.rand(100,1)*1j
C=np.dot(A.T, B)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='t')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()


print "test 15"
A=np.random.rand(100,1) + np.random.rand(100,1)*1j
B=np.random.rand(100,1) + np.random.rand(100,1)*1j
C=np.dot(A.conj().T, B)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='c')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 16"
A=np.random.rand(100,1) + np.random.rand(100,1)*1j
B=np.random.rand(1,100) + np.random.rand(1,100)*1j
C=np.dot(A.conj().T, B.T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='c', opb='t')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 17"
A=np.random.rand(100,1) + np.random.rand(100,1)*1j
B=np.random.rand(1,100) + np.random.rand(1,100)*1j
C=np.dot(A.conj().T, B.conj().T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='c', opb='c')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 18"
A=np.random.rand(100,1) + np.random.rand(100,1)*1j
B=np.random.rand(1,100) + np.random.rand(1,100)*1j
C=np.dot(A.T, B.conj().T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='t', opb='c')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 19"
A=np.random.rand(1,100) + np.random.rand(1,100)*1j
B=np.random.rand(1,100) + np.random.rand(1,100)*1j
C=np.dot(A, B.conj().T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opb='c')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 20"
A=np.random.rand(1,100) + np.random.rand(1,100)*1j
B=np.random.rand(1,100) + np.random.rand(1,100)*1j
C=np.dot(A, B.T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opb='t')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 21"
A=np.random.rand(100,1)
B=np.random.rand(1,100)
C=np.dot(A.T, B.T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='t', opb='t')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 22"
A=np.random.rand(1,1)+1j*np.random.rand(1,1)
B=np.random.rand(1,100)+1j*np.random.rand(1,100)
C=np.dot(A.conj().T, B)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='c')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 23"
A=np.random.rand(1,1)+1j*np.random.rand(1,1)
B=np.random.rand(100,1)+1j*np.random.rand(100,1)
C=np.dot(A.conj().T, B.conj().T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='c', opb='c')
print np.abs(C-d_C.get()).max()

print "test 24"
A=np.random.rand(100,1)+np.random.rand(100,1)*1j
B=np.random.rand(100,1)+np.random.rand(100,1)*1j
C=np.dot(A, B.T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B,  opb='t')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 25"
A=np.random.rand(100,1)+np.random.rand(100,1)*1j
B=np.random.rand(100,1)+np.random.rand(100,1)*1j
C=np.dot(A, B.conj().T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B,  opb='c')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 26"
A=np.random.rand(1,100)+np.random.rand(1,100)*1j
B=np.random.rand(100,1)+np.random.rand(100,1)*1j
C=np.dot(A.T, B.conj().T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='t', opb='c')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 27"
A=np.random.rand(1,100)+np.random.rand(1,100)*1j
B=np.random.rand(100,1)+np.random.rand(100,1)*1j
C=np.dot(A.conj().T, B.conj().T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='c', opb='c')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()


print "test 28"
A=np.random.rand(1,100)+np.random.rand(1,100)*1j
B=np.random.rand(100,1)+np.random.rand(100,1)*1j
C=np.dot(A.conj().T, B.T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='c', opb='t')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 29"
A=np.random.rand(1,100)+np.random.rand(1,100)*1j
B=np.random.rand(1,100)+np.random.rand(1,100)*1j
C=np.dot(A.conj().T, B)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='c') 
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 30"
A=np.random.rand(100,1)+np.random.rand(100,1)*1j
B=np.random.rand(1,100)+np.random.rand(1,100)*1j
C=np.dot(A, B)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B)
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()


print "test 31"
A=np.random.rand(100,1)
B=np.random.rand(1,100)
C=np.dot(A, B)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B)
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 32"
A=np.random.rand(100,40)+np.random.rand(100,40)*1j
B=np.random.rand(40,100)+np.random.rand(40,100)*1j
C=np.dot(A, B)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B)
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 33"
A=np.random.rand(100,40)+np.random.rand(100,40)*1j
B=np.random.rand(100,40)+np.random.rand(100,40)*1j
C=np.dot(A, B.T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opb='t')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 34"
A=np.random.rand(100,40)+np.random.rand(100,40)*1j
B=np.random.rand(100,40)+np.random.rand(100,40)*1j
C=np.dot(A, B.conj().T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opb='c')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()


print "test 35"
A=np.random.rand(40,100)+np.random.rand(40,100)*1j
B=np.random.rand(40,100)+np.random.rand(40,100)*1j
C=np.dot(A.T, B)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='t')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 36"
A=np.random.rand(40,100)+np.random.rand(40,100)*1j
B=np.random.rand(40,100)+np.random.rand(40,100)*1j
C=np.dot(A.conj().T, B)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='c')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 37"
A=np.random.rand(40,100)+np.random.rand(40,100)*1j
B=np.random.rand(100,40)+np.random.rand(100,40)*1j
C=np.dot(A.conj().T, B.T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='c', opb='t')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 38"
A=np.random.rand(40,100)+np.random.rand(40,100)*1j
B=np.random.rand(100,40)+np.random.rand(100,40)*1j
C=np.dot(A.conj().T, B.conj().T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='c', opb='c')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 39"
A=np.random.rand(100,100)+np.random.rand(100,100)*1j
B=np.random.rand(100,1)+np.random.rand(100,)*1j
C=np.dot(A, B)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='n', opb='n')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()


print "test 40"
A=np.random.rand(100,100)+np.random.rand(100,100)*1j
B=np.random.rand(100,1)+np.random.rand(100,)*1j
C=np.dot(A.T, B)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='t', opb='n')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 41"
A=np.random.rand(100,100)+np.random.rand(100,100)*1j
B=np.random.rand(100,1)+np.random.rand(100,)*1j
C=np.dot(A.conj().T, B)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='c', opb='n')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 42"
A=np.random.rand(100,100)+np.random.rand(100,100)*1j
B=np.random.rand(1,100)+np.random.rand(1,100)*1j
C=np.dot(A.conj().T, B.T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='c', opb='t')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 43"
A=np.random.rand(100,100)+np.random.rand(100,100)*1j
B=np.random.rand(1,100)+np.random.rand(1,100)*1j
C=np.dot(A.conj().T, B.conj().T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='c', opb='c')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 44"
A=np.random.rand(100,100)+np.random.rand(100,100)*1j
B=np.random.rand(1,100)+np.random.rand(1,100)*1j
C=np.dot(A.T, B.conj().T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='t', opb='c')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 45"
A=np.random.rand(100,100)+np.random.rand(100,100)*1j
B=np.random.rand(1,100)+np.random.rand(1,100)*1j
C=np.dot(A.T, B.T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='t', opb='t')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()

print "test 45"
A=np.random.rand(100,100)+np.random.rand(100,100)*1j
B=np.random.rand(1,100)+np.random.rand(1,100)*1j
C=np.dot(A, B.T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='n', opb='t')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()


print "test 46"
A=np.random.rand(100,100)+np.random.rand(100,100)*1j
B=np.random.rand(1,100)+np.random.rand(1,100)*1j
C=np.dot(A, B.conj().T)
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(sl.dot)(d_A,d_B, opa='n', opb='c')
print np.abs(C.real-d_C.get().real).max()
print np.abs(C.imag-d_C.get().imag).max()



def multitest(A, B, opa='n', opb = 'n'):
    for i in range(200):
        C = sl.dot(A, B, opa, opb)
    
    return C
    
def multitest1(A,B,C):
    for i in range(200):
        D = parray.empty_like(C)
        cublas.cublasZgemm('n','n', B.shape[1], A.shape[0], A.shape[1], 1.0, B.gpudata, B.ld, A.gpudata, A.ld, 0, D.gpudata, D.ld)
    return D

"""
A=np.random.rand(2000,1000) + np.random.rand(2000,1000)*1j
B=np.random.rand(1000,1000) + np.random.rand(1000,1000)*1j
d_A=parray.to_gpu(A)
d_B=parray.to_gpu(B)
d_C=func_timer(multitest)(d_A,d_B)
d_D=func_timer(multitest1)(d_A, d_B, d_C)

"""


    




