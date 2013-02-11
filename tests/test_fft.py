#!/usr/bin/env python

import pdb

import numpy as np
import scipy as sp
import scipy.fftpack as fftpack
import pycuda.gpuarray as garray

import ypcutil.setdevice as sd
import ypcutil.parray as parray
import ypcutil.fft as fft

sd.setupdevice(0)
batch = 200
size_x = 100
size_y = 100
size_z = 100


# test single precision 1d

# test 1: even, R2C forward
A = np.random.normal(size = (batch,size_x)).astype(np.float32)
d_A = garray.to_gpu(A)
d_B = fft.fft(d_A)
B = sp.fft(A)
print "test1: ", np.abs(B-d_B.get()).max() 

# test 2: even, C2C inverse
d_C = fft.ifft(d_B)
print "test2: ", np.abs(A-d_C.get()).max()
del d_C

# test 3: even, R2C forward econ
d_B = fft.fft(d_A, econ = True)
BB = d_B.get()
print "test3: ", np.abs(B[:,0:size_x/2+1]-d_B.get()).max()

# test 4: even, C2R inverse
d_C = fft.ifft(d_B, econ=True)
print "test4: ", np.abs(A-d_C.get()).max()

# test 5: even, C2C forward
d_A = garray.to_gpu(A.astype(np.complex64))
d_B = fft.fft(d_A)
print "test5: ", np.abs(B-d_B.get()).max()

# test 6: odd, R2C forward
A = np.random.normal(size = (batch,size_x+1)).astype(np.float32)
d_A = garray.to_gpu(A)
d_B = fft.fft(d_A)
B = sp.fft(A)
print "test6: ", np.abs(B-d_B.get()).max() 

# test 7: odd, C2C inverse
d_C = fft.ifft(d_B)
print "test7: ", np.abs(A-d_C.get()).max()
del d_C

# test 8: odd, R2C foward econ
d_B = fft.fft(d_A, econ = True)
BB = d_B.get()
print "test8: ", np.abs(B[:,0:(size_x+1)/2+1]-d_B.get()).max()

# test 9: odd, C2R inverse
d_C = fft.ifft(d_B, econ=True)
print "test9: ", np.abs(A-d_C.get()).max()

# test 10: odd, C2C forward
d_A = garray.to_gpu(A.astype(np.complex64))
d_B = fft.fft(d_A)
print "test10: ", np.abs(B-d_B.get()).max()

# test double precision 1d

# test 11: even, R2C forward
A = np.random.normal(size = (batch,size_x)).astype(np.float64)
d_A = garray.to_gpu(A)
d_B = fft.fft(d_A)
B = sp.fft(A)
print "test11: ", np.abs(B-d_B.get()).max() 

# test 12: even, C2C inverse
d_C = fft.ifft(d_B)
print "test12: ", np.abs(A-d_C.get()).max()
del d_C

# test 13: even, R2C foward econ
d_B = fft.fft(d_A, econ = True)
BB = d_B.get()
print "test13: ", np.abs(B[:,0:size_x/2+1]-d_B.get()).max()

# test 14: even, C2R inverse
d_C = fft.ifft(d_B, econ=True)
print "test14: ", np.abs(A-d_C.get()).max()

# test 15: even, C2C forward
d_A = garray.to_gpu(A.astype(np.complex128))
d_B = fft.fft(d_A)
print "test15: ", np.abs(B-d_B.get()).max()

# test 16: odd, R2C forward
A = np.random.normal(size = (batch,size_x+1)).astype(np.float64)
d_A = garray.to_gpu(A)
d_B = fft.fft(d_A)
B = sp.fft(A)
print "test16: ", np.abs(B-d_B.get()).max() 

# test 17: odd, C2C inverse
d_C = fft.ifft(d_B)
print "test17: ", np.abs(A-d_C.get()).max()
del d_C

# test 18: odd, R2C forward econ
d_B = fft.fft(d_A, econ = True)
BB = d_B.get()
print "test18: ", np.abs(B[:,0:(size_x+1)/2+1]-d_B.get()).max()

# test 19: odd, C2R inverse
d_C = fft.ifft(d_B, econ=True)
print "test19: ", np.abs(A-d_C.get()).max()

# test 20: odd, C2C forward
d_A = garray.to_gpu(A.astype(np.complex128))
d_B = fft.fft(d_A)
print "test20: ", np.abs(B-d_B.get()).max()


# test single precision 2d

# test 1: even, R2C forward
A = np.random.normal(size=(size_y, size_x)).astype(np.float32)
d_A = garray.to_gpu(A)
d_B = fft.fft2(d_A)
B = fftpack.fft2(A)
print "test1: ", np.abs(B-d_B.get()).max() 

# test 2: even, C2C inverse
d_C = fft.ifft2(d_B)
print "test2: ", np.abs(A-d_C.get()).max()
del d_C

# test 3: even, R2C forward econ
d_B = fft.fft2(d_A, econ = True)
BB = d_B.get()
print "test3: ", np.abs(B[:,0:size_x/2+1]-d_B.get()).max()


# test 4: even, C2R inverse
d_C = fft.ifft2(d_B, econ=True)
print "test4: ", np.abs(A-d_C.get()).max()

# test 5: even, C2C forward
d_A = garray.to_gpu(A.astype(np.complex64))
d_B = fft.fft2(d_A)
print "test5: ", np.abs(B-d_B.get()).max()


# test 6: odd, R2C forward
A = np.random.normal(size=(size_y, size_x+1)).astype(np.float32)
d_A = garray.to_gpu(A)
d_B = fft.fft2(d_A)
B = fftpack.fft2(A)
print "test6: ", np.abs(B-d_B.get()).max() 

# test 7: odd, C2C inverse
d_C = fft.ifft2(d_B)
print "test7: ", np.abs(A-d_C.get()).max()
del d_C

# test 8: odd, R2C foward econ
d_B = fft.fft2(d_A, econ = True)
BB = d_B.get()
print "test8: ", np.abs(B[:,0:(size_x+1)/2+1]-d_B.get()).max()

# test 9: odd, C2R inverse
d_C = fft.ifft2(d_B, econ=True)
print "test9: ", np.abs(A-d_C.get()).max()

# test 10: odd, C2C forward
d_A = garray.to_gpu(A.astype(np.complex64))
d_B = fft.fft2(d_A)
print "test10: ", np.abs(B-d_B.get()).max()


# test double precision 2d

# test 11: even, R2C foward
A = np.random.normal(size=(size_y, size_x)).astype(np.float64)
d_A = garray.to_gpu(A)
d_B = fft.fft2(d_A)
B = fftpack.fft2(A)
print "test11: ", np.abs(B-d_B.get()).max() 

# test 12: even, C2C inverse
d_C = fft.ifft2(d_B)
print "test12: ", np.abs(A-d_C.get()).max()
del d_C

# test 13: even, R2C foward econ
d_B = fft.fft2(d_A, econ = True)
BB = d_B.get()
print "test13: ", np.abs(B[:,0:size_x/2+1]-d_B.get()).max()

# test 14: even, C2R inverse
d_C = fft.ifft2(d_B, econ=True)
print "test14: ", np.abs(A-d_C.get()).max()

# test 15: even, C2C forward
d_A = garray.to_gpu(A.astype(np.complex128))
d_B = fft.fft2(d_A)
print "test15: ", np.abs(B-d_B.get()).max()


# test 16: odd, R2C foward
A = np.random.normal(size=(size_y, size_x+1)).astype(np.float64)
d_A = garray.to_gpu(A)
d_B = fft.fft2(d_A)
B = fftpack.fft2(A)
print "test16: ", np.abs(B-d_B.get()).max() 

# test 17: odd, C2C inverse
d_C = fft.ifft2(d_B)
print "test17: ", np.abs(A-d_C.get()).max()
del d_C

# test 18: odd, R2C foward econ
d_B = fft.fft2(d_A, econ = True)
BB = d_B.get()
print "test18: ", np.abs(B[:,0:(size_x+1)/2+1]-d_B.get()).max()

# test 19: odd, C2R inverse
d_C = fft.ifft2(d_B, econ=True)
print "test19: ", np.abs(A-d_C.get()).max()

# test 20: odd, C2C forward
d_A = garray.to_gpu(A.astype(np.complex128))
d_B = fft.fft2(d_A)
print "test20: ", np.abs(B-d_B.get()).max()


# test single precision 2d

# test 1: even, R2C foward
A = np.random.normal(size=(batch, size_y, size_x)).astype(np.float32)
d_A = garray.to_gpu(A)
d_B = fft.fft2(d_A)
B = fftpack.fft2(A)
print "test1: ", np.abs(B-d_B.get()).max() 

# test 2: even, C2C inverse
d_C = fft.ifft2(d_B)
print "test2: ", np.abs(A-d_C.get()).max()
del d_C

# test 3: even, R2C forward econ
d_B = fft.fft2(d_A, econ = True)
BB = d_B.get()
print "test3: ", np.abs(B[:,:,0:size_x/2+1]-d_B.get()).max()

# test 4: even, C2R inverse
d_C = fft.ifft2(d_B, econ=True)
print "test4: ", np.abs(A-d_C.get()).max()

# test 5: even, C2C forward
d_A = garray.to_gpu(A.astype(np.complex64))
d_B = fft.fft2(d_A)
print "test5: ", np.abs(B-d_B.get()).max()

# test 6: odd, R2C forward
A = np.random.normal(size=(batch, size_y, size_x+1)).astype(np.float32)
d_A = garray.to_gpu(A)
d_B = fft.fft2(d_A)
B = fftpack.fft2(A)
print "test6: ", np.abs(B-d_B.get()).max() 

# test 7: odd, C2C inverse
d_C = fft.ifft2(d_B)
print "test7: ", np.abs(A-d_C.get()).max()
del d_C

# test 8: odd, R2C forward econ
d_B = fft.fft2(d_A, econ = True)
BB = d_B.get()
print "test8: ", np.abs(B[:,:,0:(size_x+1)/2+1]-d_B.get()).max()

# test 9: odd, C2R inverse
d_C = fft.ifft2(d_B, econ=True)
print "test9: ", np.abs(A-d_C.get()).max()

# test 10: odd, C2C forward
d_A = garray.to_gpu(A.astype(np.complex64))
d_B = fft.fft2(d_A)
print "test10: ", np.abs(B-d_B.get()).max()


# test double precision 2d

# test 11: even, R2C foward
A = np.random.normal(size=(batch, size_y, size_x)).astype(np.float64)
d_A = garray.to_gpu(A)
d_B = fft.fft2(d_A)
B = fftpack.fft2(A)
print "test11: ", np.abs(B-d_B.get()).max() 

# test 12: even, C2C inverse
d_C = fft.ifft2(d_B)
print "test12: ", np.abs(A-d_C.get()).max()
del d_C

# test 13: even, R2C foward econ
d_B = fft.fft2(d_A, econ = True)
BB = d_B.get()
print "test13: ", np.abs(B[:,:,0:size_x/2+1]-d_B.get()).max()

# test 14: even, C2R inverse
d_C = fft.ifft2(d_B, econ=True)
print "test14: ", np.abs(A-d_C.get()).max()

# test 15: even, C2C forward
d_A = garray.to_gpu(A.astype(np.complex128))
d_B = fft.fft2(d_A)
print "test15: ", np.abs(B-d_B.get()).max()


# test 16: odd, R2C foward
A = np.random.normal(size=(batch, size_y, size_x+1)).astype(np.float64)
d_A = garray.to_gpu(A)
d_B = fft.fft2(d_A)
B = fftpack.fft2(A)
print "test16: ", np.abs(B-d_B.get()).max() 

# test 17: odd, C2C inverse
d_C = fft.ifft2(d_B)
print "test17: ", np.abs(A-d_C.get()).max()
del d_C

# test 18: odd, R2C foward econ
d_B = fft.fft2(d_A, econ = True)
BB = d_B.get()
print "test18: ", np.abs(B[:,:,0:(size_x+1)/2+1]-d_B.get()).max()

# test 19: odd, C2R inverse
d_C = fft.ifft2(d_B, econ=True)
print "test19: ", np.abs(A-d_C.get()).max()

# test 20: odd, C2C forward
d_A = garray.to_gpu(A.astype(np.complex128))
d_B = fft.fft2(d_A)
print "test20: ", np.abs(B-d_B.get()).max()


