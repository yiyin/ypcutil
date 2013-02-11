#!/usr/bin/env python

import pdb

import numpy as np
import scipy as sp
import scipy.fftpack as fftpack

import ypcutil.setdevice as sd
import ypcutil.parray as parray
import ypcutil.fft as fft

sd.setupdevice(0)
batch = 16
size_x = 100
size_y = 100
size_z = 100


# test single precision 3d


# test 1: even, R2C forward
A = np.random.normal(size=(size_z, size_y, size_x)).astype(np.float32)
d_A = parray.to_gpu(A.reshape((1, size_z*size_y*size_x)))
d_B = fft.fft3(d_A, shape=(size_z, size_y, size_x))
B = fftpack.fftn(A)
print "test1: ", np.abs(B-d_B.get().reshape((size_z,size_y,size_x))).max() 

# test 2: even, C2C inverse
d_C = fft.ifft3(d_B, shape=(size_z, size_y, size_x))
print "test2: ", np.abs(A-d_C.get().reshape((size_z,size_y,size_x))).max()
del d_C

# test 3: even, R2C forward econ
d_B = fft.fft3(d_A, econ = True, shape=(size_z, size_y, size_x))
BB = d_B.get()
print "test3: ", np.abs(B[:,:,0:size_x/2+1]-d_B.get().reshape((size_z,size_y,size_x/2+1))).max()


# test 4: even, C2R inverse
d_C = fft.ifft3(d_B, econ=True, shape=(size_z, size_y, size_x))
print "test4: ", np.abs(A-d_C.get().reshape((size_z,size_y,size_x))).max()

# test 5: even, C2C forward
d_A = parray.to_gpu(A.reshape((1, size_z*size_y*size_x)).astype(np.complex64))
d_B = fft.fft3(d_A, shape=(size_z, size_y, size_x))
print "test5: ", np.abs(B-d_B.get().reshape((size_z,size_y,size_x))).max()


# test 6: odd, R2C forward
A = np.random.normal(size=(size_z, size_y, size_x+1)).astype(np.float32)
d_A = parray.to_gpu(A.reshape((1, size_z*size_y*(size_x+1))))
d_B = fft.fft3(d_A, shape=(size_z, size_y, size_x+1))
B = fftpack.fftn(A)
print "test6: ", np.abs(B-d_B.get().reshape((size_z,size_y,size_x+1))).max() 

# test 7: odd, C2C inverse
d_C = fft.ifft3(d_B, shape=(size_z, size_y, size_x+1))
print "test7: ", np.abs(A-d_C.get().reshape((size_z,size_y,size_x+1))).max()
del d_C

# test 8: odd, R2C foward econ
d_B = fft.fft3(d_A, econ = True, shape=(size_z, size_y, size_x+1))
BB = d_B.get()
print "test8: ", np.abs(B[:, :,0:(size_x+1)/2+1]-d_B.get().reshape((size_z,size_y,(size_x+1)/2+1))).max()

# test 9: odd, C2R inverse
d_C = fft.ifft3(d_B, econ=True, shape=(size_z, size_y, size_x+1))
print "test9: ", np.abs(A-d_C.get().reshape((size_z,size_y,size_x+1))).max()

# test 10: odd, C2C forward
d_A = parray.to_gpu(A.reshape((1, size_z*size_y*(size_x+1))).astype(np.complex64))
d_B = fft.fft3(d_A, shape=(size_z, size_y, size_x+1))
print "test10: ", np.abs(B-d_B.get().reshape((size_z,size_y,size_x+1))).max()


# test double precision 2d

# test 11: even, R2C foward
A = np.random.normal(size=(size_z, size_y, size_x)).astype(np.float64)
d_A = parray.to_gpu(A.reshape((1, size_z*size_y*size_x)))
d_B = fft.fft3(d_A, shape=(size_z, size_y, size_x))
B = fftpack.fftn(A)
print "test11: ", np.abs(B-d_B.get().reshape((size_z,size_y,size_x))).max() 

# test 12: even, C2C inverse
d_C = fft.ifft3(d_B, shape=(size_z, size_y, size_x))
print "test12: ", np.abs(A-d_C.get().reshape((size_z,size_y,size_x))).max()
del d_C

# test 13: even, R2C foward econ
d_B = fft.fft3(d_A, econ = True, shape=(size_z, size_y, size_x))
BB = d_B.get()
print "test13: ", np.abs(B[:,:,0:size_x/2+1]-d_B.get().reshape((size_z,size_y,size_x/2+1))).max()

# test 14: even, C2R inverse
d_C = fft.ifft3(d_B, econ=True, shape=(size_z, size_y, size_x))
print "test14: ", np.abs(A-d_C.get().reshape((size_z,size_y,size_x))).max()

# test 15: even, C2C forward
d_A = parray.to_gpu(A.reshape((1, size_z*size_y*size_x)).astype(np.complex128))
d_B = fft.fft3(d_A, shape=(size_z, size_y, size_x))
print "test15: ", np.abs(B-d_B.get().reshape((size_z,size_y,size_x))).max()


# test 16: odd, R2C foward
A = np.random.normal(size=(size_z, size_y, size_x+1)).astype(np.float64)
d_A = parray.to_gpu(A.reshape((1, size_z*size_y*(size_x+1))))
d_B = fft.fft3(d_A, shape=(size_z, size_y, size_x+1))
B = fftpack.fftn(A)
print "test16: ", np.abs(B-d_B.get().reshape((size_z,size_y,size_x+1))).max() 

# test 17: odd, C2C inverse
d_C = fft.ifft3(d_B, shape=(size_z, size_y, size_x+1))
print "test17: ", np.abs(A-d_C.get().reshape((size_z,size_y,size_x+1))).max()
del d_C

# test 18: odd, R2C foward econ
d_B = fft.fft3(d_A, econ = True, shape=(size_z, size_y, size_x+1))
BB = d_B.get()
print "test18: ", np.abs(B[:, :,0:(size_x+1)/2+1]-d_B.get().reshape((size_z,size_y,(size_x+1)/2+1))).max()

# test 19: odd, C2R inverse
d_C = fft.ifft3(d_B, econ=True, shape=(size_z, size_y, size_x+1))
print "test19: ", np.abs(A-d_C.get().reshape((size_z,size_y,size_x+1))).max()

# test 20: odd, C2C forward
d_A = parray.to_gpu(A.reshape((1, size_z*size_y*(size_x+1))).astype(np.complex128))
d_B = fft.fft3(d_A, shape=(size_z, size_y, size_x+1))
print "test20: ", np.abs(B-d_B.get().reshape((size_z,size_y,size_x+1))).max()


# test single precision 3d batch

# test 1: even, R2C foward
A = np.random.normal(size=(batch, size_z, size_y, size_x)).astype(np.float32)
d_A = parray.to_gpu(A.reshape((batch, size_z*size_y*size_x)))
d_B = fft.fft3(d_A, shape=(size_z, size_y, size_x))
B = fftpack.fftn(A, axes = [3,2,1])
print "test1: ", np.abs(B-d_B.get().reshape((batch, size_z, size_y, size_x))).max() 

# test 2: even, C2C inverse
d_C = fft.ifft3(d_B, shape=(size_z, size_y, size_x))
print "test2: ", np.abs(A-d_C.get().reshape((batch, size_z, size_y, size_x))).max()
del d_C

# test 3: even, R2C forward econ
d_B = fft.fft3(d_A, econ = True, shape=(size_z, size_y, size_x))
BB = d_B.get()
print "test3: ", np.abs(B[:,:,:,0:size_x/2+1]-d_B.get().reshape((batch, size_z, size_y, size_x/2+1))).max()

# test 4: even, C2R inverse
d_C = fft.ifft3(d_B, econ=True, shape=(size_z, size_y, size_x))
print "test4: ", np.abs(A-d_C.get().reshape((batch, size_z, size_y, size_x))).max()

# test 5: even, C2C forward
d_A = parray.to_gpu(A.reshape((batch, size_z*size_y*size_x)).astype(np.complex64))
d_B = fft.fft3(d_A, shape=(size_z, size_y, size_x))
print "test5: ", np.abs(B-d_B.get().reshape((batch, size_z, size_y, size_x))).max()

# test 6: odd, R2C forward
A = np.random.normal(size=(batch, size_z, size_y, size_x+1)).astype(np.float32)
d_A = parray.to_gpu(A.reshape((batch, size_z*size_y*(size_x+1))))
d_B = fft.fft3(d_A, shape=(size_z, size_y, size_x+1))
B = fftpack.fftn(A, axes = [3,2,1])
print "test6: ", np.abs(B-d_B.get().reshape((batch, size_z, size_y, size_x+1))).max() 

# test 7: odd, C2C inverse
d_C = fft.ifft3(d_B, shape=(size_z, size_y, size_x+1))
print "test7: ", np.abs(A-d_C.get().reshape((batch, size_z, size_y, size_x+1))).max()
del d_C

# test 8: odd, R2C forward econ
d_B = fft.fft3(d_A, econ = True, shape=(size_z, size_y, size_x+1))
BB = d_B.get()
print "test8: ", np.abs(B[:,:,:,0:(size_x+1)/2+1]-d_B.get().reshape((batch, size_z, size_y, (size_x+1)/2+1))).max()

# test 9: odd, C2R inverse
d_C = fft.ifft3(d_B, econ=True, shape=(size_z, size_y, size_x+1))
print "test9: ", np.abs(A-d_C.get().reshape((batch, size_z, size_y, size_x+1))).max()

# test 10: odd, C2C forward
d_A = parray.to_gpu(A.reshape((batch, size_z*size_y*(size_x+1))).astype(np.complex64))
d_B = fft.fft3(d_A, shape=(size_z, size_y, size_x+1))
print "test10: ", np.abs(B-d_B.get().reshape((batch, size_z, size_y, size_x+1))).max()


# test double precision 2d

# test 11: even, R2C foward
A = np.random.normal(size=(batch, size_z, size_y, size_x)).astype(np.float64)
d_A = parray.to_gpu(A.reshape((batch, size_z*size_y*size_x)))
d_B = fft.fft3(d_A, shape=(size_z, size_y, size_x))
B = fftpack.fftn(A, axes = [3,2,1])
print "test11: ", np.abs(B-d_B.get().reshape((batch, size_z, size_y, size_x))).max() 

# test 12: even, C2C inverse
d_C = fft.ifft3(d_B, shape=(size_z, size_y, size_x))
print "test12: ", np.abs(A-d_C.get().reshape((batch, size_z, size_y, size_x))).max()
del d_C

# test 13: even, R2C foward econ
d_B = fft.fft3(d_A, econ = True, shape=(size_z, size_y, size_x))
BB = d_B.get()
print "test13: ", np.abs(B[:,:,:,0:size_x/2+1]-d_B.get().reshape((batch, size_z, size_y, size_x/2+1))).max()

# test 14: even, C2R inverse
d_C = fft.ifft3(d_B, econ=True, shape=(size_z, size_y, size_x))
print "test14: ", np.abs(A-d_C.get().reshape((batch, size_z, size_y, size_x))).max()

# test 15: even, C2C forward
d_A = parray.to_gpu(A.reshape((batch, size_z*size_y*size_x)).astype(np.complex128))
d_B = fft.fft3(d_A, shape=(size_z, size_y, size_x))
print "test15: ", np.abs(B-d_B.get().reshape((batch, size_z, size_y, size_x))).max()


# test 16: odd, R2C foward
A = np.random.normal(size=(batch, size_z, size_y, size_x+1)).astype(np.float64)
d_A = parray.to_gpu(A.reshape((batch, size_z*size_y*(size_x+1))))
d_B = fft.fft3(d_A, shape=(size_z, size_y, size_x+1))
B = fftpack.fftn(A, axes = [3,2,1])
print "test16: ", np.abs(B-d_B.get().reshape((batch, size_z, size_y, size_x+1))).max() 

# test 17: odd, C2C inverse
d_C = fft.ifft3(d_B, shape=(size_z, size_y, size_x+1))
print "test17: ", np.abs(A-d_C.get().reshape((batch, size_z, size_y, size_x+1))).max()
del d_C

# test 18: odd, R2C foward econ
d_B = fft.fft3(d_A, econ = True, shape=(size_z, size_y, size_x+1))
BB = d_B.get()
print "test18: ", np.abs(B[:,:,:,0:(size_x+1)/2+1]-d_B.get().reshape((batch, size_z, size_y, (size_x+1)/2+1))).max()

# test 19: odd, C2R inverse
d_C = fft.ifft3(d_B, econ=True, shape=(size_z, size_y, size_x+1))
print "test19: ", np.abs(A-d_C.get().reshape((batch, size_z, size_y, size_x+1))).max()

# test 20: odd, C2C forward
d_A = parray.to_gpu(A.reshape((batch, size_z*size_y*(size_x+1))).astype(np.complex128))
d_B = fft.fft3(d_A, shape=(size_z, size_y, size_x+1))
print "test20: ", np.abs(B-d_B.get().reshape((batch, size_z, size_y, size_x+1))).max()

