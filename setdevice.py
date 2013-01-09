#!/usr/bin/env python

import atexit
import pycuda.driver as cuda


def setupdevice(num):
    cuda.init()
    context1 = cuda.Device(num).make_context()
    atexit.register(cuda.Context.pop)
    
