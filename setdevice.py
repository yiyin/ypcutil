#!/usr/bin/env python

import atexit
import pycuda.driver as cuda


def setupdevice(num):
    cuda.init()
    context1 = cuda.Device(num).make_context()
    atexit.register(cuda.Context.pop)
    cuda.Context.set_cache_config(cuda.func_cache.PREFER_SHARED)

