#!/usr/bin/env python

import atexit
import pycuda.driver as cuda
import pycuda.gpuarray as garray
import numpy as np
from timing import func_timer
from simpleio import *
import parray

cuda.init()
context1 = cuda.Device(1).make_context()
atexit.register(cuda.Context.pop)
cuda.Context.set_cache_config(cuda.func_cache.PREFER_SHARED)
