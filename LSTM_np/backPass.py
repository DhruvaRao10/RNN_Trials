import numpy as np
import scipy as sp
import sympy as smp
import matplotlib.pyplot as plt


class BPTT:
    def __init__(self, ow, iw, fw, cw, ib, fb, ob, cb, n_timesteps):
        self.iw = np.random.rand(1, 256)
        self.fw = np.random.rand(1, 256)
        self.ow = np.random.rand(1, 256)
        self.cw = np.random.rand(1, 256)

        self.ib = np.random.rand(256, 1)
        self.fb = np.random.rand(256, 1)
        self.ob = np.random.rand(256, 1)
        self.cb = np.random.rand(256, 1)
        
    
    def output_backprop():
        