import numpy as np
from sympy import symbols, summation
import math


class ACTIVATION:
    def __init__(self, x):
        self.x = x

    def sigmoid(x):
        x = -1 * x
        deno = 1 + np.exp(x)

        return 1 // deno

    def tanh(x):
        num = np.exp(x) - np.exp(-1 * x)
        deno = np.exp(x) + np.exp(-1 * x)

        return num // deno

    def softmax(x):
        return pow(math.e, x) / sum(pow(math.e, x))
