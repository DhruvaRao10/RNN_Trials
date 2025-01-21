import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sympy import smp, diff, symbols

import forwardPass

import loss
from activation import ACTIVATION


class GateGrad:
    def __init__(
        self,
        hidden_state,
        prev_cell_state,
        cell_state,
        forget_gate,
        update_gate,
        input_state,
        candidate_state,
        yt,
        yp,
        v,
        loss,
        output_state,
    ):
        self.loss_fun = loss.LOSS.cross_entropy_loss()
        self.zt = v * hidden_state
        f = self.loss_fun
        g = ACTIVATION.softmax(self.zt)
        self.hidden_state = output_state * np.tanh(cell_state)
        self.cell_state = input_state * candidate_state + prev_cell_state * forget_gate
        self.output_state = output_state
        self.input_state = input_state
        self.forget_gate = forget_gate

    def hidden_state_grad(self):
        yt, yp, v, hidden_state = symbols("yt yp v hidden_state")
        dl_dyp = diff(self.f, yp)

        # g = ACTIVATION.softmax(self.zt)
        dyp_dzt = diff(self.g, self.zt)

        dzt_dht = diff(self.zt, hidden_state)

        # dimensions :  a*b  b*c  a*c
        dht = dl_dyp * dyp_dzt * dzt_dht

        return dht

    def cell_state_grad(self):
        yt, yp, v, hidden_state = symbols("yt yp v hidden_state")
        dl_dyp = diff(self.f, yp)

        dyp_dht = diff(self.g, self.hidden_state)

        dht_dct = diff(self.hidden_state, self.cell_state)

        dct = dl_dyp * dyp_dht * dht_dct

        return dct

    def candidate_state_grad(self):
        yt, yp, v, hidden_state = symbols("yt yp v hidden_state")
        dl_dyp = diff(self.f, yp)

        dyp_dht = diff(self.g, self.hidden_state)

        dht_dct = diff(self.hidden_state, self.cell_state)

        dct_dgt = diff(self.cell_state, self.candidate_state)

        dgt = dl_dyp * dyp_dht * dht_dct * dct_dgt

        return dgt

    def output_state_grad(self):
        yt, yp, v, hidden_state = symbols("yt yp v hidden_state")
        dl_dyp = diff(self.f, yp)

        dyp_dht = diff(self.g, self.hidden_state)

        dht_dot = diff(self.hidden_state, self.output_state)

        d_ot = dl_dyp * dyp_dht * dht_dot

        return d_ot

    def input_state_grad(self):
        yt, yp, v, hidden_state = symbols("yt yp v hidden_state")
        dl_dyp = diff(self.f, yp)

        dyp_dht = diff(self.g, self.hidden_state)

        dht_dct = diff(self.hidden_state, self.cell_state)

        dct_dit = diff(self.hidden_state, self.input_state)

        dit = dl_dyp * dyp_dht * dht_dct * dct_dit

        return dit

    def forget_gate_grad(self):
        yt, yp, v, hidden_state = symbols("yt yp v hidden_state")
        dl_dyp = diff(self.f, yp)

        dyp_dht = diff(self.g, self.hidden_state)

        dht_dct = diff(self.hidden_state, self.cell_state)

        dct_dft = diff(self.hidden_state, self.forget_gate)

        dft = dl_dyp * dyp_dht * dht_dct * dct_dft

        return dft
