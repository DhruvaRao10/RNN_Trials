import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sympy import smp, diff, symbols

import forwardPass

from loss import LOSS
from activation import ACTIVATION

from gateGrad import GateGrad


class weightGrad:
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
        ui,
        uf,
        uo,
        wi,
        wf,
        wo,
        loss,
        output_state,
        bi,
        bo,
        bf,
    ):
        self.loss_fun = loss.LOSS.cross_entropy_loss()
        self.v = v
        self.zt = v * hidden_state
        f = self.loss_fun
        g = ACTIVATION.softmax(self.zt)
        self.hidden_state = output_state * np.tanh(cell_state)
        self.cell_state = input_state * candidate_state + prev_cell_state * forget_gate
        self.output_state = LOSS.sigmoid(
            np.dot(uo, input_state) + np.dot(wo, hidden_state) + bi
        )
        self.input_state = LOSS.sigmoid(
            np.dot(ui, input_state) + np.dot(wi, hidden_state) + bi
        )
        self.forget_gate = LOSS.sigmoid(
            np.dot(uf, input_state) + np.dot(wf, hidden_state) + bf
        )
        self.forget_gate = forget_gate
        self.dl_dit = GateGrad.input_state_grad
        self.dl_dft = GateGrad.forget_gate_grad
        self.dl_dot = GateGrad.output_state_grad
        self.dg_dgt = GateGrad.candidate_state_grad

    def output_weight_grad(self):
        yt, yp, v, hidden_state = symbols("yt yp v hidden_state")
        dl_dyp = diff(self.f, yp)

        dyp_dzt = diff(self.g, self.zt)

        dzt_dv = diff(self.zt, self.v)

        dhw = dl_dyp * dyp_dzt * dzt_dv

        return dhw

    def input_weight_grad(self):

        dit_dwi = diff(self.input_state, self.wi)
        dwi = self.dl_dit * dit_dwi

        return dwi

    def forget_weight_grad(self):

        df_dwf = diff(self.forget_gate, self.wf)
        dl_dwf = self.dl_dft * df_dwf

        return dl_dwf

    def output_weight_grad(self):
        do_dwo = diff(self.output_state, self.wo)
        dl_dwo = do_dwo * self.dl_dot

        return dl_dwo

    def candidate_weight_grad(self):
        dg_dgo = diff(self.candidate_state, self.wc)
        dl_dgo = dg_dgo * self.dg_dgt

        return dl_dgo
