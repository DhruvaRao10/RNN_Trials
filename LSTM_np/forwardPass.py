import numpy as np
import matplotlib as plt
import RNN_Trials.LSTM_np.activation as activation


class LSTM:
    def __init__(self, x, hidden_state, cell_state, ow, iw, fw, cw, ib, fb, ob, cb):  
        self.hidden_state = np.zeros(256, dtype=float)
        self.cell_state = np.zeros(256, dtype=float)
        self.x = np.zeros(256, dtype=float)

        self.iw = np.random.rand(1, 256)
        self.fw = np.random.rand(1, 256)
        self.ow = np.random.rand(1, 256)
        self.cw = np.random.rand(1, 256)

        self.ib = np.random.rand(256, 1)
        self.fb = np.random.rand(256, 1)
        self.ob = np.random.rand(256, 1)
        self.cb = np.random.rand(256, 1)

    def forget_gate(self):
        curr_input = np.concatenate((self.hidden_state, self.x))
        curr_input = np.dot(self.fw.T, curr_input) + self.fb
        curr_input = activation.ACTIVATION.sigmoid(curr_input)

        fg_input = curr_input * self.cell_state

        return fg_input

    def input_gate(self):
        ig_input = np.concatenate((self.hidden_state, self.x))
        ig_input = np.dot(self.iw.T, ig_input) + self.ib
        ig_input = activation.ACTIVATION.sigmoid(ig_input)

        cellstate_input = np.dot(self.cw.T * cellstate_input) + self.cb
        cellstate_input = activation.ACTIVATION.tanh(cellstate_input)

        cellstate_output = cellstate_input * ig_input

        long_term_memory = (
            LSTM.forget_gate(
                self.x, self.hidden_state, self.cell_state, self.fb, self.fw
            )
            + cellstate_output
        )

        return long_term_memory

    def output_gate(self):
        og_input = np.concatenate((self.hidden_state, self.x))
        og_output = np.dot(og_input, self.ow) + self.ob

        hidden_state_output = np.tanh(LSTM.input_gate()) * og_input

        return hidden_state_output
