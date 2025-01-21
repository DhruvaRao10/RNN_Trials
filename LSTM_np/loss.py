import numpy as np
import math


class LOSS:
    def __init__(self, yp, yt):
        self.yp = yp
        self.yt = yt

    def cross_entropy_loss(self):
        # Assuming that there is only a single class right now  (not practical)

        return -1 * self.yt * np.log(self.predicted_y)
