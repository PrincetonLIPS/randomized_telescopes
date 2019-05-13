import numpy as np
import math
from scipy.signal import savgol_filter
import pdb


class NoTelescope(object):
    def __init__(self, horizon):
        self.horizon = horizon
        self.i0 = 0
        self.idxs = [self.horizon - 1]

    def check_calibrate(self):
        return False

    def sample_idx(self):
        return self.horizon - 1

    def weight(self, i, horizon):
        return i == horizon


class CollapsedFixedTelescope(object):
    def __init__(self, q, idxs, roulette=False, var_multiplier=1.0):
        self.horizon = len(q)
        self.i0 = 0
        self.q = np.array(q)
        for i in range(len(q)):
            if i not in idxs:
                self.q[i] = 0.
        self.q = self.q / self.q.sum()
        self.cdf = np.cumsum(self.q)
        self.p_n_geq_i = 1. - self.cdf + self.q
        self.idxs = idxs
        self.roulette = roulette
        self.var_multiplier = var_multiplier


    def check_calibrate(self):
        return False

    def sample_idx(self):
        # pdb.set_trace()
        return np.random.choice(len(self.q), p=self.q)

    def weight(self, i, horizon):
        if i not in self.idxs or horizon-1 not in self.idxs or self.q[horizon-1] < 1e-12:
            print("Invalid stochastic horizon encountered")
            pdb.set_trace()

        if self.roulette:
            return self.var_multiplier * 1./self.p_n_geq_i[i]

        else:
            return self.var_multiplier * 1./self.q[i]


class FixedTelescope(object):
    def __init__(self, q, W):
        self.horizon = len(q)
        self.i0 = 0
        self.q = np.array(q)
        self.q = self.q / self.q.sum()
        self.W = W

    def check_calibrate(self):
        return False

    def sample_idx(self):
        # pdb.set_trace()
        return np.random.choice(len(self.q), p=self.q)

    def weight(self, i, horizon):
        return self.W[i, horizon-1]


class AdaptiveRandomizedTelescope(object):
    def __init__(self, horizon, ema_decay, smooth_window_size, roulette=True):
        self.norms = np.zeros(T)
        self.horizon = horizon
        self.initialized = False
        self.ema_decay = ema_decay
        self.smooth_window_size = smooth_window_size
        self.finalized = False
        self.roulette = roulette

    def calibrate(self, i, last, next, final):
        if self.roulette:
            # Check this
            val = np.sqrt(np.linalg.norm(
                2 * (final - (next + last)/2.) * (next - last)))
        else:
            val = np.linalg.norm(next - last)
        self.finalized = False
        self.norms[i] *= self.ema_decay
        self.norms[i] += (1. - self.ema_decay) * val

    def check_calibrate(self):
        if not self.finalized:
            return True
        else:
            return np.random.rand() < math.log(self.horizon) / self.horizon

    def finalize(self):
        if np.count_nonzero(self.norms) < len(self.norms):
            raise Exception("Have not calibrated all values in array")
        if self.smooth_std is not None:
            norms = savgol_filter(self.norms, self.smooth_window_size, 3)
        else:
            norms = self.norms
        self.pdf = norms / norms.sum()
        self.cdf = np.cumsum(self.pdf)
        self.finalized = True

    def sample_idx(self):
        if not self.finalized:
            raise Exception("Must finalize and construct cdf first")
        r = np.random.rand()
        idx = np.searchsorted(self.cdf, r)
        return idx

    def weight(self, i, horizon):
        if self.finalized:
            if self.roulette:
                pos = 1./(1 - self.cdf[i])
                neg = 0. if i == horizon else 1./(1 - self.cdf[i+1])
                return pos - neg
            else:
                pos = 1./self.pdf[horizon] if i == horizon else 0
                neg = 1./self.pdf[horizon] if i == horizon - 1 else 0
                return pos - neg
        else:
            return 1. if i == horizon else 0.
