import math
import numpy as np
import pdb
from scipy.special import zeta


MAX_ARR = 10**6

class RandomizedTelescopeTemplate(object):

    def sample_vms(self, idx=None, *args, **kwargs):
        """Return a RT-VMS estimate and the compute used for f(i+1) and f(i)"""

        if idx is None:
            idx = self.sample_idx()

        inv_weight = self.inverse_q(idx)

        if self.i0 is not None:
            idx += self.i0

        if self.f_twostep is not None:
            f_iplus1, f_i = self.f_twostep(idx, *args, **kwargs)

        else:
            f_iplus1 = self.f(idx+1, *args, **kwargs)
            f_i = self.f(idx, *args, **kwargs)

        rt_estimator = (f_iplus1 - f_i) / inv_weight
        vms_estimator = (
            rt_estimator * (1/self.N) +
            f_iplus1 * (self.N - 1) / self.N
            )

        if self.i0 is not None:
            vms_estimator += self.f(self.i0, *args, **kwargs)

        return vms_estimator

    def sample_idx(self):
        raise NotImplementedError()


class GeometricRandomizedTelescope(RandomizedTelescopeTemplate):

    def __init__(self, p, f, f_twostep=None, N=1, i0=None,
                 T=None, eps=1e-7, analytic_max_sampling=False):
        '''Unbiased estimation of lim i->T of f(i, inputs) using
        Randomized Telescopes with Virtual Multiple Sampling.

        To recover non-VMS Randomized Telescopes, just use N=1.

        Args:
            f:            function f(i, inputs)

            f_twostep:    function f(i, inputs) which returns
                        (f(i+1, inputs), f(i, inputs)) - this
                        can reduce compute by allowing user to reuse
                        intermediate compute. Either f or f_twostep must
                        be passed - if both are passed we preferentially use
                        f_twostep.
            p:              rate of convergence
            N:              Number of (virtual) samples for Virtual Multiple
                        Sampling. Setting N=1 yields the vanilla
                        Randomized Telescope estimator.
            T:              max iteration of iterative program = max index
                        in sequence / telescoping series. If None, assume
                        T = infinity.
            analytic_max_sampling:  if True, construct the CDF of the maximum
                        and sample. if False, sample N times from the CDF of
                        the N=1 distribution, and take the max.
            i0:           if not None, instead of doing f(i, inputs), do
                        f(i + i0, inputs). When calling the RT estimator,
                        return f(i0, inputs) + the original RT esitmator.
                        This should be used in the case where f(0) is not 0.
        '''
        self.f = f
        self.f_twostep = f_twostep
        self.p = p
        if N < 1:
            raise Exception("Require at least one sample for VMS")
        self.N = N
        if T is not None and T < 1:
            raise Exception("Require horizon of minimum 1")
        self.T = T
        self.i0 = i0
        self.eps = eps
        self.analytic_max_sampling = analytic_max_sampling
        self._construct_importance_weights()


    def sample_idx(self):
        if self.analytic_max_sampling:
            return self._sample_index_analytic_geometric()
        else:
            return self._sample_index_empirical_geometric()


    def _sample_index_analytic_geometric(self):
        '''Sample the maximum index out of N virtual index samples'''
        r = np.random.rand()
        idx = np.searchsorted(self.max_cdf, r)
        if idx >= self.len_probs:
            return max(
                self.len_probs + self._sample_index_empirical_geometric(1),
                self._sample_index_empirical_geometric(self.N - 1)
                )
        else:
            return idx


    def _sample_index_empirical_geometric(self, N=None):
        N = N or self.N
        r = np.random.rand(N)

        n_overflow = np.count_nonzero(r >= self.q_cdf[-1])
        if n_overflow > 0:
            return (self.len_probs +
                    self._sample_index_empirical_geometric(n_overflow))
        else:
            return np.searchsorted(self.q_cdf, r.max())


    def _construct_importance_weights(self):
        '''Construct explicit importance weights for the first
        min(2*m, k, max_arr)
        indexes, where m is the expected maximum, k is the largest
        index with probability (under N=1 vanilla RT) greater than eps,
        and max_arr is a default maximum size for the array.

        We do this pre-construction to improve the speed of sampling. Sampling
        is done using the CDF and binary search. Timing the binary search
        on a macbook gives very little slowdown from 10 to 10^8
        (~3.7e-6s to 2.9e-5) but constructing arrays over 10^7 becomes slow.
        So max_arr is set to 10^6.

        If the sampling routine draws a [0,1] uniform r.v. greater than the
        largest preconstructed CDF value, the sampling routine will fall back
        on an incremental rejection method to sample the correct index.'''
        max_len_probs = MAX_ARR

        if self.T is not None and self.T < max_len_probs:
            self.len_probs = self.T
        else:
            self.len_probs = max_len_probs

        # q is the importance weights for vanilla RT
        indexes = np.cumsum(np.ones(self.len_probs)) - 1.

        # This is the 1-sample RT CDF for the first len_probs indices
        self.q_cdf = (1. - np.power(self.p, indexes + 1))

        # If T < max_len_probs, renormalize so that the cdf[-1] is 1.
        if self.T is not None and self.T <= max_len_probs:
            self.q_cdf = self.q_cdf / self.q_cdf[-1]

        # This is the CDF of the maximum of N samples for the first len_probs
        # indices.
        self.max_cdf = np.power(self.q_cdf, self.N)

        # Use a function instead of array for the inverse of
        # importance weights (as we will only need some indexes)
        self.inverse_q = lambda i: ((1/self.p)**i) / (1-self.p)
        self.inverse_cdf = lambda i: 1. / (1 - self.p ** (i + 1))


    def _pr_max_geometric(self, i, p, N):
        '''Returns the probability the maximum index from N samples is i'''
        return (1 - p**(i+1))**N - (1 - p**i)**N


    def _expected_max_geometric(self, p, N):
        '''Gives an upper bound on the expected max of N samples
        from geometric(p)'''
        return 1 + (1 / - math.log(p)) * (1 + math.log(N))


class PolynomialRandomizedTelescope(RandomizedTelescopeTemplate):

    def __init__(self, p, f, f_twostep=None, N=1, i0=None,
                 T=None, eps=1e-7, analytic_max_sampling=False):
        '''Unbiased estimation of lim i->T of f(i, inputs) using
        Randomized Telescopes with Virtual Multiple Sampling.

        To recover non-VMS Randomized Telescopes, just use N=1.

        Args:
            f:            function f(i, inputs)

            f_twostep:    function f(i, inputs) which returns
                        (f(i+1, inputs), f(i, inputs)) - this
                        can reduce compute by allowing user to reuse
                        intermediate compute. Either f or f_twostep must
                        be passed - if both are passed we preferentially use
                        f_twostep.
            p:              rate of convergence
            N:              Number of (virtual) samples for Virtual Multiple
                        Sampling. Setting N=1 yields the vanilla
                        Randomized Telescope estimator.
            T:              max iteration of iterative program = max index
                        in sequence / telescoping series. If None, assume
                        T = infinity.
            i0:           if not None, instead of doing f(i, inputs), do
                        f(i + i0, inputs). When calling the RT estimator,
                        return f(i0, inputs) + the original RT esitmator.
                        This should be used in the case where f(0) is not 0.
        '''
        self.f = f
        self.f_twostep = f_twostep
        self.p = p
        if N < 1:
            raise Exception("Require at least one sample for VMS")
        self.N = N
        if T is not None and T < 1:
            raise Exception("Require horizon of minimum 1")
        self.T = T
        self.i0 = i0
        self.eps = eps
        self._construct_importance_weights()

    def sample_idx(self, N=None):
        N = N or self.N
        r = np.random.rand(N)
        # pdb.set_trace()
        n_overflow = np.count_nonzero(r >= self.q_cdf[-1])
        if n_overflow > 0:
            return self._fallback_sample_idx(n_overflow)
        else:
            return np.searchsorted(self.q_cdf, r.max())

    def _fallback_sample_idx(self, N):
        idx = len(self.q_cdf) - 1
        past = self.cdf_sum
        while N > 0:
            idx += 1
            idx_prob = 1./(idx + 1)**self.p
            normalized_prob = idx_prob / past
            r = np.random.rand(N)
            N = np.count_nonzero(r >= normalized_prob)
            past += idx_prob
        return idx

    def _construct_importance_weights(self):
        '''Construct explicit importance weights for the first
        max_arr indexes.

        We do this pre-construction to improve the speed of sampling. Sampling
        is done using the CDF and binary search. Timing the binary search
        on a macbook gives very little slowdown from 10 to 10^8
        (~3.7e-6s to 2.9e-5) but constructing arrays over 10^7 becomes slow.
        So max_arr is set to 10^5.

        If the sampling routine draws a [0,1] uniform r.v. greater than the
        largest preconstructed CDF value, the sampling routine will fall back
        on an incremental rejection method to sample the correct index.'''

        if self.T is not None:
            if self.T < MAX_ARR:
                self.len_probs = self.T
            else:
                raise Exception()
        else:
            self.len_probs = MAX_ARR

        # q is the importance weights for vanilla RT
        indexes = np.cumsum(np.ones(self.len_probs)) - 1.

        self.q_pdf = np.power(1./(1 + indexes), self.p)

        if self.T is not None:
            self.normalizer = sum(self.q_pdf)
        else:
            self.normalizer = zeta(self.p, 1)

        self.q_pdf = self.q_pdf / self.normalizer

        # This is the 1-sample RT CDF for the first len_probs indices
        self.q_cdf = np.cumsum(self.q_pdf)


        # Use a function instead of array for the inverse of
        # importance weights (as we will only need some indexes)
        self.inverse_q = lambda idx: (1. + idx)**self.p * self.normalizer
        self.inverse_cdf = lambda idx: (
            1./self.q_cdf[idx] if idx < len(self.q_cdf)
            else 1./(self.q_cdf[-1] +
                     sum(1./(1+i)**self.p
                         for i in range(len(self.q_cdf), idx+1)
                         )/self.normalizer))

        self.cdf_sum = np.sum(self.q_cdf)


class ShuffleRandomizedTelescope(GeometricRandomizedTelescope):
    def __init__(self, p, f, buff_size, f_twostep=None, N=1, i0=None,
                 T=None, eps=1e-7, analytic_max_sampling=False):
        super(ShuffleRandomizedTelescope, self).__init__(p, f, f_twostep,
                                                         N, i0, T, eps,
                                                         analytic_max_sampling)
        self.buff_size = buff_size
        self.buff_idx = 0
        self.make_buff()

    def make_buff(self):
        cdf_positions = np.linspace(0., 1., self.buff_size+2)
        resolution = cdf_positions[1] - cdf_positions[0]
        cdf_positions += np.random.uniform(
            -resolution, resolution, size=len(cdf_positions))
        if cdf_positions[0] < 0.:
            cdf_positions = cdf_positions[1:]
        if cdf_positions[-1] > 1.:
            cdf_positions = cdf_positions[:-1]
        overflow = np.count_nonzero(cdf_positions >= self.q_cdf[-1])
        idxs = np.searchsorted(self.q_cdf, cdf_positions)
        if overflow > 0:
            for j in range(overflow):
                idxs[-(j+1)] = self._fallback_sample_idx(1)
        np.random.shuffle(idxs)
        self.buff = idxs

    def sample_idx(self, N=1):
        assert N == 1, "No VMS yet"
        if self.buff_idx >= len(self.buff):
            self.make_buff()
            self.buff_idx = 0
        sample_idx = self.buff[self.buff_idx]
        self.buff_idx += 1
        return sample_idx
