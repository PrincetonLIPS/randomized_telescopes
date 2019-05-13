import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import io
import numpy as np
from tensorflow import flags
import pdb
import math

import tensorflow as tf

import randomized_telescope_runner as runner

import torch
import torch.nn.functional as F
from torch import nn
import torch.distributions as D


FLAGS = flags.FLAGS

flags.DEFINE_integer('train_batch_size', 64, 'batch size')
flags.DEFINE_integer('eval_batch_size', 512, 'batch size')

flags.DEFINE_boolean('clip_intermediate', True,
                     'clip intermediate grads to '
                     'max norm of observed final grad')

flags.DEFINE_float('tmax', 5.0, 'time to integrate to')

flags.DEFINE_float('x0_low', 1.0, 'min initial rabbits')
flags.DEFINE_float('x1_low', 0.4, 'min initial foxes')
flags.DEFINE_float('a_low', 0.8, 'min a')
flags.DEFINE_float('b_low', 0.4, 'min b')
flags.DEFINE_float('c_low', 1.5, 'min c')
flags.DEFINE_float('d_low', 0.4, 'min d')
flags.DEFINE_float('x0_high', 1.5, 'max initial rabbits')
flags.DEFINE_float('x1_high', 0.6, 'max initial foxes')
flags.DEFINE_float('a_high', 1.2, 'max a')
flags.DEFINE_float('b_high', 0.6, 'max b')
flags.DEFINE_float('c_high', 2.0, 'max c')
flags.DEFINE_float('d_high', 0.6, 'max d')

flags.DEFINE_boolean('use_cuda', True, 'use Cuda')

flags.DEFINE_float('x_min', 1e-3, 'min x val')
flags.DEFINE_float('x_softmin', 1e-1, 'softmin x')
flags.DEFINE_float('x_softmax', 1e3, 'softmax x')
flags.DEFINE_float('x_max', 1e5, 'max x')

flags.DEFINE_float('init_std', 0.1, 'init std of normal')

flags.DEFINE_float('noise_std', 0.1, 'noise standard deviation')

flags.DEFINE_integer('observations', 5, 'number of observations')

flags.DEFINE_integer('test_observations', 10000, 'test observations')

flags.DEFINE_float('meta_lr', None, 'meta-optimization learning rate')
flags.DEFINE_float('exp_decay', 0.9, 'exp decay constant')

flags.DEFINE_float('beta1', 0.9, 'adam beta1')
flags.DEFINE_float('beta2', 0.999, 'adam beta2')
flags.DEFINE_float('adam_eps', 1e-8, 'adam eps')

flags.DEFINE_string('optimizer', 'sgd', 'sgd adam or mom')
flags.DEFINE_float('momentum', 0.9, 'momentum for SGD')

flags.DEFINE_float('norm_clip', -1.0, 'clip grads to this norm before doing RT')
flags.DEFINE_float('post_clip', 1.0, 'clip before applying grads')

flags.DEFINE_integer('train_horizon', 9, 'truncated horizon of problem')
flags.DEFINE_integer('test_horizon', 9, 'full horizon of problem')
flags.DEFINE_integer('test_frequency', None, 'test freq')
flags.DEFINE_integer('calibrate_frequency', 5, 'calibrate freq')
flags.DEFINE_boolean('compute_penalty', True, 'penalize RT due to multiple '
                     'computations required')

# if FLAGS.test_horizon > FLAGS.train_horizon, we will take more than this
# many steps using the fully unrolled estimator
# (we would take this * FLAGS.test_horizon**2 / FLAGS.train_horizon**2 steps)
flags.DEFINE_integer('budget', 2000, 'multiple of test_horizon we run for')

flags.DEFINE_integer('seed', 0, 'Random seed for numpy, pytorch and random')


def _cuda(x):
    if not x.is_cuda and FLAGS.use_cuda and torch.cuda.is_available() and isinstance(x, torch.Tensor):
        return x.cuda()
    else:
        return x


class RungeKutta(object):
    def __init__(self, order, A, b, c):
        '''Create RungeKutta method.

        A, b, c define the Butcher tableau.

        A: len-S matrix or listlike, where S is the order.
        b: S-dim vector.
        c: S-dim vector with 0 in first entry.'''
        self.order = order
        assert len(A) == self.order
        assert len(b) == self.order
        assert len(c) == self.order
        self.A = A
        self.b = b
        self.c = c

    def int_step(self, dxdt, x0, t1, t0):
        h = t1 - t0
        ks = []
        for i in range(self.order):
            ti = t0 + self.c[i] * h
            xi = x0
            for j in range(i):
                xi = xi + ks[j] * h * self.A[i][j]
            ks.append(dxdt(xi, ti))
        out = x0
        for i in range(self.order):
            out = out + ks[i] * self.b[i] * h
        #out = torch.clamp(out, FLAGS.x_min , FLAGS.x_max)
        #out[out != out] = FLAGS.x_max
        return out

    def integrate(self, dxdt, x0, ts):
        xs = [x0]
        while len(xs) < len(ts):
            idx = len(xs) - 1
            xnew = self.int_step(dxdt, x0=xs[idx],
                                 t1=ts[idx+1], t0=ts[idx])
            xs.append(xnew)
        return xs


RK4 = RungeKutta(
    order=4,
    A=[[0.,   0.,   0., 0.],
       [1/2., 0.,   0., 0.],
       [0.,   1/2., 0., 0.],
       [0.,   0.,   1., 0.]],
    b=[1/6., 1/3., 1/3., 1/6.],
    c=[0.,   1/2., 1/2., 1.])


class LoktaVolterra(object):
    def __init__(self,
                 a=1., b=0.1, c=1.5, d=0.1):
        self.dx = self.dx_dt
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def dx_dt(self, x, t=0):
        """Growth rate of fox and rabbit populations"""
        x_time_derivative = [
            # Rabbits
            x[:, 0] * (self.a - self.b * x[:, 1]) +
            (
                (x[:, 0] > FLAGS.x_softmax).float() *
                torch.log((FLAGS.x_max - x[:, 0])/(FLAGS.x_max - FLAGS.x_softmax))
            ) -
            (
                (x[:, 0] < FLAGS.x_softmin).float() *
                torch.log((x[:, 0])/(FLAGS.x_softmin))
            ),
            # Foxes
            x[:, 1] * (-self.c + self.d * x[:, 0]) +
            (
                (x[:, 1] > FLAGS.x_softmax).float() *
                torch.log((FLAGS.x_max - x[:, 1])/(FLAGS.x_max - FLAGS.x_softmax))
            ) -
            (
                (x[:, 1] < FLAGS.x_softmin).float() *
                torch.log((x[:, 1])/(FLAGS.x_softmin))
            )
        ]
        return torch.stack(x_time_derivative, dim=1)

    def __call__(self, x, t=0):
        return self.dx_dt(x, t)


def make_problem():
        # Set lr to be optimal val from grid search
    if FLAGS.meta_lr is None:
        if FLAGS.optimizer == 'sgd':
            FLAGS.meta_lr = 1e-2
        elif FLAGS.optimizer == 'mom':
            FLAGS.meta_lr = 3e-3
        elif FLAGS.optimizer == 'adam':
            FLAGS.meta_lr = 3e-2

    if FLAGS.seed is not None:
        np.random.seed(FLAGS.seed)
        torch.manual_seed(FLAGS.seed)

    theta_low = [FLAGS.x0_low, FLAGS.x1_low, FLAGS.a_low, FLAGS.b_low, FLAGS.c_low, FLAGS.d_low]
    theta_high = [FLAGS.x0_high, FLAGS.x1_high, FLAGS.a_high, FLAGS.b_high, FLAGS.c_high, FLAGS.d_high]

    theta_mean = [(tl+th)/2 for tl, th in zip(theta_low, theta_high)]

    if np.any([b<a for b, a in zip(theta_high, theta_low)]):
        print("High val < low val found in args")
        raise Exception("High val < low val found in args")

    prior_mean = theta_mean
    prior_std = [(th-tm) for th, tm in zip(theta_high, prior_mean)]
    # pdb.set_trace()
    # softplus is y = log(exp(x) + 1)
    # so if we want to fix y, set log(exp(y) - 1) = x
    init_mean = [np.log(np.exp(tm) - 1) for tm in theta_mean]
    if FLAGS.init_std > 0.0:
        init_stds = [np.log(np.exp(FLAGS.init_std) - 1) for _ in theta_mean]
        params = [nn.Parameter(_cuda(torch.FloatTensor([x]))) for x in
                  (init_mean + init_stds)]
    else:
        init_stds = [0.0 for _ in range(len(theta_low))]
        params = [nn.Parameter(_cuda(torch.FloatTensor([x]))) for x in
                  init_mean]

    #pdb.set_trace()

    true_theta = _cuda(torch.FloatTensor(np.random.uniform(theta_low, theta_high)))

    t_true = np.linspace(0, FLAGS.tmax, FLAGS.test_observations)

    x0 = true_theta[:2].view([1, -1])

    lv_true = LoktaVolterra(*true_theta[2:])

    #pdb.set_trace()
    x_true = torch.stack(RK4.integrate(lv_true, x0, t_true)).cpu().numpy()
    x_true = np.squeeze(x_true)

    x_true_noise = x_true + np.random.normal(0., FLAGS.noise_std, size=x_true.shape)

    x_test = x_true[::(FLAGS.test_observations)//(FLAGS.observations)]

    x_test_noise = x_true_noise[::(FLAGS.test_observations)//(FLAGS.observations)]

    x_test_nll = _cuda(torch.FloatTensor(x_test_noise)).view([-1, 1, 2])

    x_true_nll = _cuda(torch.FloatTensor(x_true_noise)).view([-1, 1, 2])

    t_test = t_true[::FLAGS.test_observations//FLAGS.observations]

    prior_weight = FLAGS.noise_std**2 / FLAGS.observations

    # pdb.set_trace()

    def nll(x, x_test=x_test_nll, *args):
        x = x.permute(2, 1, 0)
        x_test = x_test.permute(2, 1, 0)
        if x.size()[2] > x_test.size()[2]:
            x = F.interpolate(x, x_test.size()[2])
        else:
            x_test = F.interpolate(x_test, x.size()[2])
        x = x.permute(2, 1, 0)
        x_test = x_test.permute(2, 1, 0)
        mse = torch.mean((x-x_test)**2)
        return mse

    '''
    def kl_divergence(theta_mean, theta_std,
                      prior_mean=_cuda(torch.FloatTensor(prior_mean)),
                      prior_std=_cuda(torch.FloatTensor(prior_std))):
        posterior = D.Normal(loc=theta_mean, scale=theta_std)
        prior = D.Normal(loc=prior_mean, scale=prior_std)
        kl = D.kl_divergence(posterior, prior)
        return torch.sum(kl)
    '''

    def param_log_prob(sample_params,
                prior_mean=_cuda(torch.FloatTensor(prior_mean)),
                prior_std=_cuda(torch.FloatTensor(prior_std))):
        prior = D.Normal(loc=prior_mean, scale=prior_std)
        return torch.mean(prior.log_prob(sample_params))

    def posterior_entropy(theta_mean, theta_std):
        if FLAGS.init_std <= 0.0:
            return torch.mean(_cuda(torch.Tensor([0.0])))
        posterior = D.Normal(loc=theta_mean, scale=theta_std)
        return torch.mean(posterior.entropy())

    def draw_plots(xbatch, title=None):
        ts = np.linspace(0, FLAGS.tmax,
                         len(xbatch[0]))
        fig = plt.figure()
        x = np.mean(xbatch, axis=0)
        if xbatch.shape[0] > 1:
            xhigh = np.percentile(xbatch, 90, axis=0)
            xlow = np.percentile(xbatch, 10, axis=0)
        r, f = x.T
        if xbatch.shape[0] > 1:
            rhigh, fhigh = xhigh.T
            rlow, flow = xlow.T
        rt, ft = x_test.T
        rtrue, ftrue = x_true.T
        ax = fig.gca()
        ax.plot(ts, r, 'r.', label='EstimatedRabbits')
        ax.plot(ts, f, 'b.', label='EstimatedFoxes')
        if xbatch.shape[0] > 1:
            ax.plot(ts, rhigh, 'r-')
            ax.plot(ts, rlow, 'r-')
            ax.plot(ts, fhigh, 'b-')
            ax.plot(ts, flow, 'b-')
            '''
            for i in range(min(xbatch.shape[0], 4)):
                r, f = xbatch[i].T
                ax.plot(ts, r, 'm--', label='SampleRabbits' if i==0 else None)
                ax.plot(ts, f, 'c--', label='SampleFoxes' if i==0 else None)
            '''
        else:
            ax.plot(ts, r, 'r-', label='EstimatedRabbits')
            ax.plot(ts, f, 'b-', label='EstimatedFoxes')
        ax.plot(t_test, rt, 'ro--', label='Observed rabbits')
        ax.plot(t_test, ft, 'bo--', label='Observed foxes')
        ax.plot(t_true, rtrue, 'r--', label='True rabbits')
        ax.plot(t_true, ftrue, 'b--', label='True foxes')
        ax.legend()
        if title is not None:
            plt.title(title)
        fig.canvas.draw()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf

    def train_loss_fn(state, params, horizon):
        old_state = np.random.get_state()
        np.random.set_state(state)
        if FLAGS.init_std > 0.0:
            means = torch.cat(params[:len(params)//2])
            stds = torch.cat(params[len(params)//2:])
            stds = F.softplus(stds)
        else:
            means = torch.cat(params)
            stds = torch.zeros_like(means)
        means = F.softplus(means)
        ts = np.linspace(0, FLAGS.tmax, horizon)
        sample_params = means.view([1, -1]) + _cuda(torch.FloatTensor(
            np.random.normal(
                size=[FLAGS.train_batch_size, len(means)]))) * stds.view([1, -1])
        sample_params = torch.abs(sample_params)
        lv = LoktaVolterra(sample_params[:, 2],
                           sample_params[:, 3],
                           sample_params[:, 4],
                           sample_params[:, 5])
        x0 = sample_params[:, :2]
        xs = torch.stack(RK4.integrate(lv.dx, x0, ts))
        kl_term = posterior_entropy(means, stds) - param_log_prob(sample_params)
        loss = nll(xs) + prior_weight * kl_term
        # pdb.set_trace()
        np.random.set_state(old_state)
        compute = horizon
        return loss, compute

    def make_state_fn(horizon):
        return np.random.RandomState().get_state()

    def log_params(prefix, params, tflogger, step):
        pre = 'param_' + prefix + '_'
        tflogger.log_scalar(pre+'x0[0]', torch.mean(params[0]).item(), step)
        tflogger.log_scalar(pre+'x0[1]', torch.mean(params[1]).item(), step)
        tflogger.log_scalar(pre+'a', torch.mean(params[2]).item(), step)
        tflogger.log_scalar(pre+'b', torch.mean(params[3]).item(), step)
        tflogger.log_scalar(pre+'c', torch.mean(params[4]).item(), step)
        tflogger.log_scalar(pre+'d', torch.mean(params[5]).item(), step)

    def eval_fn(params, horizon, tflogger, step):
        if FLAGS.init_std > 0.0:
            means = torch.cat(params[:len(params)//2])
            stds = torch.cat(params[len(params)//2:])
            stds = F.softplus(stds)
        else:
            means = torch.cat(params)
            stds = torch.zeros_like(means)
        means = F.softplus(means)
        ts = np.linspace(0, FLAGS.tmax, horizon)
        sample_params = means.view([1, -1]) + _cuda(torch.FloatTensor(
            np.random.normal(
                size=[FLAGS.eval_batch_size, len(means)]))) * stds.view([1, -1])
        sample_params = torch.abs(sample_params)

        log_params('means', means, tflogger, step)
        log_params('stds', stds, tflogger, step)
        log_params('sample_mean', torch.mean(sample_params, dim=0), tflogger, step)
        log_params('sample_std', torch.std(sample_params, dim=0), tflogger, step)
        log_params('true', true_theta, tflogger, step)
        lv = LoktaVolterra(sample_params[:, 2],
                           sample_params[:, 3],
                           sample_params[:, 4],
                           sample_params[:, 5])
        x0 = sample_params[:, :2]
        xs = torch.stack(RK4.integrate(lv.dx, x0, ts))
        nll_val = nll(xs, x_true_nll)
        p_z_term = param_log_prob(sample_params)
        h_q_z_term = posterior_entropy(means, stds)
        kl_term = h_q_z_term - p_z_term
        xs = xs.data.cpu().numpy()
        xbatch = np.swapaxes(xs, 0, 1)
        tflogger.log_scalar('min_x', np.min(xbatch), step)
        tflogger.log_scalar('nll', nll_val.data.cpu().numpy(), step)
        tflogger.log_scalar('p(z), z~q', p_z_term.data.cpu().numpy(), step)
        tflogger.log_scalar('H(q(z))', h_q_z_term.data.cpu().numpy(), step)
        tflogger.log_scalar('kld', kl_term.data.cpu().numpy(), step)
        tflogger.log_images('Rabbits and Foxes',
                            [draw_plots(xbatch,
                                        title='Estimated Rabbits and Foxes')],
                            step)
        #posterior = D.Normal(loc=means, scale=stds)
        #log_prob_true_params = torch.sum(posterior.log_prob(true_theta)).data.cpu().numpy()
        posterior = D.Normal(loc=means, scale=stds)
        mean_param_distance = torch.mean((true_theta - means)**2).data.cpu().numpy()
        if FLAGS.noise_std <= 0.0 or FLAGS.init_std <= 0.0:
            log_prob_true_params = -mean_param_distance
        else:
            log_prob_true_params = torch.sum(
                posterior.log_prob(true_theta)).data.cpu().numpy()
        return {
            'nll': nll_val.data.cpu().numpy(),
            'test_elbo': - nll_val.data.cpu().numpy() - prior_weight * kl_term.data.cpu().numpy(),
            'log_prob_true_params': log_prob_true_params,
            'mean_param_distance': mean_param_distance}

    def make_plot_from_batch(params, horizon):
        if FLAGS.init_std > 0.0:
            means = torch.cat(params[:len(params)//2])
            stds = torch.exp(torch.cat(params[len(params)//2:]))
        else:
            means = torch.cat(params)
            stds = torch.zeros_like(means)
        ts = np.linspace(0, FLAGS.tmax, horizon)
        sample_params = means.view([1, -1]) + _cuda(torch.FloatTensor(
            np.random.normal(
                size=[FLAGS.eval_batch_size, len(means)]))) * stds.view([1, -1])
        sample_params = torch.abs(sample_params)
        lv = LoktaVolterra(sample_params[:, 2],
                           sample_params[:, 3],
                           sample_params[:, 4],
                           sample_params[:, 5])
        x0 = sample_params[:, :2]
        xs = torch.stack(RK4.integrate(lv.dx, x0, ts))
        xs = xs.data.cpu().numpy()
        xbatch = np.swapaxes(xs, 0, 1)
        return draw_plots(xbatch, title='Estimated Rabbits and Foxes')

    return true_theta, params, train_loss_fn, make_state_fn, eval_fn, make_plot_from_batch


def main(argv):
    true_params, params, train_loss_fn, make_state_fn, eval_fn, make_plot_from_batch = make_problem()
    runner.run_experiment(
        params=params,
        train_loss_fn=train_loss_fn,
        make_state_fn=make_state_fn,
        eval_fn=eval_fn)


if __name__ == '__main__':
    tf.app.run(main)
