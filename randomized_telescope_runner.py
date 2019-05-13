import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorflow import flags
import logging
import os
from randomized_telescope import GeometricRandomizedTelescope, ShuffleRandomizedTelescope
from adaptive_randomized_telescope import NoTelescope, CollapsedFixedTelescope
import pdb
import random
from tensorboard_logger import Logger as TFLogger
import optimize_sampling_greedy
import optimize_sampling_greedy_roulette

from timer import Timer

from setproctitle import setproctitle

import io
import math
import time

import copy
FLAGS = flags.FLAGS

# Toggle these
flags.DEFINE_boolean('rt', False, 'use randomized telescope')
flags.DEFINE_boolean('cdf', False, 'use russian roulette cdf-rt trick')
flags.DEFINE_float('geom_p', 0.5, 'convergence param for geom rt')
# flags.DEFINE_integer('i0', 0, 'starting index for RT')

flags.DEFINE_boolean('partial_update', False, 'do partial updates')

flags.DEFINE_boolean('averaged_test', False, 'use polynomial average of iterates at test')

flags.DEFINE_integer('burn_in', 0,
                     'force using full horizon for this many steps')

flags.DEFINE_boolean('verbose', False, 'log to stderr')

flags.DEFINE_boolean('nesterov', True, 'nesterov momentum')

flags.DEFINE_boolean('cumulative_regret', False,
                     'Optimize cumulative rather than terminal regret')

flags.DEFINE_boolean('stratified_sample', False,
                     'use ShuffleRandomizedTelescope')

flags.DEFINE_boolean('slow_optimizer', False,
                     'only update optimizer on convergence steps')

flags.DEFINE_boolean('scale_intermediate', False,
                     'scale intermediate by k = min(Egi/EgH, 1.0)')

flags.DEFINE_boolean('linear_schedule', False,
                     'Use linear schedule on index')

flags.DEFINE_integer('vms', 1, 'N for vms')

flags.DEFINE_integer('drop_lr_counter_limit', 5, 'N for vms')

flags.DEFINE_boolean('optimize_subseq', True,
                     'Optimize/collapse the telescope')

flags.DEFINE_boolean('force_all_idxs', False, 'force using all indexes')

flags.DEFINE_boolean('drop_lr', False, 'drop lr by factor of half if we '
                     ' havent beaten best previous in 5 tests')

# Maybe sometimes toggle these

flags.DEFINE_string('name', None, 'name of experiment')

flags.DEFINE_string('results_dir', 'results', 'dir for results')

flags.DEFINE_string('f', '', 'kernel - hack to work with jupyter')

flags.DEFINE_float('variance_weight', 1.0, 'Controls weight on variance '
                   'vs compute')

flags.DEFINE_boolean('use_tflogger', False, 'Use tflogger')


def clip_by_norm(array, norm):
    if isinstance(array, list):
        return clip_by_norm_list(array, norm)
    vec = array.reshape([-1])
    if not np.all(np.isfinite(vec)):
        return np.zeros_like(array)
    elif np.linalg.norm(vec) > norm:
        return norm * array / np.linalg.norm(vec)
    else:
        return array

def clip_by_norm_torch(array, norm):
    if isinstance(array, list):
        return clip_by_norm_torch_list(array, norm)
    clip_non_finite_torch(array)
    vec = array.view(-1)
    if vec.norm(p=2).item() > norm:
        return norm * array / vec.norm(p=2).item()
    else:
        return array

def clip_by_norm_list(arraylist, norm):
    vec = np.concatenate([v.reshape([-1]) for v in arraylist])
    vnorm = np.linalg.norm(vec)
    scale = 1. if vnorm < norm else norm / vnorm
    if not np.all(np.isfinite(vec)):
        return [np.zeros_like(v) for v in arraylist]
    else:
        return [v * scale for v in arraylist]

def clip_by_norm_torch_list(arraylist, norm):
    for v in arraylist:
        clip_non_finite_torch(v)
    vec = torch.cat([v.view(-1) for v in arraylist])
    vnorm = vec.norm(p=2).item()
    scale = 1. if vnorm < norm else norm / vnorm
    return [v * scale for v in arraylist]

def make_logger():

    if FLAGS.name is None:
        print("Must set name for experiment")
        raise Exception("Must set name for experiment")

    # --- Paths
    params_name = str(FLAGS.seed) + '_' + str(time.time())

    if not os.path.isdir(FLAGS.results_dir):
        try:
            os.mkdir(FLAGS.results_dir)
        except Exception as e:
            if not os.path.isdir(FLAGS.results_dir):
                raise Exception(e)

    name = FLAGS.name
    if not os.path.isdir(os.path.join(FLAGS.results_dir, name)):
        os.mkdir(os.path.join(FLAGS.results_dir, name))

    path = os.path.join(FLAGS.results_dir, name, params_name)
    if not os.path.isdir(path):
        os.mkdir(path)

    if os.path.exists(os.path.join(path, 'out.log')):
        os.remove(os.path.join(path, 'out.log'))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if FLAGS.verbose:
        logger.addHandler(logging.StreamHandler())

    logger.addHandler(
        logging.FileHandler(os.path.join(path, 'out.log')))
    
    if FLAGS.use_tflogger:
        tflogger = TFLogger(path)
    else:
        class TFLoggerDummy(object):
            def __init__(self):
                pass
            def log_scalar(self, *args, **kwargs):
                pass
            def log_images(self, *args, **kwargs):
                pass
        tflogger = TFLoggerDummy()

    logger.info(str(FLAGS.flag_values_dict()))

    return logger, tflogger


def is_power2(x):
    return x != 0 and ((x & (x - 1)) == 0)

def clip_non_finite_torch(v):
    v[v!=v] = 0.
    v[v == float("Inf")] = 0.
    v[v == -float("Inf")] = 0.

def proc_loss(losses, weight_fn=None, idxs=None):
    for i in range(len(losses)):
        if not np.isfinite(losses[i]):
            losses[i] = 0.0
    if idxs is None:
        idxs = [i for i in range(len(losses))]
    else:
        idxs = [i for i in idxs if i < len(losses)]
    if weight_fn is None:
        loss = losses[-1]
    else:
        loss = 0.
        # pdb.set_trace()
        for i in range(len(idxs)):
            l_i = losses[idxs[i]]
            l_iminus1 = 0. if i < 1 else losses[idxs[i-1]]
            diff = l_i - l_iminus1
            loss = diff * weight_fn(idxs[i], len(losses)) + loss

    return loss

def proc_grads(grads_torch, weight_fn=None, idxs=None):
    for g in grads_torch:
        clip_non_finite_torch(g)
    if idxs is None:
        idxs = [i for i in range(len(grads_torch))]
    else:
        idxs = [i for i in idxs if i < len(grads_torch)]
    if weight_fn is None:
        grad = grads_torch[-1]
    else:
        grad = 0.
        # pdb.set_trace()
        for i in range(len(idxs)):
            g_i = grads_torch[idxs[i]]
            g_iminus1 = 0. if i < 1 else grads_torch[idxs[i-1]]
            diff = g_i - g_iminus1
            grad = diff * weight_fn(idxs[i], len(grads_torch)) + grad

    return grad


def make_img_plot(y, x=None, title=None, xlabel=None, ylabel=None):
    f = plt.figure()
    ax = f.gca()
    if x is None:
        x = np.arange(len(y))
    ax.plot(x, y)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    f.canvas.draw()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf


from matplotlib.colors import Normalize


class Normalizer(Normalize):
    def __init__(self, x):
        r = max(x.max(), -x.min())
        Normalize.__init__(self, -r, r, False)


def optimize_collapsed_telescope(sq_norm_estimates, c,
                                 tflogger=None, logger=None,
                                 step=None):
    # c = [2**i + 1 for i in range(FLAGS.train_horizon + 1)]
    idxs = [i for i in range(FLAGS.train_horizon + 1)]
    if FLAGS.cdf:
        idxs, qs = optimize_sampling_greedy_roulette.optimize_greedy_roulette(
            sq_norm_estimates, c, idxs, verbose=FLAGS.verbose, logger=logger)
        cost, cval, vval = optimize_sampling_greedy_roulette.cost(sq_norm_estimates,
                                                         c, idxs,
                                                         return_cv=True)
    else:
        idxs, qs = optimize_sampling_greedy.optimize_greedy(
            sq_norm_estimates, c, idxs, verbose=FLAGS.verbose, logger=logger)
        cost, cval, vval = optimize_sampling_greedy.cost(sq_norm_estimates,
                                                c, idxs,
                                                return_cv=True)

    base_c = c[-1]
    base_v = sq_norm_estimates[0, -1]
    base_cost = base_c * base_v
    var_multiplier = vval / base_v
    idx_to_q = dict(zip(idxs, qs))
    qs = [idx_to_q.get(i, 0.0) for i in range(FLAGS.train_horizon + 1)]
    if step is not None and tflogger is not None:
        tflogger.log_scalar('Relative efficiency lower bound',
                            cost/base_cost, step)
        tflogger.log_scalar('Relative expected compute',
                            cval/base_c, step)
        tflogger.log_scalar('Relative expected variance',
                            vval/base_v, step)
        tflogger.log_images('qs_optimized',
                            [make_img_plot(qs, title='q_by_idx')], step)
    telescope = CollapsedFixedTelescope(qs, idxs, roulette=FLAGS.cdf,
                                        var_multiplier=var_multiplier)
    weight_fn = telescope.weight
    return telescope, weight_fn


def make_w_plot(w, title=None):
    f = plt.figure()
    plt.imshow(w, norm=Normalizer(w), interpolation='nearest',
               cmap='seismic')
    plt.colorbar()
    if title:
        plt.title(title)
    f.canvas.draw()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf


def make_telescope_and_weight():
    T = FLAGS.train_horizon + 1

    if FLAGS.vms > 1 and FLAGS.cdf:
        raise NotImplementedError()

    if not FLAGS.rt:
        weight_fn = None
        telescope = NoTelescope(T)

    else:
        if FLAGS.stratified_sample:
            telescope = ShuffleRandomizedTelescope(FLAGS.geom_p, f=None,
                                                   N=FLAGS.vms, i0=0,#FLAGS.i0,
                                                   T=T, buff_size=100)
        else:
            telescope = GeometricRandomizedTelescope(FLAGS.geom_p, f=None,
                                                     N=FLAGS.vms, i0=0,#FLAGS.i0,
                                                     T=T)
        if FLAGS.cdf:
            def weight_fn(i, horizon):
                if i == 0:
                    return 1.
                else:
                    p_n_lessthan_i = 1. / telescope.inverse_cdf(i - 1)
                    return 1. / (1 - p_n_lessthan_i)

        else:
            def weight_fn(i, horizon):
                return (1 - 1 / FLAGS.vms) + (1 / FLAGS.vms) * telescope.inverse_q(i) * (i == horizon - 1)

    return telescope, weight_fn


def loss_and_grads(loss_fn, state, params, optimizer, i):
    optimizer.zero_grad()
    loss, compute = loss_fn(state, params, i)
    loss.backward()
    grads = []
    for p in params:
        g = p.grad.clone().detach()
        clip_non_finite_torch(g)
        grads.append(g)
    if FLAGS.norm_clip > 0:
        grads = clip_by_norm_torch(grads, FLAGS.norm_clip)
    loss = loss.data.cpu().numpy()
    loss[not np.isfinite(loss)] = 0.0
    return loss, grads, compute


def deltas_from_grads_torch(grads_per_idx_torch):
    '''grads_per_idx_torch: list n_idxs of list n_params'''
    sq_norms = np.zeros([len(grads_per_idx_torch) + 1, len(grads_per_idx_torch)])
    for i in range(len(grads_per_idx_torch) + 1):
        g1 = [0. for _ in grads_per_idx_torch[0]] if i == 0 else grads_per_idx_torch[i - 1]
        for j in range(i, len(grads_per_idx_torch)):
            g2 = grads_per_idx_torch[j]
            diffs = [(g2_ - g1_) for g2_, g1_ in zip(g2, g1)]
            for d in diffs:
                clip_non_finite_torch(d)
            if FLAGS.post_clip > 0:
                diffs = clip_by_norm_torch(diffs, FLAGS.post_clip)
            param_diff_sum_sq = [(d**2).sum().item() for d in diffs]
            sq_norms[i, j] = sum(param_diff_sum_sq) + 1e-9
    return sq_norms

def deltas_from_grads(grads_per_idx):
    '''grads_per_param: n_params x n_idxs'''
    grads_per_idx = np.array(grads_per_idx)
    grads_per_idx = np.reshape(grads_per_idx, [grads_per_idx.shape[0], -1])
    sq_norms = np.zeros([len(grads_per_idx) + 1, len(grads_per_idx)])
    for i in range(len(grads_per_idx) + 1):
        v1 = 0. if i == 0 else grads_per_idx[i - 1]
        for j in range(i, len(grads_per_idx)):
            v2 = grads_per_idx[j]
            v2[np.logical_not(np.isfinite(v2))] = 0.0
            sq_norms[i, j] = np.linalg.norm(
                v2 - v1) ** 2 + 1e-12
    return sq_norms


def cosines_from_grads(grads_per_idx):
    '''grads_per_param: n_params x n_idxs'''
    grads_per_idx = np.array(grads_per_idx)
    grads_per_idx = np.reshape(grads_per_idx, [grads_per_idx.shape[0], -1])
    cosines = np.zeros([len(grads_per_idx) + 1, len(grads_per_idx)])
    for i in range(len(grads_per_idx) + 1):
        for j in range(i, len(grads_per_idx)):
            v2 = grads_per_idx[j]
            v1 = np.zeros_like(v2) if i == 0 else grads_per_idx[i - 1]
            cosines[i, j] = np.dot(v1, v2) / (
                np.linalg.norm(v1)*np.linalg.norm(v2)+1e-12
            )
    return cosines


class RunningNorms(object):
    def __init__(self, horizon):
        self.estimated_norms = np.zeros([horizon+1, horizon])
        self.ts = np.zeros([horizon+1, horizon])

    def update(self, sq_norms):
        self.estimated_norms[:sq_norms.shape[0], :sq_norms.shape[1]] = (
            FLAGS.exp_decay * self.estimated_norms[:sq_norms.shape[0], :sq_norms.shape[1]] +
            (1. - FLAGS.exp_decay) * sq_norms)
        self.ts[:sq_norms.shape[0], :sq_norms.shape[1]] += 1.

    def get_norms(self):
        return self.estimated_norms / (1. - FLAGS.exp_decay**self.ts)

'''
def update_running_dnorm(running_dnorm, sq_norms):
    sq_norms = np.array(sq_norms)
    running_dnorm[:sq_norms.shape[0], :sq_norms.shape[1]] = (
        FLAGS.exp_decay * running_dnorm[:sq_norms.shape[0], :sq_norms.shape[1]] +
        (1. - FLAGS.exp_decay) * sq_norms)
    return running_dnorm
'''
def do_convergence_update(grads_torch, params, optimizer):
    grads_per_param = [[g[pidx] for g in grads_torch]
           for pidx in range(len(params))]
    processed_grads = [
        proc_grads(g, None)
        for g in grads_per_param]

    if FLAGS.post_clip > 0:
        processed_grads = clip_by_norm_torch(processed_grads, FLAGS.post_clip)

    optimizer.zero_grad()

    for p, g in zip(params, processed_grads):
        p.grad = g

    optimizer.step()


def run_experiment(params, train_loss_fn, eval_fn, make_state_fn):
    setproctitle(FLAGS.name)

    logger, tflogger = make_logger()

    running_dnorm = RunningNorms(FLAGS.train_horizon+1)

    weight_decay = 0. if not hasattr(FLAGS, 'weight_decay') else FLAGS.weight_decay
    #if FLAGS.variance_weight < 1.0 and FLAGS.rt:
    #    FLAGS.meta_lr = FLAGS.meta_lr * FLAGS.variance_weight
    #if FLAGS.rt:
    #    FLAGS.meta_lr = FLAGS.meta_lr / 2.0

    if FLAGS.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=FLAGS.meta_lr,
                                     betas=(FLAGS.beta1, FLAGS.beta2),
                                     eps=FLAGS.adam_eps,
                                     weight_decay=weight_decay)
    elif FLAGS.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=FLAGS.meta_lr,
                                    weight_decay=weight_decay)
    elif FLAGS.optimizer == 'mom':
        optimizer = torch.optim.SGD(params, lr=FLAGS.meta_lr,
                                    weight_decay=weight_decay,
                                    momentum=FLAGS.momentum,
                                    nesterov=(FLAGS.nesterov and
                                              FLAGS.momentum > 0.0)
                                    )
    total_losses = []
    total_compute = []

    if hasattr(FLAGS, 'lr_drop_computes') and not FLAGS.drop_lr and FLAGS.lr_drop_computes is not None:
        lr_drop_computes = [int(s) for s in FLAGS.lr_drop_computes.split(',')]
    else:
        lr_drop_computes = []

    test_param_accumulators = []
    for p in params:
        test_param_accumulators.append(p.data.clone())

    test_counter = 0
    convergence_counter = 0
    total_loss = 0.
    compute = 0
    idx_counter = 0

    best_loss_so_far = None
    drop_counter = 0

    last_good_params = copy.deepcopy(params)

    max_norm = None

    step = 0

    test_stats = None

    eval_headers = None

    telescope = NoTelescope(FLAGS.train_horizon+1)
    weight_fn = None

    if FLAGS.test_frequency is None:
        FLAGS.test_frequency = 1

    tflogger.log_scalar('meta_lr_by_compute',
                        optimizer.param_groups[0]['lr'],
                        int(compute/(2**FLAGS.train_horizon+1)))
    logger.info("Running until compute > {}".format(
                2**FLAGS.test_horizon * FLAGS.budget))
    while idx_counter < 2**FLAGS.test_horizon * FLAGS.budget:

        if FLAGS.linear_schedule:
            assert not FLAGS.rt
            n_increments = FLAGS.train_horizon + 1 - 2
            increment_size = float(2**FLAGS.test_horizon * FLAGS.budget) / n_increments
            increment_number = int(math.floor(idx_counter / increment_size)) + 2
            telescope = NoTelescope(increment_number+1)

        if len(lr_drop_computes) > 0 and idx_counter > lr_drop_computes[0]:
            for pg in optimizer.param_groups:
                pg['lr'] /= 10.
            logger.info("Dropping lr to {}".format(optimizer.param_groups[0]['lr']))
            tflogger.log_scalar('meta_lr_by_compute',
                                optimizer.param_groups[0]['lr'],
                                int(compute/(2**FLAGS.train_horizon+1)))
            if len(lr_drop_computes) > 1:
                lr_drop_computes = lr_drop_computes[1:]
            else:
                lr_drop_computes = []

        if idx_counter >= test_counter:
            with Timer() as t:
                # logger.info('Evaluating...')
                if FLAGS.averaged_test:
                    testparams = test_param_accumulators
                else:
                    testparams = params
                #try:
                test_stats = eval_fn(
                    testparams, 2**(FLAGS.test_horizon) + 1,
                    tflogger, step)

                state = make_state_fn(2**FLAGS.test_horizon+1)
                test_horizon_loss, _ = train_loss_fn(state, params,
                                                     2**FLAGS.test_horizon+1)
                test_horizon_loss = test_horizon_loss.item()
                optimizer.zero_grad()

                if eval_headers is None:
                    eval_headers = sorted(list(test_stats.keys()))
                #except Exception as e:
                #    pdb.set_trace()
                #    test_stats = eval_fn(
                #        testparams, 2**(FLAGS.test_horizon) + 1,
                #        tflogger, step)
                test_counter += (2**FLAGS.test_horizon + 1) * FLAGS.test_frequency
                for k in eval_headers:
                    v = test_stats[k]
                    tflogger.log_scalar(k, v, step)
                    tflogger.log_scalar(k + '_by_compute', v,
                                        int(compute / (2**FLAGS.test_horizon + 1)))
                    if not np.isfinite(v):
                        logger.info("NaN test stat, reverting params")
                        params = last_good_params
                    # logger.info('full horizon loss: %f' % total_loss)
                # eval_headers = [k for k in test_stats]
                last_good_params = copy.deepcopy(params)
            logger.info("Test time: {}".format(t.interval))

        if idx_counter >= convergence_counter:
            with Timer() as t:
                losses = []
                grads_torch = []
                grads = []
                computes = []
                with Timer() as t2:
                    state = make_state_fn(2**FLAGS.train_horizon+1)
                    for j in range(FLAGS.train_horizon + 1):
                        l, g, c = loss_and_grads(
                            train_loss_fn, state,
                            params, optimizer, 2**(j) + 1)
                        if j > 0 and FLAGS.cumulative_regret:
                            l = l + losses[j-1]
                            g = [g1 + g2 for g1, g2 in zip(g, grads_torch[j-1])]
                        losses.append(l)
                        grads_torch.append(g)
                        #grads.append([gi.cpu().numpy() for gi in g])
                        computes.append(c)

                    tflogger.log_scalar('train_horizon_loss_by_compute',
                                        losses[-1],
                                        int(compute/(2**FLAGS.test_horizon+1)))

                    if FLAGS.drop_lr:
                        if best_loss_so_far is None or losses[-1] <= best_loss_so_far:
                            logger.info("Loss {} better than best {}".format(
                                losses[-1], best_loss_so_far))
                            best_loss_so_far = losses[-1]
                            drop_counter = 0
                        else:
                            logger.info("Loss {} not better than best {}".format(
                                losses[-1], best_loss_so_far))
                            logger.info("Incrementing drop_counter to {}".format(
                                drop_counter))
                            drop_counter += 1
                            if drop_counter >= FLAGS.drop_lr_counter_limit:
                                drop_counter = 0
                                best_loss_so_far = losses[-1]
                                for pg in optimizer.param_groups:
                                    pg['lr'] /= 2.
                                logger.info("Dropping lr to {}".format(optimizer.param_groups[0]['lr']))
                        tflogger.log_scalar('meta_lr_by_compute',
                                            optimizer.param_groups[0]['lr'],
                                            int(compute/(2**FLAGS.test_horizon+1)))
                    convergence_outputs = measure_convergence(
                        losses, grads_torch, params, telescope,
                        weight_fn, optimizer, running_dnorm, tflogger, logger,
                        step, max_norm, computes, compute
                    )
                logger.info("Convergence measurement time: {}".format(t2.interval))
                with Timer() as t2:
                    do_convergence_update(grads_torch, params, optimizer)

                    telescope, weight_fn, running_dnorm, max_norm = (
                        convergence_outputs
                    )
                    convergence_counter += 2**FLAGS.train_horizon * FLAGS.calibrate_frequency + 1
                    if FLAGS.rt:
                        compute = compute + sum(computes)
                logger.info("Convergence update time: {}".format(t2.interval))
            logger.info("Convergence time: {}".format(t.interval))

        with Timer() as t:
            log = str(step)
            optimizer.zero_grad()
            if step >= FLAGS.burn_in:
                idx = telescope.sample_idx()
            else:
                idx = FLAGS.train_horizon

            # logger.info("horizon: %i, index: %i" % (FLAGS.test_horizon, idx+1))
            log += ",{}".format(idx)
            tflogger.log_scalar('idx', idx, step)

            losses = []
            grads_torch = []
            computes = []
            state = make_state_fn(2**idx + 1)
            for j in range(idx + 1):
                if j in telescope.idxs:
                    l, g, c = loss_and_grads(
                        train_loss_fn, state,
                        params, optimizer, 2**(j) + 1)
                else:
                    l = 0.0
                    g = [torch.zeros_like(p.data)
                         for p in params]
                    c = 0
                if j > 0 and FLAGS.cumulative_regret:
                    l = l + losses[j-1]
                    g = [g1 + g2 for g1, g2 in zip(g, grads[j-1])]
                losses.append(l)
                grads_torch.append(g)
                computes.append(c)
            #pdb.set_trace()
            if step >= FLAGS.burn_in:
                loss = proc_loss(losses, weight_fn,
                                  idxs=telescope.idxs if FLAGS.optimize_subseq
                                  and FLAGS.rt
                                  else None)
            else:
                loss = proc_loss(losses, None)

            if FLAGS.clip_intermediate:
                grads_torch = [clip_by_norm_torch(gpi, max_norm)
                                 for gpi in grads_torch]

            grads_per_param = [[g[pidx] for g in grads_torch]
                   for pidx in range(len(params))]

            if step >= FLAGS.burn_in:
                processed_grads = [proc_grads(g, weight_fn,
                                   idxs=telescope.idxs if FLAGS.optimize_subseq
                                   and FLAGS.rt else None)
                                   for g in grads_per_param]
            else:
                processed_grads = [proc_grads(g, None)
                                   for g in grads_per_param]

            '''
            if idx == FLAGS.train_horizon:
                convergence_outputs = measure_convergence(
                    losses, grads, params, telescope,
                    weight_fn, optimizer, running_dnorm, tflogger, logger,
                    step, max_norm
                )
                telescope, weight_fn, running_dnorm, max_norm = (
                    convergence_outputs
                )
            elif FLAGS.partial_update:
                flat_grads_per_idx = [np.concatenate([
                    grads_per_param[pidx][idx].flatten()
                    for pidx in range(len(params))])
                    for idx in range(len(losses))]
                sq_norms = deltas_from_grads(flat_grads_per_idx)
                running_dnorm.update(sq_norms)

                if FLAGS.optimize_subseq and FLAGS.rt:
                    telescope, weight_fn = optimize_collapsed_telescope(
                        running_dnorm.get_norms())
            '''

            if FLAGS.post_clip > 0:
                processed_grads = clip_by_norm_torch(processed_grads, FLAGS.post_clip)

            optimizer.zero_grad()
            for p, g in zip(params, processed_grads):
                p.grad = g
            if FLAGS.slow_optimizer:
                _opt_state = copy.deepcopy(optimizer.state_dict)
            optimizer.step()
            if FLAGS.slow_optimizer:
                optimizer.load_state_dict(_opt_state)

            idx_costs = [2**i+1 for i in range(len(computes))]
            if FLAGS.rt and FLAGS.partial_update:
                compute = compute + sum(computes)
                idx_counter = idx_counter + sum(idx_costs)


            elif FLAGS.rt and FLAGS.compute_penalty and step >= FLAGS.burn_in:
                if FLAGS.cdf:
                    if FLAGS.optimize_subseq:
                        compute = compute + sum(computes)
                        idx_counter = idx_counter + sum(
                            idx_costs[j] for j in telescope.idxs if j <= idx
                        )
                    else:
                        compute = compute + sum(computes)
                        idx_counter = idx_counter + sum(idx_costs)

                else:
                    compute = compute + computes[-1]
                    idx_counter = idx_counter + idx_costs[-1]
                    if FLAGS.optimize_subseq:
                        idx2 = idx - 1
                        while idx2 >= 0:
                            if idx2 in telescope.idxs:
                                compute = compute + computes[idx2]
                                idx_counter = idx_counter + idx_costs[idx2]
                                idx2 = -1
                            idx2 -= 1
                    else:
                        compute = compute + (0. if idx < 1 else computes[idx-1])
                        idx_counter = idx_counter + (0. if idx < 1 else idx_costs[idx-1])


            else:
                compute = compute + computes[idx]
                idx_counter = idx_counter + idx_costs[idx]

            for k in eval_headers:
                v = test_stats[k]
                log += ",{}".format(v)
            log += ",{}".format(test_horizon_loss)

            total_losses.append(total_loss)
            total_compute.append(compute)
            # logger.info('total compute: %i' % total_compute[-1])
            log += ",{}".format(total_compute[-1])
            log += ",{}".format(idx_counter)

            tflogger.log_scalar('total_compute', total_compute[-1], step)

            if not np.isnan(loss):
                tflogger.log_scalar('loss', loss, step)
                # raise Exception("NaN loss found")
            # print('loss2: %f' % np.sum(A_numpy[-1]))
            # logger.info('lr: %f' % lr.data.cpu().numpy())
            # logger.info('mom: %f' % mom.data.cpu().numpy())
            log += ",{}".format(loss)

            log += ",{}".format(params[0].data.cpu().numpy().mean())
            '''
            for pidx, p in enumerate(params):
                tflogger.log_scalar('param_' + str(pidx),
                                    p.data.cpu().numpy(), step)
            '''

            # print(A_numpy.T[-2:])
            # logger.info("\n")
            if step == 0:
                logger.info("iter,idx," + ','.join(eval_headers) +
                            ",test_horizon_loss,total_compute,idx_counter,cost,first_param_mean")
            logger.info(log)
            step += 1
            for i, p in enumerate(params):
                test_param_accumulators[i] = (
                    (1 - 2/(step+2)) * test_param_accumulators[i] +
                    (2/(step+2)) * p.data.clone()
                )
        logger.info("Step time: {}".format(t.interval))
    logger.info("Completed run with compute {}".format(compute))


def measure_convergence(losses, grads_per_idx_torch, params, telescope,
                        weight_fn, optimizer,
                        running_dnorm, tflogger, logger, step, max_norm,
                        computes, compute):
    '''MEASURE CONVERGENCE'''
    # logger.info('Measuring convergence...')
    # N_VARIABLES = sum(p.numel() for p in params)
    '''
    grads_per_idx = []
    for g_idx in grads:
        grads_per_idx.append(np.concatenate([g.reshape([-1]) for g in g_idx]))
    grads_per_idx = np.array(grads_per_idx)
    grads_per_param = grads_per_idx.swapaxes(0, 1)
    '''

    sq_norms = deltas_from_grads_torch(grads_per_idx_torch)
    final_grad_norm = min(np.sqrt(sq_norms[0, -1]), 1e3)

    tflogger.log_scalar('final grad norm', final_grad_norm, step)
    tflogger.log_scalar(
        'exp moving average of final grad norm', max_norm, step)

    # cosines = cosines_from_grads(grads_per_idx_torch)
    if FLAGS.clip_intermediate and max_norm:
        clipped_grads_per_idx = grads_per_idx_torch
        for i in range(len(clipped_grads_per_idx)):
            clipped_grads_per_idx[i] = clip_by_norm_torch(
                clipped_grads_per_idx[i], max_norm)
    else:
        clipped_grads_per_idx = grads_per_idx_torch

    clipped_norms = deltas_from_grads_torch(clipped_grads_per_idx)
    # clipped_cosines = cosines_from_grads(clipped_grads_per_idx)

    tflogger.log_images('loss',
                        [make_img_plot(losses, title='loss by unroll step')], step)

    tflogger.log_images('computes',
                        [make_img_plot(computes, title='compute by unroll step')], step)

    tflogger.log_images(
        'delta norms',
        [make_w_plot(sq_norms,
                     title='Norms of g_{i-1} - g_j,   g_{-1} := 0')],
        step)

    tflogger.log_images(
        'clipped delta norms',
        [make_w_plot(clipped_norms,
                     title='Norms of g_{i-1} - g_j,   g_{-1} := 0')],
        step)

    for pidx in range(len(params)):
        if pidx > 10:
            break
        tflogger.log_scalar('param_' + str(pidx) + '_by_compute',
                            params[pidx].data.sum().item(),
                            int(compute/(2**FLAGS.test_horizon+1)))
        if FLAGS.optimizer == 'mom':
            state = optimizer.state[params[pidx]]
            if 'momentum_buffer' not in state:
                vel = 0.0
            else:
                vel = state['momentum_buffer'].sum().item()
            tflogger.log_scalar('param_vel_' + str(pidx) + '_by_compute',
                                vel,
                                int(compute/(2**FLAGS.test_horizon+1)))
        tflogger.log_images('param_' + str(pidx) + '_grad',
                            [make_img_plot(
                                [gpi[pidx].sum().item() for gpi in grads_per_idx_torch],
                                title='param_' + str(pidx) + '_grad by unroll step')],
                            step)

    running_dnorm.update(clipped_norms)

    if FLAGS.optimize_subseq and FLAGS.rt:
        telescope, weight_fn = optimize_collapsed_telescope(
            running_dnorm.get_norms(), computes, tflogger, logger, step)

    if np.any(np.less(running_dnorm.get_norms(), 0.)):
        print("Norms less than 0.")
        raise Exception("Norms less than 0.")

    tflogger.log_images(
        'running_norm_estimates',
        [make_w_plot(running_dnorm.get_norms(),
                     title='Running average of squared norms g_i - g_j')],
        step)

    valid_dnorms = "\n".join([str(d[i:])
                    for i, d in enumerate(running_dnorm.get_norms()[:-1])])
    logger.info("Running dnorms: \n" + valid_dnorms)
    logger.info("Computes: " + str(computes))

    if max_norm is None and final_grad_norm > 1e-14:
        max_norm = final_grad_norm
    elif final_grad_norm > 1e-14:
        max_norm = FLAGS.exp_decay * max_norm + (1. - FLAGS.exp_decay) * final_grad_norm

    return telescope, weight_fn, running_dnorm, max_norm

    # --------------------------------------------------
