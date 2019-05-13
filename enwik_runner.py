import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tensorflow import flags
import pdb
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data
from util import suppress_stderr, suppress_stdout

import tensorflow as tf

import io
import math

import os
import hashlib

import randomized_telescope_runner as runner

from awd_lstm_lm import data as salesforce_data
from awd_lstm_lm import model as salesforce_model
from awd_lstm_lm.splitcross import SplitCrossEntropyLoss


flags.DEFINE_integer('embedding_size', 400, '')
flags.DEFINE_integer('nhidden', 1000, '')
flags.DEFINE_integer('nlayers', 1, '')
flags.DEFINE_float('dropout', 0.0, '')
flags.DEFINE_float('dropouti', 0.0, '')
flags.DEFINE_float('dropoute', 0.0, '')
flags.DEFINE_float('dropouth', 0.0, '')
flags.DEFINE_float('wdrop', 0.0, '')
flags.DEFINE_boolean('tied', False, '')
flags.DEFINE_float('weight_decay', 0.0, '')
flags.DEFINE_float('l2_reg', 1e-6, '')
#flags.DEFINE_string('lr_drop_computes', '3662,18310,54931',
#                    'computes at which to drop LR. default corresponds to '
#                    '1,5,15 epochs.')

flags.DEFINE_integer('batch_size', 128, '')
flags.DEFINE_integer('eval_batch_size', 128, '')
flags.DEFINE_integer('test_batch_size', 1, '')

flags.DEFINE_integer('horizon_multiplier', 1, '')

flags.DEFINE_float('act_reg', 0.0, '')
flags.DEFINE_float('temp_act_reg', 0.0, '')

flags.DEFINE_boolean('test', False, 'use test set instead of valid')

flags.DEFINE_string('optimizer', 'sgd', 'sgd adam or mom')
flags.DEFINE_float('momentum', 0.9, 'momentum for SGD with mom')

flags.DEFINE_integer('seed', 0, 'Random seed for numpy, pytorch and random')

flags.DEFINE_boolean('use_cuda', True, 'use Cuda')

flags.DEFINE_boolean('fresh_hidden', False,
                     'create new hidden state for each batch')

flags.DEFINE_float('meta_lr', None, 'meta-optimization learning rate')
flags.DEFINE_float('exp_decay', 0.9, 'exp decay constant')

# Parameters to reproduce Le, et al
flags.DEFINE_float('beta1', 0.9, 'adam beta1')
flags.DEFINE_float('beta2', 0.999, 'adam beta2')
flags.DEFINE_float('adam_eps', 1e-8, 'adam eps')

flags.DEFINE_float('norm_clip', -1.0,
                   'clip grads to this norm before doing RT')
flags.DEFINE_float('post_clip', 1.0, 'clip before applying grads')

flags.DEFINE_integer('train_horizon', 5, 'truncated horizon of problem')
flags.DEFINE_integer('test_horizon', 5, 'full horizon of problem')
flags.DEFINE_integer('test_frequency', 25, 'test freq')
flags.DEFINE_integer('calibrate_frequency', 5, 'calibrate freq')

flags.DEFINE_boolean('compute_penalty', False, 'penalize RT due to multiple '
                     'computations required')

flags.DEFINE_integer('budget', 250000, 'multiple of test_horizon we run for')

flags.DEFINE_boolean('clip_intermediate', False,
                     'clip intermediate grads to '
                     'max norm of observed final grad')

FLAGS = flags.FLAGS


def assign_params(model, params):
    # pdb.set_trace()
    static_named_parameters = []
    for n_and_p in model.named_parameters():
        static_named_parameters.append(n_and_p)
    for name_and_param, new_param in zip(
            static_named_parameters, params):
        name, old_param = name_and_param
        if name == 'encoder.weight' and FLAGS.tied:
            setattr(model.decoder, 'weight', new_param)
        if name == 'decoder.weight' and FLAGS.tied:
            pdb.set_trace()
        module = model
        while len(name.split('.')) > 1:
            component_name = name.split('.')[0]
            module = getattr(module, component_name)
            name = '.'.join(name.split('.')[1:])
        setattr(module, name, new_param)


def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)


def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


def get_batch(source, i, seq_len, evaluation=False):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    if FLAGS.use_cuda:
        data = data.cuda(0)
        target = target.cuda(0)
    return data, target


class TrainData(object):
    def __init__(self, data, init_hidden):
        self.data = data
        self.i = 0
        self.init_hidden = init_hidden
        self.hidden = self.init_hidden(FLAGS.batch_size)
        self.stale_hidden = False

    def get_batch(self, seq_len):
        if seq_len + self.i >= len(self.data) - 1:
            self.i = 0
            self.hidden = self.init_hidden(FLAGS.batch_size)
        if FLAGS.fresh_hidden:
            self.hidden = self.init_hidden(FLAGS.batch_size)
        seq_len = min(seq_len, len(self.data) - 1 - self.i)
        inputs = self.data[self.i:self.i+seq_len]
        target = self.data[self.i+1:self.i+1+seq_len].view(-1)
        self.i += seq_len
        if FLAGS.use_cuda:
            inputs = inputs.cuda(0)
            target = target.cuda(0)
        return inputs, target


def make_corpus():
    fn = 'corpus.{}.data'.format(hashlib.md5(
        'awd_lstm_lm/data/enwik8'.encode()).hexdigest())
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset...')
        corpus = salesforce_data.Corpus('awd_lstm_lm/data/enwik8')
        torch.save(corpus, fn)

    train_data = batchify(corpus.train, FLAGS.batch_size)
    val_data = batchify(corpus.valid, FLAGS.eval_batch_size)
    test_data = batchify(corpus.test, FLAGS.test_batch_size)
    return corpus, train_data, val_data, test_data


def main(argv):
    if FLAGS.seed is not None:
        np.random.seed(FLAGS.seed)
        torch.manual_seed(FLAGS.seed)

    def _cuda(x):
        if FLAGS.use_cuda and torch.cuda.is_available():
            return x.cuda(0)
        elif FLAGS.use_cuda:
            raise Exception("Cuda is not available")
        else:
            return x

    if FLAGS.meta_lr is None:
        if FLAGS.optimizer == 'sgd':
            FLAGS.meta_lr = 2.2
        elif FLAGS.optimizer == 'mom':
            FLAGS.meta_lr = 1.0
        elif FLAGS.optimizer == 'adam':
            FLAGS.meta_lr = 2.2e-4

    corpus, train_data, val_data, test_data = make_corpus()

    eval_train_data = deepcopy(train_data)

    model = salesforce_model.RNNModel(
        'LSTM', len(corpus.dictionary), FLAGS.embedding_size,
        FLAGS.nhidden, FLAGS.nlayers, FLAGS.dropout, FLAGS.dropouth,
        FLAGS.dropouti, FLAGS.dropoute, FLAGS.wdrop, FLAGS.tied
    )

    model = _cuda(model)

    criterion = _cuda(SplitCrossEntropyLoss(FLAGS.embedding_size, splits=[],
                                            verbose=False))

    train_data = TrainData(train_data, model.init_hidden)

    def make_state_fn(horizon):
        batch = train_data.get_batch(FLAGS.horizon_multiplier*horizon)
        if train_data.stale_hidden:
            raise Exception("Hidden state is stale!")
        else:
            hidden = repackage_hidden(train_data.hidden)
        train_data.stale_hidden = True
        return batch, hidden

    def train_loss_fn(state, params, horizon):
        #pdb.set_trace()
        assign_params(model, params)
        #pdb.set_trace()

        model.train()

        batch, running_hidden = state
        hidden = repackage_hidden(running_hidden)
        full_data, targets = batch
        #pdb.set_trace()
        data = full_data[:FLAGS.horizon_multiplier*horizon]
        targets = targets[:FLAGS.horizon_multiplier*horizon*FLAGS.batch_size]

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden,
                                                       return_h=True)

        # pdb.set_trace()

        raw_loss = criterion(model.decoder.weight, model.decoder.bias,
                             output, targets)

        loss = raw_loss

        # Activiation Regularization
        if FLAGS.act_reg:
            loss = loss + sum(FLAGS.act_reg * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if FLAGS.temp_act_reg:
            loss = loss + sum(FLAGS.temp_act_reg * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])

        if FLAGS.l2_reg:
            loss = loss + FLAGS.l2_reg * sum(p.pow(2).sum() for p in model.parameters())
        if FLAGS.horizon_multiplier*horizon == len(full_data):
            train_data.hidden = repackage_hidden(hidden)
            train_data.stale_hidden = False
        compute = horizon
        return loss, compute

    def eval_fn(params, horizon, tflogger, step):
        del horizon
        # Turn on evaluation mode which disables dropout.
        #pdb.set_trace()
        assign_params(model, params)
        #pdb.set_trace()
        model.eval()
        total_loss = 0
        if FLAGS.test:
            data_source = test_data
            batch_size = FLAGS.test_batch_size
        else:
            data_source = val_data
            batch_size = FLAGS.eval_batch_size
        hidden = model.init_hidden(batch_size)
        # Ignore horizon
        seq_len = FLAGS.horizon_multiplier*(2**FLAGS.train_horizon+1)
        # pdb.set_trace()
        eval_len = data_source.size(0) - 1
        for i in range(0, eval_len, seq_len):
            # print('{} / {}'.format(i, data_source.size(0)-1))
            data, targets = get_batch(data_source, i, seq_len, evaluation=True)
            output, hidden = model(data, hidden)
            total_loss += len(data) * criterion(
                model.decoder.weight, model.decoder.bias, output, targets).data
            hidden = repackage_hidden(hidden)
        loss = total_loss.item() / len(data_source)

        data_source = eval_train_data
        batch_size = FLAGS.batch_size
        hidden = model.init_hidden(batch_size)
        total_loss = 0.
        data_len = data_source.size(0) - 1
        start_idx = np.random.randint(0, data_len-eval_len)

        for i in range(start_idx, start_idx+eval_len, seq_len):
            # print('{} / {}'.format(i, data_source.size(0)-1))
            data, targets = get_batch(data_source, i, seq_len, evaluation=True)
            output, hidden = model(data, hidden)
            total_loss += len(data) * criterion(
                model.decoder.weight, model.decoder.bias, output, targets).data
            hidden = repackage_hidden(hidden)
        train_loss = total_loss.item() / (eval_len+1)
        # pdb.set_trace()
        #pdb.set_trace()
        return {'test_bpc': loss/math.log(2),
                'test_entropy': loss,
                'train_bpc': train_loss/math.log(2),
                'train_entropy': train_loss}

    params = []
    for p in model.parameters():
        params.append(nn.Parameter(p.data.clone()))
    params = tuple(params)

    runner.run_experiment(
        params=params,
        train_loss_fn=train_loss_fn,
        make_state_fn=make_state_fn,
        eval_fn=eval_fn)

    print("Finished.")


if __name__ == '__main__':
    print("Starting.")
    tf.app.run()
