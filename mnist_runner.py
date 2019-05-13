import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from tensorflow import flags
import pdb

import tensorflow as tf

import copy

import io
import math

import randomized_telescope_runner as runner

FLAGS = flags.FLAGS

# Almost always leave these fixed
flags.DEFINE_boolean('use_cuda', True, 'use Cuda')

flags.DEFINE_float('meta_lr', None, 'meta-optimization learning rate')
flags.DEFINE_float('exp_decay', 0.9, 'exp decay constant')

flags.DEFINE_float('beta1', 0.9, 'adam beta1')
flags.DEFINE_float('beta2', 0.999, 'adam beta2')
flags.DEFINE_float('adam_eps', 1e-8, 'adam eps')

flags.DEFINE_float('mnist_momentum', 0.9, 'momentum of learner on mnist')

flags.DEFINE_float('warm_start_lr', 0.1, 'warm start learning rate')

flags.DEFINE_integer('warm_start_steps', 50, 'warm start steps')

flags.DEFINE_string('optimizer', 'sgd', 'sgd adam or mom')
flags.DEFINE_float('momentum', 0.9, 'momentum for SGD')

flags.DEFINE_integer('batch_size', 100, 'batch size')

flags.DEFINE_float('init_lr', 0.01, 'init lr')
flags.DEFINE_float('init_decay', 0.1, 'init decay')

flags.DEFINE_float('norm_clip', -1.0, 'clip grads to this norm before doing RT')
flags.DEFINE_float('post_clip', 1.0, 'clip before applying grads')

flags.DEFINE_integer('train_horizon', 10, 'truncated horizon of problem')
flags.DEFINE_integer('test_horizon', 10, 'full horizon of problem')
flags.DEFINE_integer('test_frequency', 5, 'test freq')
flags.DEFINE_integer('calibrate_frequency', 5, 'calibrate freq')
flags.DEFINE_boolean('compute_penalty', False, 'penalize RT due to multiple '
                     'computations required')

flags.DEFINE_boolean('clip_intermediate', False,
                     'clip intermediate grads to '
                     'max norm of observed final grad')

flags.DEFINE_boolean('polyak', True, 'use polyak averaging in inner loop')

flags.DEFINE_float('decay_scale', 1.0,
                   'Scale in which to optimize LR decay parameter')

flags.DEFINE_integer('budget', 5000, 'multiple of test_horizon we run for')

flags.DEFINE_integer('seed', 0, 'Random seed for numpy, pytorch and random')

def copy_params(src_net, dest_net):
    for p1, p2 in zip(src_net.weights.values(), dest_net.weights.values()):
        p2.data.copy_(p1.data)

def assign_weights(net, weights):
    # pdb.set_trace()
    for name, new_weight in zip(net.weights.keys(), weights):
        net.weights[name] = new_weight

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main(argv):
    if FLAGS.seed is not None:
        np.random.seed(FLAGS.seed)
        torch.manual_seed(FLAGS.seed)

    if FLAGS.meta_lr is None:
        if FLAGS.optimizer == 'sgd':
            FLAGS.meta_lr = 1e-2
        elif FLAGS.optimizer == 'mom':
            FLAGS.meta_lr = 2.2e-3
        elif FLAGS.optimizer == 'adam':
            FLAGS.meta_lr = 5e-4

    if FLAGS.use_cuda:
        if torch.cuda.is_available():
            CUDA = True
        else:
            raise Exception("Cuda is not available, run without --cuda")
    else:
        CUDA = False

    def _cuda(x):
        if CUDA:
            return x.cuda()
        else:
            return x

    kwargs = {'num_workers': 1, 'pin_memory': True}# if FLAGS.use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=FLAGS.batch_size, shuffle=True, **kwargs)

    eval_train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=FLAGS.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=FLAGS.batch_size, shuffle=True, **kwargs)

    eval_test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=FLAGS.batch_size, shuffle=False, **kwargs)


    class Net(object):
        def __init__(self):
            self.weights = {
                'w1': Variable(_cuda(0.1*torch.randn(784, 100)), requires_grad=True),
                'b1': Variable(_cuda(torch.zeros(1, 100)), requires_grad=True),
                'w2': Variable(_cuda(0.1*torch.randn(100, 100)), requires_grad=True),
                'b2': Variable(_cuda(torch.zeros(1, 100)), requires_grad=True),
                'w3': Variable(_cuda(0.1*torch.randn(100, 10)), requires_grad=True),
                'b3': Variable(_cuda(torch.zeros(1, 10)), requires_grad=True)
            }

        def forward(self, x):
            x = torch.matmul(x, self.weights['w1']) + self.weights['b1']
            x = F.relu(x)
            x = torch.matmul(x, self.weights['w2']) + self.weights['b2']
            x = F.relu(x)
            x = torch.matmul(x, self.weights['w3']) + self.weights['b3']
            return F.log_softmax(x, dim=1)

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    base_net = Net()

    def train_net(params, loader, net, timesteps, meta_train=False):
        log_lr, log_decay = params
        log_decay = log_decay * FLAGS.decay_scale
        lr0 = torch.exp(log_lr)
        decay = torch.exp(log_decay)
        #optimizer = optim.SGD(net.parameters(), lr=FLAGS.init_lr,
        #                      momentum=0.9)
        #optimizer.zero_grad()
        #pdb.set_trace()
        weights = []
        velocities = []
        for w in net.weights.values():
            weights.append(w)
            velocities.append(torch.zeros_like(w))
        if FLAGS.polyak:
            test_weights = []
            for w in net.weights.values():
                test_weights.append(torch.zeros_like(w))

        steps = 0
        while steps <= timesteps:
            for batch_idx, (data, target) in enumerate(loader):
                if steps > timesteps:
                    break
                data, target = _cuda(data), _cuda(target)
                data = data.view([FLAGS.batch_size, -1])
                output = net(data)
                loss = F.nll_loss(output, target)
                #pdb.set_trace()
                grads = torch.autograd.grad(loss, weights,
                                            create_graph=meta_train,
                                            retain_graph=meta_train)
                lr = lr0 * (1 / (1 + steps/5000)**decay)
                # pdb.set_trace()
                for i in range(len(weights)):
                    velocities[i] = (
                        FLAGS.mnist_momentum * velocities[i] -
                        lr * grads[i]
                    )
                    weights[i] = (
                        weights[i] + velocities[i]
                    )
                    if FLAGS.polyak:
                        test_weights[i] = (
                            test_weights[i] + weights[i] / timesteps
                        )

                assign_weights(net, weights)
                # optimizer.zero_grad()
                steps += 1
        if FLAGS.polyak:
            assign_weights(net, test_weights)

    def test_net(loader, net, timesteps, predict=False, eval_all=False):
        test_loss = 0
        correct = 0
        # pdb.set_trace()
        for batch_idx, (data, target) in enumerate(loader):
            if not eval_all:
                if batch_idx > timesteps:
                    break
            data, target = _cuda(data), _cuda(target)
            data = data.view([FLAGS.batch_size, -1])
            output = net(data)
            test_loss += F.nll_loss(output, target) # sum up batch loss
            #pdb.set_trace()
            if predict:
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).float().mean()

        '''
        if eval_all:
            N = sum([len(b) for b in batches])
        else:
            N = min(sum([len(b) for b in batches]), timesteps*FLAGS.batch_size)
        '''
        if predict:
            return test_loss/(batch_idx+1), float(correct.item())/(batch_idx+1)
        else:
            return test_loss/(batch_idx+1)

    pretrain_net = Net()
    train_net(
        (_cuda(torch.Tensor([np.log(FLAGS.warm_start_lr)])),
         _cuda(torch.Tensor([-np.inf]))),
        train_loader,
        pretrain_net, FLAGS.warm_start_steps)

    copy_params(pretrain_net, base_net)

    def train_fn(state, params, timesteps):
        net = Net()
        copy_params(base_net, net)

        train_net(params, train_loader, net, timesteps, meta_train=True)

        avg_loss = test_net(test_loader, net, timesteps)
        compute = timesteps
        return avg_loss, compute

    def eval_fn(params, timesteps, tflogger, step):
        this_train_loader = copy.deepcopy(eval_train_loader)# copy.deepcopy(base_train_test_loader)
        #test_loader = base_test_loader# copy.deepcopy(base_test_loader)

        # pdb.set_trace()

        net = Net()
        copy_params(base_net, net)
        train_net(params, train_loader, net, timesteps)

        tflogger.log_scalar('LR',
                            np.exp(params[0].item()), step)
        tflogger.log_scalar('Decay',
                            np.exp(params[1].item()*FLAGS.decay_scale), step)

        with torch.no_grad():
            #pdb.set_trace()
            train_loss, train_accuracy = test_net(this_train_loader, net, timesteps,
                                                  predict=True, eval_all=True)
            eval_loss, eval_accuracy = test_net(eval_test_loader, net, timesteps,
                                                predict=True, eval_all=True)
        return {
            'fullhorizon_train_cross_entropy': train_loss.item(),
            'fullhorizon_train_accuracy': train_accuracy,
            'fullhorizon_eval_cross_entropy': eval_loss.item(),
            'fullhorizon_eval_accuracy': eval_accuracy}

    log_lr = nn.Parameter(_cuda(torch.log(torch.Tensor([FLAGS.init_lr] * 1))))
    log_decay = nn.Parameter(_cuda(
        torch.log(torch.Tensor([FLAGS.init_decay] * 1))/FLAGS.decay_scale
        ))

    params = (log_lr, log_decay)

    runner.run_experiment(
        params=params,
        train_loss_fn=train_fn,
        make_state_fn=lambda horizon: None,
        eval_fn=eval_fn)


if __name__ == '__main__':
    tf.app.run()
