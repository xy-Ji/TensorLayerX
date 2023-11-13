#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import torch
import warnings
import inspect
import torch.optim as optimizer
from torch.optim import _functional as F
from typing import cast
from tensorlayerx.optimizers.lr import LRScheduler

__all__ = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD', 'Momentum', 'Lamb', 'LARS']


class Adadelta(object):

    def __init__(
        self,
        lr=0.001,
        rho=0.95,
        eps=1e-10,
        weight_decay=0.0,
        grad_clip=None,
    ):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.init_optim = False
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    @torch.no_grad()
    def apply_gradients(self, grads_and_vars=None, closure=None):
        if not self.init_optim:
            raise AttributeError("Can not apply gradients before zero_grad call.")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.optimizer_adadelta.param_groups:
            params_with_grad = []
            grads = []
            square_avgs = []
            acc_deltas = []

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('Adadelta does not support sparse gradients')
                grads.append(p.grad)

                state = self.optimizer_adadelta.state[p]

                # Lazy state initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['acc_delta'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avgs.append(state['square_avg'])
                acc_deltas.append(state['acc_delta'])

                state['step'] += 1

            adadelta_signature = inspect.signature(F.adadelta)
            params = adadelta_signature.parameters

            adadelta_args = {
                'params': params_with_grad,
                'grads': grads,
                'square_avgs': square_avgs,
                'acc_deltas': acc_deltas,
                'lr': get_lr(self.lr),
                'rho': group['rho'],
                'eps': group['eps'],
                'weight_decay': group['weight_decay']
            }
            for param in ['maximize', 'foreach', 'differentiable']:
                if param in params:
                    adadelta_args[param] = group[param]

            F.adadelta(**adadelta_args)

        return loss

    def gradient(self, loss, weights=None, return_grad=True):
        if weights is None:
            raise AttributeError("Parameter train_weights must be entered.")
        if not self.init_optim:
            self.optimizer_adadelta = optimizer.Adadelta(
                params=weights, lr=get_lr(self.lr), rho=self.rho, eps=self.eps, weight_decay=self.weight_decay
            )
            self.init_optim = True
        self.optimizer_adadelta.zero_grad()
        loss.backward()

        if self.grad_clip is not None:
            self.grad_clip(weights)

        if return_grad ==True:
            return _grads(weights)
        else:
            return None


class Adagrad(object):

    def __init__(
        self,
        lr=0.001,
        initial_accumulator_value=0.1,
        eps=1e-10,
        weight_decay=0.0,
        grad_clip=None,
    ):
        self.lr = lr
        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps
        self.init_optim = False
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def _init_group(self, group, params_with_grad, grads, state_sums, state_steps):
        has_sparse_grad = False
        for p in group["params"]:
            if p.grad is not None:
                if p.grad.is_sparse:
                    has_sparse_grad = True
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]
                state_sums.append(state["sum"])
                state_steps.append(state["step"])

        return has_sparse_grad

    @torch.no_grad()
    def apply_gradients(self, grads_and_vars=None, closure=None):
        if not self.init_optim:
            raise AttributeError("Can not apply gradients before zero_grad call.")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.optimizer_adagrad.param_groups:
            params_with_grad = []
            grads = []
            state_sums = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.optimizer_adagrad.state[p]
                    state_sums.append(state['sum'])
                    state['step'] += 1
                    state_steps.append(state['step'])

            adagrad_signature = inspect.signature(F.adagrad)
            params = adagrad_signature.parameters

            adagrad_args = {
                'params': params_with_grad,
                'grads': grads,
                'state_sums': state_sums,
                'state_steps': state_steps,
                'lr': get_lr(self.lr),
                'weight_decay': group['weight_decay'],
                'eps': group['eps']
            }
            for param in ['maximize', 'foreach', 'differentiable']:
                if param in params:
                    adagrad_args[param] = group[param]

            has_sparse_grad = self._init_group(group, params_with_grad, grads, state_sums, state_steps)
            adagrad_args['has_sparse_grad'] = has_sparse_grad
            F.adagrad(**adagrad_args)

        return loss

    def gradient(self, loss, weights=None, return_grad=True):
        if weights is None:
            raise AttributeError("Parameter train_weights must be entered.")
        if not self.init_optim:
            self.optimizer_adagrad = optimizer.Adagrad(
                params=weights, lr=get_lr(self.lr), lr_decay=self.initial_accumulator_value,
                weight_decay=self.weight_decay
            )
            self.init_optim = True
        self.optimizer_adagrad.zero_grad()
        loss.backward()

        if self.grad_clip is not None:
            self.grad_clip(weights)

        if return_grad ==True:
            return _grads(weights)
        else:
            return None


class Adam(object):

    def __init__(
        self,
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        grad_clip=None,
    ):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.init_optim = False
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    @torch.no_grad()
    def apply_gradients(self, grads_and_vars=None, closure=None):
        if not self.init_optim:
            raise AttributeError("Can not apply gradients before zero_grad call.")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.optimizer_adam.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.optimizer_adam.state[p]

                    capturable = group.get('capturable', False)
                    differentiable = group.get('differentiable', False)
                    foreach = group.get('foreach', None)
                    # Lazy state initialization
                    if len(state) == 0:
                        # note(crcrpar): [special device hosting for step]
                        # Deliberately host `step` on CPU if both capturable and fused are off.
                        # This is because kernel launches are costly on CUDA and XLA.
                        fused = group.get('fused', None)
                        state['step'] = (
                            torch.zeros((), dtype=torch.float, device=p.device)
                            if capturable or fused
                            else torch.tensor(0.)
                        )
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                    if differentiable and state['step'].requires_grad:
                        raise RuntimeError('`requires_grad` is not supported for `step` in differentiable mode')

                    # Foreach without capturable does not support a tensor lr
                    if foreach and torch.is_tensor(group['lr']) and not group['capturable']:
                        raise RuntimeError('lr as a Tensor is not supported for capturable=False and foreach=True')
                    state['step'] += 1
                    state_steps.append(state['step'])

            adam_signature = inspect.signature(F.adam)
            params = adam_signature.parameters

            adam_args = {
                'params': params_with_grad,
                'grads': grads,
                'exp_avgs': exp_avgs,
                'exp_avg_sqs': exp_avg_sqs,
                'max_exp_avg_sqs': max_exp_avg_sqs,
                'state_steps': state_steps,
                'amsgrad': group['amsgrad'],
                'beta1': beta1,
                'beta2': beta2,
                'lr': get_lr(self.lr),
                'weight_decay': group['weight_decay'],
                'eps': group['eps']
            }
            for param in ['maximize', 'foreach', 'capturable', 'differentiable', 'fused']:
                if param in params:
                    adam_args[param] = group[param]
            F.adam(**adam_args)

        return loss

    def gradient(self, loss, weights=None, return_grad=True):
        if weights is None:
            raise AttributeError("Parameter train_weights must be entered.")
        if not self.init_optim:
            self.optimizer_adam = optimizer.Adam(
                params=weights, lr=get_lr(self.lr), betas=(self.beta_1, self.beta_2), eps=self.eps,
                weight_decay=self.weight_decay
            )
            self.init_optim = True
        self.optimizer_adam.zero_grad()
        loss.backward()

        if self.grad_clip is not None:
            self.grad_clip(weights)

        if return_grad ==True:
            return _grads(weights)
        else:
            return None


class Adamax(object):

    def __init__(
        self,
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        grad_clip=None,
    ):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.init_optim = False
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    @torch.no_grad()
    def apply_gradients(self, grads_and_vars=None, closure=None):
        if not self.init_optim:
            raise AttributeError("Can not apply gradients before zero_grad call.")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.optimizer_adamax.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_infs = []
            state_steps = []

            self._init_group(group, params_with_grad, grads, exp_avgs, exp_infs, state_steps)

            adamax_signature = inspect.signature(F.adamax)
            params = adamax_signature.parameters

            adamax_args = {
                'params': params_with_grad,
                'grads': grads,
                'exp_avgs': exp_avgs,
                'exp_infs': exp_infs,
                'state_steps': state_steps,
                'eps': group['eps'],
                'beta1': group['betas'][0],
                'beta2': group['betas'][1],
                'lr': get_lr(self.lr),
                'weight_decay': group['weight_decay']
            }
            for param in ['maximize', 'foreach', 'differentiable']:
                if param in params:
                    adamax_args[param] = group[param]
            F.adam(**adamax_args)

        return loss

    def _init_group(self, group, params_with_grad, grads, exp_avgs, exp_infs, state_steps):
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("Adamax does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state["step"] = torch.tensor(0.0)
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state["exp_inf"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

            exp_avgs.append(state["exp_avg"])
            exp_infs.append(state["exp_inf"])
            state['step'] += 1
            state_steps.append(state["step"])

    def gradient(self, loss, weights=None, return_grad=True):
        if weights is None:
            raise AttributeError("Parameter train_weights must be entered.")
        if not self.init_optim:
            self.optimizer_adamax = optimizer.Adamax(
                params=weights, lr=get_lr(self.lr), betas=(self.beta_1, self.beta_2), eps=self.eps,
                weight_decay=self.weight_decay
            )
            self.init_optim = True
        self.optimizer_adamax.zero_grad()
        loss.backward()

        if self.grad_clip is not None:
            self.grad_clip(weights)

        if return_grad ==True:
            return _grads(weights)
        else:
            return None


class Ftrl(object):

    def __init__(self):
        raise NotImplementedError("Ftrl optimizer is not implemented")

    def apply_gradients(self):
        pass

    def gradient(self, train_weights=None):
        pass


class Nadam(object):

    def __init__(self):
        raise NotImplementedError("Nadam optimizer is not implemented")

    def apply_gradients(self):
        pass

    def gradient(self, train_weights=None):
        pass


class RMSprop(object):

    def __init__(
        self,
        lr=0.001,
        rho=0.99,
        momentum=0.0,
        eps=1e-08,
        centered=False,
        weight_decay=0.0,
        grad_clip=None,
    ):
        self.lr = lr
        self.alpha = rho
        self.momentum = momentum
        self.eps = eps
        self.centered = centered
        self.init_optim = False
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def _init_group(self, group, params_with_grad, grads, square_avgs, momentum_buffer_list, grad_avgs):
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)

            if p.grad.is_sparse:
                raise RuntimeError("RMSprop does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]

            differentiable = group.get('differentiable', False)
            # State initialization
            if len(state) == 0:
                state["step"] = 0
                state["square_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                if group["momentum"] > 0:
                    state["momentum_buffer"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                if group["centered"]:
                    state["grad_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
            square_avgs.append(state["square_avg"])

            if group["momentum"] > 0:
                momentum_buffer_list.append(state["momentum_buffer"])
            if group["centered"]:
                grad_avgs.append(state["grad_avg"])

            if differentiable and isinstance(state["step"], torch.Tensor):
                raise RuntimeError("`step` can't be a tensor")

            state["step"] += 1

    @torch.no_grad()
    def apply_gradients(self, grads_and_vars=None, closure=None):
        if not self.init_optim:
            raise AttributeError("Can not apply gradients before zero_grad call.")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.optimizer_rmsprop.param_groups:
            params_with_grad = []
            grads = []
            square_avgs = []
            grad_avgs = []
            momentum_buffer_list = []

            self._init_group(group, params_with_grad, grads, square_avgs, momentum_buffer_list, grad_avgs)

            rmsprop_signature = inspect.signature(F.rmsprop)
            params = rmsprop_signature.parameters

            rmsprop_args = {
                'params': params_with_grad,
                'grads': grads,
                'exp_avgs': square_avgs,
                'grad_avgs': grad_avgs,
                'momentum_buffer_list': momentum_buffer_list,
                'lr': get_lr(self.lr),
                'alpha': group["alpha"],
                'eps': group['eps'],
                'weight_decay': group['weight_decay'],
                'momentum': group['momentum'],
                'centered': group['centered']
            }
            for param in ['maximize', 'foreach', 'differentiable']:
                if param in params:
                    rmsprop_args[param] = group[param]
            F.rmsprop(**rmsprop_args)

        return loss

    def gradient(self, loss, weights=None, return_grad=True):
        if weights is None:
            raise AttributeError("Parameter train_weights must be entered.")
        if not self.init_optim:
            self.optimizer_rmsprop = optimizer.RMSprop(
                params=weights, lr=get_lr(self.lr), alpha=self.rho, eps=self.eps, momentum=self.momentum,
                centered=self.centered, weight_decay=self.weight_decay
            )
            self.init_optim = True
        self.optimizer_rmsprop.zero_grad()
        loss.backward()

        if self.grad_clip is not None:
            self.grad_clip(weights)

        if return_grad ==True:
            return _grads(weights)
        else:
            return None


class SGD(object):

    def __init__(
        self,
        lr=0.001,
        momentum=0,
        weight_decay=0.0,
        grad_clip=None,
    ):
        self.lr = lr
        self.momentum = momentum
        self.init_optim = False
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def _init_group(self, group, params_with_grad, d_p_list, momentum_buffer_list):
        has_sparse_grad = False

        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])

        return has_sparse_grad

    @torch.no_grad()
    def apply_gradients(self, grads_and_vars=None, closure=None):
        if not self.init_optim:
            raise AttributeError("Can not apply gradients before zero_grad call.")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.optimizer_sgd.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            has_sparse_grad = self._init_group(group, params_with_grad, d_p_list, momentum_buffer_list)
            
            sgd_signature = inspect.signature(F.sgd)
            params = sgd_signature.parameters

            sgd_args = {
                'params': params_with_grad,
                'd_p_list': d_p_list,
                'momentum_buffer_list': momentum_buffer_list,
                'weight_decay': group['weight_decay'],
                'momentum': group['momentum'],
                'lr': get_lr(self.lr),
                'dampening': group['dampening'],
                'nesterov': group['nesterov'],
                'has_sparse_grad': has_sparse_grad
            }
            for param in ['maximize', 'foreach', 'differentiable']:
                if param in params:
                    sgd_args[param] = group[param]
            F.sgd(**sgd_args)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.optimizer_sgd.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

    def gradient(self, loss, weights=None, return_grad=True):
        if weights is None:
            raise AttributeError("Parameter train_weights must be entered.")
        if not self.init_optim:
            self.optimizer_sgd = optimizer.SGD(
                params=weights, lr=get_lr(self.lr), momentum=self.momentum, weight_decay=self.weight_decay
            )
            self.init_optim = True
        self.optimizer_sgd.zero_grad()
        loss.backward()

        if self.grad_clip is not None:
            self.grad_clip(weights)

        if return_grad ==True:
            return _grads(weights)
        else:
            return None


class Momentum(object):

    def __init__(
        self,
        lr=0.001,
        momentum=0,
        weight_decay=0.0,
        nesterov=False,
        grad_clip=None,
    ):
        self.lr = lr
        self.momentum = momentum
        self.init_optim = False
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.grad_clip = grad_clip

    def _init_group(self, group, params_with_grad, d_p_list, momentum_buffer_list):
        has_sparse_grad = False

        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])

        return has_sparse_grad

    @torch.no_grad()
    def apply_gradients(self, grads_and_vars=None, closure=None):
        if not self.init_optim:
            raise AttributeError("Can not apply gradients before zero_grad call.")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.optimizer_momentum.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            has_sparse_grad = self._init_group(group, params_with_grad, d_p_list, momentum_buffer_list)
            
            momentum_signature = inspect.signature(F.sgd)
            params = momentum_signature.parameters

            momentum_args = {
                'params': params_with_grad,
                'd_p_list': d_p_list,
                'momentum_buffer_list': momentum_buffer_list,
                'weight_decay': group['weight_decay'],
                'momentum': group['momentum'],
                'lr': get_lr(self.lr),
                'dampening': group['dampening'],
                'nesterov': group['nesterov'],
                'has_sparse_grad': has_sparse_grad
            }
            for param in ['maximize', 'foreach', 'differentiable']:
                if param in params:
                    momentum_args[param] = group[param]
            F.sgd(**momentum_args)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.optimizer_sgd.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

    def gradient(self, loss, weights=None, return_grad=True):
        if weights is None:
            raise AttributeError("Parameter train_weights must be entered.")
        if not self.init_optim:
            self.optimizer_sgd = optimizer.SGD(
                params=weights, lr=get_lr(self.lr), momentum=self.momentum, weight_decay=self.weight_decay
            )
            self.init_optim = True
        self.optimizer_sgd.zero_grad()
        loss.backward()

        if self.grad_clip is not None:
            self.grad_clip(weights)

        if return_grad ==True:
            return _grads(weights)
        else:
            return None


def Lamb(**kwargs):
    raise Exception('Lamb optimizer function not implemented')


def LARS(**kwargs):
    raise Exception('LARS optimizer function not implemented')


def _grads(weights):
    grads = []
    for w in weights:
        grads.append(w.grad)
    return grads

def get_lr(lr):
    if isinstance(lr, LRScheduler):
        return lr()
    return lr

# def update_param_group(index, defaults, optimizer):

#     param_group = optimizer.param_groups[index]
#     if not set(defaults.keys())==(set(param_group.keys())):
#         for name, default in defaults.items():
#             cast(dict, param_group).setdefault(name, default)

#     optimizer.param_groups[index] = param_group