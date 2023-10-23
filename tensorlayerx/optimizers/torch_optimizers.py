#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import torch.optim as optimizer
from torch.optim import _functional as F
import torch
from tensorlayerx.optimizers.lr import LRScheduler
import warnings
from typing import cast

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
        self.defaults = dict(
            lr=lr,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
            maximize=False,
            foreach=None,
            differentiable=False,
        )

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
            lr, rho, eps, weight_decay, foreach, maximize, differentiable = (
                get_lr(self.lr),
                group['rho'],
                group['eps'],
                group['weight_decay'],
                group["foreach"],
                group["maximize"],
                group["differentiable"],
            )

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

            F.adadelta(
                    params_with_grad,
                    grads,
                    square_avgs,
                    acc_deltas,
                    lr=lr,
                    rho=rho,
                    eps=eps,
                    weight_decay=weight_decay,
                    foreach=foreach,
                    maximize=maximize,
                    differentiable=differentiable,
            )

        return loss

    def gradient(self, loss, weights=None, return_grad=True):
        if weights is None:
            raise AttributeError("Parameter train_weights must be entered.")
        if not self.init_optim:
            self.optimizer_adadelta = optimizer.Adadelta(
                params=weights, lr=get_lr(self.lr), rho=self.rho, eps=self.eps, weight_decay=self.weight_decay
            )
            for param_group in self.optimizer_adadelta.param_groups:
                add_param_group(cast(dict, param_group), self.defaults, self.optimizer_adadelta)
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
        self.defaults = dict(
            lr=lr,
            lr_decay=0,
            eps=eps,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            foreach=None,
            maximize=False,
            differentiable=False,
        )

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
                    state_steps.append(state['step'])

            F.adagrad(
                params_with_grad,
                grads,
                state_sums,
                state_steps,
                lr=get_lr(self.lr),
                weight_decay=group['weight_decay'],
                lr_decay=group['lr_decay'],
                eps=group['eps'],
                foreach=group["foreach"],
                maximize=group["maximize"],
                differentiable=group["differentiable"]
            )

        return loss

    def gradient(self, loss, weights=None, return_grad=True):
        if weights is None:
            raise AttributeError("Parameter train_weights must be entered.")
        if not self.init_optim:
            self.optimizer_adagrad = optimizer.Adagrad(
                params=weights, lr=get_lr(self.lr), lr_decay=self.initial_accumulator_value,
                weight_decay=self.weight_decay
            )
            for param_group in self.optimizer_adagrad.param_groups:
                add_param_group(cast(dict, param_group), self.defaults, self.optimizer_adagrad)
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
        self.defaults = dict(lr=lr, beta_1=beta_1, beta_2=beta_2, eps=eps,
                        weight_decay=weight_decay, amsgrad=False,
                        maximize=False, foreach=None, capturable=False,
                        differentiable=False, fused=None)

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
                    # Lazy state initialization
                if len(state) == 0:
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable and fused are off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    state['step'] = (
                        torch.zeros((), dtype=torch.float, device=p.device)
                        if group['capturable'] or group['fused']
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
                if group['differentiable'] and state['step'].requires_grad:
                    raise RuntimeError('`requires_grad` is not supported for `step` in differentiable mode')

                # Foreach without capturable does not support a tensor lr
                if group['foreach'] and torch.is_tensor(group['lr']) and not group['capturable']:
                    raise RuntimeError('lr as a Tensor is not supported for capturable=False and foreach=True')

                state_steps.append(state['step'])

            F.adam(params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad=group['amsgrad'],
                    beta1=beta1,
                    beta2=beta2,
                    lr=get_lr(self.lr),
                    weight_decay=group['weight_decay'],
                    eps=group['eps'],
                    maximize=group['maximize'],
                    foreach=group['foreach'],
                    capturable=group['capturable'],
                    differentiable=group['differentiable'],
                    fused=group['fused']
                )
        return loss

    def gradient(self, loss, weights=None, return_grad=True):
        if weights is None:
            raise AttributeError("Parameter train_weights must be entered.")
        if not self.init_optim:
            self.optimizer_adam = optimizer.Adam(
                params=weights, lr=get_lr(self.lr), betas=(self.beta_1, self.beta_2), eps=self.eps,
                weight_decay=self.weight_decay
            )
            for param_group in self.optimizer_adam.param_groups:
                add_param_group(cast(dict, param_group), self.defaults, self.optimizer_adam)
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
        self.defaults = dict(
            lr=lr,
            beta_1=beta_1,
            beta_2=beta_2,
            eps=eps,
            weight_decay=weight_decay,
            foreach=None,
            maximize=False,
            differentiable=False,
        )

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

            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = get_lr(self.lr)
            weight_decay = group['weight_decay']
            foreach = group["foreach"]
            maximize = group["maximize"]
            differentiable = group["differentiable"]

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('Adamax does not support sparse gradients')
                grads.append(p.grad)

                state = self.optimizer_adamax.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = torch.tensor(0.0)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_inf'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_infs.append(state['exp_inf'])
                state_steps.append(state['step'])

            F.adamax(
                params_with_grad,
                grads,
                exp_avgs,
                exp_infs,
                state_steps,
                eps=eps,
                beta1=beta1,
                beta2=beta2,
                lr=lr,
                weight_decay=weight_decay,
                foreach=foreach,
                maximize=maximize,
                differentiable=differentiable
            )

        return loss

    def gradient(self, loss, weights=None, return_grad=True):
        if weights is None:
            raise AttributeError("Parameter train_weights must be entered.")
        if not self.init_optim:
            self.optimizer_adamax = optimizer.Adamax(
                params=weights, lr=get_lr(self.lr), betas=(self.beta_1, self.beta_2), eps=self.eps,
                weight_decay=self.weight_decay
            )
            for param_group in self.optimizer_adamax.param_groups:
                add_param_group(cast(dict, param_group), self.defaults, self.optimizer_adamax)
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
        self.defaults = dict(
            lr=lr,
            momentum=momentum,
            alpha=rho,
            eps=eps,
            centered=centered,
            weight_decay=weight_decay,
            foreach=None,
            maximize=False,
            differentiable=False,
        )
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

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)

                if p.grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                grads.append(p.grad)

                state = self.optimizer_rmsprop.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avgs.append(state['square_avg'])

                if group['momentum'] > 0:
                    momentum_buffer_list.append(state['momentum_buffer'])
                if group['centered']:
                    grad_avgs.append(state['grad_avg'])

                state['step'] += 1

            F.rmsprop(params_with_grad,
                      grads,
                      square_avgs,
                      grad_avgs,
                      momentum_buffer_list,
                      lr=get_lr(self.lr),
                      alpha=group['alpha'],
                      eps=group['eps'],
                      weight_decay=group['weight_decay'],
                      momentum=group['momentum'],
                      centered=group['centered'])

        return loss

    def gradient(self, loss, weights=None, return_grad=True):
        if weights is None:
            raise AttributeError("Parameter train_weights must be entered.")
        if not self.init_optim:
            self.optimizer_rmsprop = optimizer.RMSprop(
                params=weights, lr=get_lr(self.lr), alpha=self.rho, eps=self.eps, momentum=self.momentum,
                centered=self.centered, weight_decay=self.weight_decay
            )
            for param_group in self.optimizer_rmsprop.param_groups:
                add_param_group(cast(dict, param_group), self.defaults, self.optimizer_rmsprop)
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
        self.defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=0,
            weight_decay=weight_decay,
            nesterov=False,
            maximize=False,
            foreach=None,
            differentiable=False
        )

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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            maximize = group['maximize']
            foreach = group['foreach']
            lr = get_lr(self.lr)

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.optimizer_sgd.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            F.sgd(
                params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=nesterov,
                maximize= maximize,
                foreach=foreach
            )

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
            for param_group in self.optimizer_sgd.param_groups:
                add_param_group(cast(dict, param_group), self.defaults, self.optimizer_sgd)
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
        self.defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=0,
            weight_decay=weight_decay,
            nesterov=False,
            maximize=False,
            foreach=None,
            differentiable=False
        )

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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            maximize = group['maximize']
            foreach = group['foreach']
            lr = get_lr(self.lr)

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.optimizer_momentum.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            F.sgd(
                params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=nesterov,
                maximize= maximize,
                foreach=foreach
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.optimizer_momentum.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

    def gradient(self, loss, weights=None, return_grad=True):
        if weights is None:
            raise AttributeError("Parameter train_weights must be entered.")
        if not self.init_optim:
            self.optimizer_momentum = optimizer.SGD(
                params=weights, lr=get_lr(self.lr), momentum=self.momentum, weight_decay=self.weight_decay, nesterov=self.nesterov
            )
            for param_group in self.optimizer_momentum.param_groups:
                add_param_group(cast(dict, param_group), self.defaults, self.optimizer_momentum)
            self.init_optim = True
        self.optimizer_momentum.zero_grad()
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

def add_param_group(param_group, defaults, optimizer):
    r"""Add a param group to the :class:`Optimizer` s `param_groups`.

    This can be useful when fine tuning a pre-trained network as frozen layers can be made
    trainable and added to the :class:`Optimizer` as training progresses.

    Args:
        param_group (dict): Specifies what Tensors should be optimized along with group
            specific optimization options.
    """
    if not isinstance(param_group, dict):
        raise TypeError(f"param_group must be a dict, but got {type(param_group)}")

    params = param_group['params']
    if isinstance(params, torch.Tensor):
        param_group['params'] = [params]
    elif isinstance(params, set):
        raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                        'the ordering of tensors in sets will change between runs. Please use a list instead.')
    else:
        param_group['params'] = list(params)

    for param in param_group['params']:
        if not isinstance(param, torch.Tensor):
            raise TypeError("optimizer can only optimize Tensors, "
                            "but one of the params is " + torch.typename(param))
        if not defaults.get('differentiable', None) and not (param.is_leaf or param.retains_grad):
            raise ValueError("can't optimize a non-leaf Tensor")

    for name, default in defaults.items():
        param_group.setdefault(name, default)

    params = param_group['params']
    if len(params) != len(set(params)):
        warnings.warn("optimizer contains a parameter group with duplicate parameters; "
                        "in future, this will cause an error; "
                        "see github.com/pytorch/pytorch/issues/40967 for more information", stacklevel=3)

    param_set = set()
    for group in optimizer.param_groups:
        param_set.update(set(group['params']))

    if not param_set.isdisjoint(set(param_group['params'])):
        raise ValueError("some parameters appear in more than one parameter group")

    optimizer.param_groups.append(param_group)