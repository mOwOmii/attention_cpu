"""Autograd semantics for replicated inputs and row-parallel partial outputs."""

import torch


class _CopyToTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, parallel):
        ctx.parallel = parallel
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return ctx.parallel.sum(gradient.contiguous().clone(), "tp"), None


class _ReduceFromTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, parallel):
        return parallel.sum(value.contiguous().clone(), "tp")

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def copy_to_tp(value, parallel):
    return _CopyToTP.apply(value, parallel)


def reduce_from_tp(value, parallel):
    return _ReduceFromTP.apply(value, parallel)
