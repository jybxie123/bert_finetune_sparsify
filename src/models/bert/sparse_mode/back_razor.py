import sys
sys.path.insert(0, '/disk3/Haonan/yanbo_random/bert_finetune_sparsify/src/models/bert')
from sparse_mode import rand_layers as rl
import torch
import torch.nn.functional as F
import torch.nn as nn
# ================== back razor ==================
# class LinearSparse(nn.Linear, custom_quant.Quant):
class BackRazorLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, keep_frac=0.5, linear_idx = None, act_type = None):
        super(BackRazorLinear, self).__init__(in_features, out_features, bias=bias)
        self.keep_frac = keep_frac
        self.tag = 'fc'
        self.step_idx = 0
        self.linear_idx = linear_idx
        self.act_type = act_type

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        # print('=================backrazor sparse linear forward=================')
        y = backRazorLinear.apply(x, self.weight, self.bias, self.keep_frac)
        # cal_zero_ratio(result, self.linear_idx, self.step_idx, self.act_type)
        return y

class backRazorLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, keep_frac = 0.5):
        masker = rl.Masker(prune_ratio=1-keep_frac)
        mask = masker(x)
        shape_x, mask_x, sparse_x = rl.sparsify(x, mask, with_batch_size=False)
        # if half and (not quantize):
        #     sparse_x = sparse_x.half()

        ctx.save_for_backward(weight, bias, shape_x, mask_x, sparse_x)

        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_weight = grad_bias = None

        tensors = ctx.saved_tensors

        # if len(tensors) == 5:
        weight, bias, shape_x, mask_x, sparse_x = tensors
        # else:
        #     weight, bias, shape_x, mask_x = tensors
        #     sparse_x = custom_quant.Quant.restore(ctx)

        sparse_x = sparse_x.float()
        input = rl.unsparsify(shape_x, mask_x, sparse_x, with_batch_size=False)

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight.to(dtype=grad_output.dtype))

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(-2, -1).matmul(input.to(dtype=grad_output.dtype))

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None

class BackRazorMatMul(nn.Module):
    def __init__(self, args=None, keep_frac=0.5, linear_idx = None, act_type = None, sparse_mode = 'bcrz'):
        super(BackRazorMatMul, self).__init__()
        self.keep_frac = keep_frac
        self.sparse_mode = sparse_mode
        self.step_idx = 0
        self.linear_idx = linear_idx
        self.act_type = act_type

    def forward(self, x1, x2):
        # print('Our sparse matmul')
        y = backRazorMatMul.apply(x1, x2, self.keep_frac)
        return y

import time

# new one
class backRazorMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, to_matmul_input2, keep_frac):
        input2 = to_matmul_input2.transpose(-2, -1)
        masker = rl.Masker(prune_ratio=1-keep_frac)
        mask1 = masker(input1)
        mask2 = masker(input2)
        shape_x_1, mask_x_1, sparse_x_1 = rl.sparsify(input1, mask1, with_batch_size=False)
        shape_x_2, mask_x_2, sparse_x_2 = rl.sparsify(input2, mask2, with_batch_size=False)
        ctx.save_for_backward(shape_x_1, mask_x_1, sparse_x_1, shape_x_2, mask_x_2, sparse_x_2)
        with torch.autograd.grad_mode.no_grad():
            output = input1.matmul(to_matmul_input2)
        return output

    @staticmethod # backward很慢。
    def backward(ctx, grad_output):
        shape_x_1, mask_x_1, sparse_x_1, shape_x_2, mask_x_2, sparse_x_2 = ctx.saved_tensors

        sparse_x_1 = sparse_x_1.float()
        sparse_x_2 = sparse_x_2.float()

        input1 = rl.unsparsify(shape_x_1, mask_x_1, sparse_x_1)
        input2 = rl.unsparsify(shape_x_2, mask_x_2, sparse_x_2)
        if ctx.needs_input_grad[0]:
            grad_input1 = grad_output.matmul(input2.to(dtype=grad_output.dtype))
        if ctx.needs_input_grad[1]:
            grad_input2 = input1.transpose(-2, -1).to(dtype=grad_output.dtype).matmul(grad_output)
        return grad_input1, grad_input2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

class BackRazorLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True,
                 keep_frac = 0.5):
        super(BackRazorLayerNorm, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.tag = 'layernorm'
        self.keep_frac = 0.5

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        y = backRazorLayerNorm.apply(x, self.normalized_shape, self.weight, self.bias, self.keep_frac, self.eps)
        return y

class backRazorLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias, keep_frac, eps):
        print('=================backrazor sparse layernorm forward=================')
        masker = rl.Masker(prune_ratio=1-keep_frac)
        mask = masker(input)
        shape_x, mask_x, sparse_x = rl.sparsify(input, mask)

        ctx.save_for_backward(shape_x, mask_x, sparse_x)

        if input.dtype != weight.data.dtype:
            input = input.to(dtype=weight.data.dtype)
        eps = ctx.eps
        total_dim = torch.prod(torch.tensor(normalized_shape))
        input = input.reshape(-1, total_dim)
        grad_output = grad_output.reshape(-1, total_dim)
        _, D = input.shape
        input_mean = torch.mean(input, dim=-len(normalized_shape), keepdim=True) # 用dense的
        input_var = torch.var(input, dim=-len(normalized_shape), keepdim=True, unbiased=False)
        inputmu = input-input_mean
        input_var = input_var.expand_as(inputmu)
        inputivar = torch.sqrt(input_var+eps)
        with torch.autograd.grad_mode.no_grad():
            ctx.layer_norm_parameters =  (input_mean, inputivar, weight, D) # 传递需要norm的维度
            ctx.eps = eps
            out = torch.layer_norm(input, normalized_shape, weight, bias, eps)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        tensors = ctx.saved_tensors
        shape_x, mask_x, sparse_x = tensors
        sparse_x = sparse_x.float()
        input = rl.unsparsify(shape_x, mask_x, sparse_x, with_batch_size=True)
        input_mean, inputivar, weight, D = ctx.layer_norm_parameters
        inputmu = input-input_mean # here use sparse input
        grad_input = grad_weight = grad_bias = None
        output_mask = [ctx.needs_input_grad[0], ctx.needs_input_grad[2], ctx.needs_input_grad[3]]
        grad_input, grad_weight, grad_bias = layernorm_backward(grad_output, (inputmu, inputivar, weight, D), output_mask)
        ctx.layer_norm_parameters = None

        return grad_input, None, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None


def layernorm_backward(dout, cache, output_mask):
    dx, dgamma, dbeta = None, None, None
    (xmu, xivar, gamma, D) = cache
    dlxhat = dout*gamma
    dxhatx = 1/xivar
    dlvar = -0.5*torch.sum(gamma*xmu*xivar**(-3)*dout,axis=1,keepdims=True)
    dlvarx = 2*xmu/D
    dlmu = -1.*torch.sum(dlxhat/xivar,dim=1,keepdims=True)-2.*torch.sum(dlvar*xmu,dim=1,keepdims=True)/D
    if output_mask[0]:
        dx = dlxhat*dxhatx + dlvar*dlvarx + dlmu/D
    if output_mask[1]:
        dgamma = torch.sum(dout*xmu/xivar,dim=0,keepdims=True)
    if output_mask[2]:
        dbeta = torch.sum(dout,dim=0,keepdims=True)
    return dx, dgamma, dbeta

class BackRazorSoftmaxMatMul(nn.Module):
    def __init__(self, dim=-1, keep_frac=0.5,sparse_mode = 'norm'):
        super(BackRazorSoftmaxMatMul, self).__init__()
        self.dim = dim
        self.sparse_mode = sparse_mode
        self.keep_frac = keep_frac

    def forward(self, x1, x2):
        y = backRazorSoftmaxMatmul.apply(x1, x2, self.dim, self.keep_frac)
        return y


class backRazorSoftmaxMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, to_matmul_input2, dim, keep_frac = 0.5):
        input1_sm = F.softmax(input1, dim)
        input2 = to_matmul_input2.transpose(-2, -1)
        masker = rl.Masker(prune_ratio=1-keep_frac)
        mask1 = masker(input1_sm)
        mask2 = masker(input2)

        shape_x_1, mask_x_1, sparse_x_1 = rl.sparsify(input1_sm, mask1)
        shape_x_2, mask_x_2, sparse_x_2 = rl.sparsify(input2, mask2)

        ctx.save_for_backward(shape_x_1, shape_x_2, mask_x_1, mask_x_2, sparse_x_1, sparse_x_2)

        output = input1_sm.matmul(input2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input1 = grad_input2 = None

        tensors = ctx.saved_tensors
        shape_x_1, shape_x_2, mask_x_1, mask_x_2, sparse_x_1, sparse_x_2 = tensors

        sparse_x_1 = sparse_x_1.float()
        sparse_x_2 = sparse_x_2.float()

        input1_sm = rl.unsparsify(shape_x_1, mask_x_1, sparse_x_1)
        input2 = rl.unsparsify(shape_x_2, mask_x_2, sparse_x_2)

        if ctx.needs_input_grad[0]:
            grad_input_1 = softmax_grad(grad_output.matmul(input2.to(dtype=grad_output.dtype)), input1_sm)
        if ctx.needs_input_grad[1]:
            grad_input_2 = input1_sm.transpose(-2, -1).to(dtype=grad_output.dtype).matmul(grad_output)
        return grad_input_1, grad_input_2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

def softmax_grad(grad_output,out):
    grad_input = out * (grad_output - (grad_output * out).sum(dim=-1, keepdim=True))
    return grad_input





