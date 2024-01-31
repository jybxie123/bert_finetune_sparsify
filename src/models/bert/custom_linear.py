
import sys
import logging

# from mesa import custom_quant
# from mesa import native
from . import rand_layers as rl
from pdb import set_trace
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================== our method ==================
class OurLinear(torch.nn.Linear):
    def __init__(self, *args, keep_frac=0.5, linear_idx = None, act_type = None, **kwargs):
        super(OurLinear, self).__init__(*args, **kwargs)
        self.keep_frac = keep_frac
        self.step_idx = 0
        self.linear_idx = linear_idx
        self.act_type = act_type

    def forward(self, input, retain=False, skip_rand=False):
        if not retain:
            self.random_seed = torch.randint(low=10000000000, high=99999999999, size=(1,))

        if skip_rand:
            keep_frac = 1.0
        else:
            keep_frac = self.keep_frac

        result = sparseMatMul.apply(input, self.weight, self.bias, keep_frac)
        cal_zero_ratio(result, self.linear_idx, self.step_idx, self.act_type)
        # print('Ourlinear step idx + 1')
        self.step_idx += 1
        return result

def cal_zero_ratio(layer_output, layer_idx, iteration, act_type):
    temp_total = float(layer_output.view(-1).shape[0])
    temp_act = torch.sum(torch.eq(layer_output, 0).float()) #eq: equal to 0
    ratio = temp_act/temp_total
    with open(f'zero-ratio-of-bert-layer{act_type}.txt', 'a+') as file:
        file.write(f"iteration:{iteration};layer{layer_idx}:{ratio}\n")



# 单个hidden state
class sparseMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, keep_frac):
        gather_index = rl.get_batch_score(input, keep_frac=keep_frac) # 这里暂时范数计算是也考虑了w，因为w大的话自然影响也大
        sparse_input = rl.get_sparse_input(input, gather_index)
        ctx.save_for_backward(sparse_input.to_sparse(), weight, bias)
        with torch.autograd.grad_mode.no_grad():
            return F.linear(input, weight, bias=bias)

    @staticmethod
    def backward(ctx, grad_output):
        sparse_input, weight, bias = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            input_grad = grad_output.matmul(weight.to(dtype=grad_output.dtype)) #这里这俩都是dense的
        if ctx.needs_input_grad[1]:
            grad_output = grad_output.float()
            sparse_input = sparse_input.float()
            input = sparse_input.to_dense()
            weight_grad = grad_output.transpose(-2, -1).matmul(input.to(dtype=grad_output.dtype)) # bmm
        if bias is not None and ctx.needs_input_grad[2]:
            bias_grad = grad_output.sum(0)
        return input_grad, weight_grad, bias_grad, None, None, None, None

class OurMatMul(nn.Module):
    def __init__(self, args=None, keep_frac=0.5, linear_idx = None, act_type = None ):
        super(OurMatMul, self).__init__()
        self.keep_frac = keep_frac
        self.step_idx = 0
        self.linear_idx = linear_idx
        self.act_type = act_type
    def forward(self, x1, x2):
        # print('Our sparse matmul')
        y = doubleSparseMatMul.apply(x1, x2, self.keep_frac)
        return y

import time

# new one
class doubleSparseMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, to_matmul_input2, keep_frac):
        output = input1.matmul(to_matmul_input2)
        input2 = to_matmul_input2.transpose(-2, -1)
        gather_index = rl.get_batch_score(input1, input2, keep_frac)
        input1 = rl.get_sparse_input(input1, gather_index)
        input2 = rl.get_sparse_input(input2, gather_index)
        ctx.save_for_backward(input1.to_sparse(), input2.to_sparse())
        return output

    @staticmethod # backward很慢。
    def backward(ctx, grad_output):
        grad_input1 = grad_input2 = None    
        sparse_input1, sparse_input2 = ctx.saved_tensors
        input1 = sparse_input1.to_dense()
        input2 = sparse_input2.to_dense()
        if ctx.needs_input_grad[0]:
            # grad_input1 = torch.sparse.mm(grad_output, sparse_input2)
            grad_input1 = grad_output.matmul(input2.to(dtype=grad_output.dtype))
        if ctx.needs_input_grad[1]:
            # grad_input2 = torch.sparse.mm(sparse_input1.transpose(-2, -1).to(dtype=grad_output.dtype), grad_output)
            grad_input2 = input1.transpose(-2, -1).to(dtype=grad_output.dtype).matmul(grad_output)
        return grad_input1, grad_input2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class OurLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, keep_frac=0.5):
        super(OurLayerNorm, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.tag = 'layernorm'
        self.keep_frac = keep_frac

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        # print('Our sparse norm')
        y = sparse_layer_norm.apply(x, self.normalized_shape, self.weight, self.bias, self.eps, self.keep_frac)
        return y

class sparse_layer_norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias, eps=1e-5, keep_frac = 0.5): # 这里normalized shape是一个int，就是hidden size
        if input.dtype != weight.data.dtype:
            input = input.to(dtype=weight.data.dtype)
        # here is a linear func
        with torch.autograd.grad_mode.no_grad():
            gather_index = rl.get_batch_score(input,keep_frac= keep_frac) # 这里暂时范数计算是也考虑了w，因为w大的话自然影响也大
            sparse_input = rl.get_sparse_input(input, gather_index)
            ctx.layer_norm_parameters =  normalized_shape # 传递需要norm的维度
            input_mean = torch.mean(input, dim=-len(normalized_shape), keepdim=True)
            input_var = torch.var(input, dim=-len(normalized_shape), keepdim=True,unbiased=False)
            inputmu = input-input_mean
            inputivar = torch.sqrt(input_var+eps)
            ctx.eps = eps
            ctx.save_for_backward(sparse_input.to_sparse(), weight, bias, gather_index)
            out = weight*inputmu/inputivar + bias
        return out
    # 这里试了超级久，最后发现backward实际的做法并不能用在这种情景下。我们的forward用的是另一个向量，因此input没办法计算。
    @staticmethod
    def backward(ctx, grad_output):
        # print('======================start of the backward========================')
        sparse_input, weight, bias, gather_index = ctx.saved_tensors
        ori_input = sparse_input.to_dense()
        ori_input_shape = ori_input.shape
        normalized_shape = ctx.layer_norm_parameters
        eps = ctx.eps
        total_dim = torch.prod(torch.tensor(normalized_shape))
        input = ori_input.reshape(-1, total_dim)
        grad_output = grad_output.reshape(-1, total_dim)
        N, D = input.shape
        input.requires_grad_(True)
        weight.requires_grad_(True)
        bias.requires_grad_(True)
        input_mean = torch.mean(input, dim=-len(normalized_shape), keepdim=True)
        input_var = torch.var(input, dim=-len(normalized_shape), keepdim=True, unbiased=False)
        inputmu = input-input_mean
        input_var = input_var.expand_as(inputmu)
        inputivar = torch.sqrt(input_var+eps)
        grad_weight = torch.sum(grad_output*inputmu/inputivar,dim=0,keepdims=True)
        grad_bias = torch.sum(grad_output,dim=0,keepdims=True)
        dlxhat = grad_output*weight
        dxhatx = 1/inputivar
        dlvar = -0.5*torch.sum(weight*inputmu*inputivar**(-3)*grad_output,axis=1,keepdims=True)
        dlvarx = 2*inputmu/D
        dlmu = -1.*torch.sum(dlxhat/inputivar,dim=1,keepdims=True)-2.*torch.sum(dlvar*inputmu,dim=1,keepdims=True)/D
        grad_input = dlxhat*dxhatx + dlvar*dlvarx + dlmu/D
        grad_input = grad_input.reshape(ori_input_shape)
        ctx.layer_norm_parameters = None
        return grad_input, None, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None

def layernorm_backward(dout, normalized_shape, cache):
    dx, dgamma, dbeta = None, None, None
    (x, xmu, xivar, gamma) = cache
    D = normalized_shape[-1]
    dgamma = torch.sum(dout*xmu/xivar,dim=0,keepdims=True)
    dbeta = torch.sum(dout,dim=0,keepdims=True)
    dlxhat = dout*gamma
    dxhatx = 1/xivar
    dlvar = -0.5*torch.sum(gamma*xmu*xivar**(-3)*dout,axis=1,keepdims=True)
    dlvarx = 2*xmu/D
    dlmu = -1.*torch.sum(dlxhat/xivar,dim=1,keepdims=True)-2.*torch.sum(dlvar*xmu,dim=1,keepdims=True)/D
    dx = dlxhat*dxhatx + dlvar*dlvarx + dlmu/D
    return dx, dgamma, dbeta

class OurSoftmaxMatMul(nn.Module):
    def __init__(self, dim=-1):
        super(OurSoftmaxMatMul, self).__init__()
        self.dim = dim

    def forward(self, x1, x2):
        y = softmax_matmul.apply(x1, x2, self.dim)
        return y


class softmax_matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, to_matmul_input2, dim):
        input1_sm = softmax(input1, dim)
        input2 = to_matmul_input2.transpose(-2, -1)
        ctx.dim = dim
        print('input1 shape : ', input1.shape)
        print('input2 shape : ', input2.shape)
        gather_index = rl.get_batch_score(input1, input2, 0.5)
        sparse_x_1 = rl.get_sparse_input(input1_sm, gather_index)
        sparse_x_2 = rl.get_sparse_input(input2, gather_index)
        ctx.save_for_backward(sparse_x_1.to_sparse(), sparse_x_2.to_sparse())
        output = input1_sm.matmul(to_matmul_input2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input_1 = grad_input_2 = None
        sparse_x_1, sparse_x_2 = ctx.saved_tensors
        
        sparse_x_1 = sparse_x_1.float().requires_grad_(True)
        sparse_x_2 = sparse_x_2.float().requires_grad_(True)

        input_1_sm = sparse_x_1.to_dense()
        input_2 = sparse_x_2.to_dense()
        if ctx.needs_input_grad[0]:
            grad_input_1_sm = grad_output.matmul(input_2.to(dtype=grad_output.dtype))
            grad_input_1 = softmax_grad(grad_input_1_sm, input_1_sm)
        if ctx.needs_input_grad[1]:
            grad_input_2 = input_1_sm.transpose(-2, -1).to(dtype=grad_output.dtype).matmul(grad_output)
        return grad_input_1, grad_input_2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

def softmax(features, dim=-1):
    exps = torch.exp(features)
    _sum = torch.sum(exps, dim=dim, keepdim=True)
    return exps / _sum

def softmax_grad(grad_output,out):
    temp1 = out*(1-out)
    # 对角线上的值
    temp2 = torch.diag_embed(temp1)
    temp3 = out.unsqueeze(-1)
    temp4 = out.unsqueeze(-2)
    # 非对角线上的值需要列乘法
    temp5 = torch.matmul(temp3, temp4)*(-1)
    # 去掉对角线上的值
    temp5[...,range(temp5.shape[-2]),range(temp5.shape[-1])] = 0
    temp6 = temp5+temp2
    # 现在层的梯度需要乘以上一层的梯度，并且做列变换。
    grad_input = grad_output.unsqueeze(-2).matmul(temp6).squeeze(-2)
    return grad_input

# ================== back razor ==================
class linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, mask=None, quantize=True, half=False, clip_val=None, level=256, iteration=None, ema_decay=None, quant_groups=None, shift=None):
        shape_x, mask_x, sparse_x = rl.sparsify(x, mask, with_batch_size=False)
        if half and (not quantize):
            sparse_x = sparse_x.half()
        # if quantize:
        #     custom_quant.Quant.forward(ctx, sparse_x, clip_val, level, iteration, ema_decay, quant_groups, shift)
        #     ctx.save_for_backward(weight, bias, shape_x, mask_x)
        # else:
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


# class LinearSparse(nn.Linear, custom_quant.Quant):
class LinearSparse(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, args=None, logger=None, quant_groups=1, masker=None,
                 quantize=True, half=False, act_prune=False):
        super(LinearSparse, self).__init__(in_features, out_features, bias=bias)
        # custom_quant.Quant.__init__(self, args=args, logger=logger, quant_groups=quant_groups)
        self.masker = masker
        self.quantize = quantize
        self.act_prune = act_prune
        self.half = half
        self.tag = 'fc'

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        # print("type(x) is {}".format(type(x)))
        if self.masker is not None and self.training:
            mask = self.masker(x)
            # print("mask sum is {}".format((~mask).sum()))
            if self.act_prune:
                x = x * mask
            y = linear.apply(x, self.weight, self.bias, mask, self.quantize, self.half, self.clip_val, self.level,
                             self.iteration, self.ema_decay, self.quant_groups, self.shift)
        else:
            y = F.linear(x, self.weight, self.bias)
        return y

