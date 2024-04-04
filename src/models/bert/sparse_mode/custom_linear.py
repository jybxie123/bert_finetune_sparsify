
import sys
import logging

# from mesa import custom_quant
# from mesa import native
sys.path.insert(0, '/disk3/Haonan/yanbo_random/bert_finetune_sparsify/src/models/bert')
import sparse_mode.rand_layers as rl
from pdb import set_trace
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================== our method ==================
class OurLinear(torch.nn.Linear):
    def __init__(self, *args, keep_frac=0.5, linear_idx = None, act_type = None, sparse_mode = 'norm', **kwargs):
        super(OurLinear, self).__init__(*args, **kwargs)
        self.keep_frac = keep_frac
        # self.step_idx = 0
        # self.linear_idx = linear_idx
        # self.act_type = act_type
        self.sparse_mode = sparse_mode
        
    def forward(self, input):
        result = sparseMatMul.apply(input, self.weight, self.bias, self.keep_frac, self.sparse_mode)
        # cal_zero_ratio(result, self.linear_idx, self.step_idx, self.act_type)
        # print('Ourlinear step idx + 1')
        # self.step_idx += 1
        return result

# def cal_zero_ratio(layer_output, layer_idx, iteration, act_type):
#     temp_total = float(layer_output.view(-1).shape[0])
#     temp_act = torch.sum(torch.eq(layer_output, 0).float()) #eq: equal to 0
#     ratio = temp_act/temp_total
#     with open(f'zero-ratio-of-bert-layer{act_type}.txt', 'a+') as file:
#         file.write(f"iteration:{iteration};layer{layer_idx}:{ratio}\n")


# 单个hidden state
class sparseMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, keep_frac, sparse_mode):
        with torch.autograd.grad_mode.no_grad():
            result = F.linear(input, weight, bias=bias)
        sparse_index = rl.get_batch_score(input, keep_frac = keep_frac, sparse_mode = sparse_mode)
        sparse_input = rl.get_sparse_input(input, sparse_index)
        ctx.save_for_backward(sparse_input.to_sparse(), weight, bias)
        return result

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
        del sparse_input, grad_output, weight, bias
        return input_grad, weight_grad, bias_grad, None, None, None, None

class OurMatMul(nn.Module):
    def __init__(self, args=None, keep_frac=0.5, linear_idx = None, act_type = None, sparse_mode = 'norm'):
        super(OurMatMul, self).__init__()
        self.keep_frac = keep_frac
        self.sparse_mode = sparse_mode
        # self.step_idx = 0
        # self.linear_idx = linear_idx
        # self.act_type = act_type

    def forward(self, x1, x2):
        # print('Our sparse matmul')
        y = doubleSparseMatMul.apply(x1, x2, self.keep_frac, self.sparse_mode)
        return y

import time

# new one
class doubleSparseMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, to_matmul_input2, keep_frac, sparse_mode):
        input2 = to_matmul_input2.transpose(-2, -1)
        with torch.autograd.grad_mode.no_grad():
            output = input1.matmul(to_matmul_input2)
        gather_index = rl.get_batch_score(input1, input2, keep_frac, sparse_mode=sparse_mode)
        sparse_input1 = rl.get_sparse_input(input1, gather_index)
        sparse_input2 = rl.get_sparse_input(input2, gather_index)
        ctx.save_for_backward(sparse_input1.to_sparse(), sparse_input2.to_sparse())
        del input1, input2, gather_index, sparse_input1, sparse_input2, to_matmul_input2
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
        # del grad_output, input1, input2, sparse_input1, sparse_input2
        return grad_input1, grad_input2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class OurLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, keep_frac=0.5, sparse_mode='norm'):
        super(OurLayerNorm, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.tag = 'layernorm'
        self.keep_frac = keep_frac
        self.sparse_mode = sparse_mode

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        # print('Our sparse norm')
        y = sparse_layer_norm.apply(x, self.normalized_shape, self.weight, self.bias, self.eps, self.keep_frac, self.sparse_mode)
        return y

class sparse_layer_norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias, eps=1e-5, keep_frac = 0.5, sparse_mode='norm'): # 这里normalized shape是一个int，就是hidden size
        if input.dtype != weight.data.dtype:
            input = input.to(dtype=weight.data.dtype)
        # here is a linear func
        with torch.autograd.grad_mode.no_grad():
            gather_index = rl.get_batch_score(input, keep_frac = keep_frac, sparse_mode=sparse_mode)
            sparse_input = rl.get_sparse_input(input, gather_index)
            input_mean = torch.mean(input, dim=-len(normalized_shape), keepdim=True)
            input_var = torch.var(input, dim=-len(normalized_shape), keepdim=True, unbiased=False).expand_as(input_mean)
            inputivar = torch.sqrt(input_var+eps)
            ctx.layer_norm_parameters =  input_mean, inputivar, normalized_shape # 传递需要norm的维度
            ctx.save_for_backward(sparse_input.to_sparse(), weight)
            out = torch.layer_norm(input, normalized_shape, weight, bias, eps)
            del input, gather_index, sparse_input, input_mean, input_var, inputivar, weight, bias
        return out
    # 这里试了超级久，最后发现backward实际的做法并不能用在这种情景下。我们的forward用的是另一个向量，因此input没办法计算。
    @staticmethod
    def backward(ctx, grad_output):
        sparse_input, weight = ctx.saved_tensors
        ori_input = sparse_input.to_dense()
        input_mean, inputivar, normalized_shape = ctx.layer_norm_parameters
        total_dim = torch.prod(torch.tensor(normalized_shape))
        input = ori_input.reshape(-1, total_dim)
        grad_output = grad_output.reshape(-1, total_dim)
        _, D = input.shape
        inputmu = input-input_mean
        output_mask = [ctx.needs_input_grad[0], ctx.needs_input_grad[2], ctx.needs_input_grad[3]]
        grad_input, grad_weight, grad_bias = layernorm_backward(grad_output, (inputmu, inputivar, weight, D), output_mask)
        ctx.layer_norm_parameters = None
        del sparse_input, ori_input, input_mean, inputivar, input, grad_output, inputmu, D, total_dim, weight
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


class OurSoftmaxMatMul(nn.Module):
    def __init__(self, dim=-1, keep_frac=0.5,sparse_mode = 'norm'):
        super(OurSoftmaxMatMul, self).__init__()
        self.dim = dim
        self.sparse_mode = sparse_mode
        self.keep_frac = keep_frac

    def forward(self, x1, x2):
        y = softmax_matmul.apply(x1, x2, self.dim, self.keep_frac,self.sparse_mode)
        return y


class softmax_matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, to_matmul_input2, dim, keep_frac, sparse_mode):
        input1_sm = torch.softmax(input1, dim)
        input2 = to_matmul_input2.transpose(-2, -1)
        ctx.dim = dim
        gather_index = rl.get_batch_score(input1_sm, input2, keep_frac = keep_frac, sparse_mode = sparse_mode)
        sparse_x_1 = rl.get_sparse_input(input1_sm, gather_index)
        sparse_x_2 = rl.get_sparse_input(input2, gather_index)
        ctx.save_for_backward(sparse_x_1.to_sparse(), sparse_x_2.to_sparse())
        with torch.autograd.grad_mode.no_grad():
            output = input1_sm.matmul(to_matmul_input2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input_1 = grad_input_2 = None
        sparse_x_1, sparse_x_2 = ctx.saved_tensors
        # sparse_x_1 = sparse_x_1.float().requires_grad_(True)
        # sparse_x_2 = sparse_x_2.float().requires_grad_(True)
        sparse_x_1 = sparse_x_1.to_dense()
        sparse_x_2 = sparse_x_2.to_dense()
        if ctx.needs_input_grad[0]:
            grad_input_1 = softmax_grad(grad_output.matmul(sparse_x_2.to(dtype=grad_output.dtype)), sparse_x_1)
        if ctx.needs_input_grad[1]:
            grad_input_2 = sparse_x_1.transpose(-2, -1).to(dtype=grad_output.dtype).matmul(grad_output)
        return grad_input_1, grad_input_2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

def softmax(features, dim=-1):
    exps = torch.exp(features)
    _sum = torch.sum(exps, dim=dim, keepdim=True)
    return exps / _sum

def softmax_grad(grad_output,out):
    grad_input = out * (grad_output - (grad_output * out).sum(dim=-1, keepdim=True))
    return grad_input

# attention matrix乘法，和kq的乘法没什么太大区别，用统一的一套技术做稀疏。有故事性。
# story：我们拿出了一套逻辑，应用到了各个乘法的地方，得到了提升。
# 如果不同的地方遵循的逻辑不同，可能需要单独解释，不如统一的方法来的好。
