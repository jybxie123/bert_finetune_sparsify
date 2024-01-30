import torch
import numpy as np

# return sparse
def get_sparse_input(input, gather_index):
    gathered_input = get_selected_indices(input, gather_index)
    # sparse_input = denseToSparse(gathered_input)
    # return sparse_input
    return gathered_input # 这里假设后续用to sparse来替代

def get_selected_indices(input, indices):
    # print('input shape : ', input.shape)
    # print('indices shape : ', indices.shape)
    if len(input.shape) == 1:
        input[indices] = 0
    elif len(input.shape) == 2:
        input[:,indices] = 0
    elif len(input.shape) == 3:
        input[:,:,indices] = 0
    elif len(input.shape) == 4:
        input[:,:,:,indices] = 0
    elif len(input.shape) == 5:
        input[:,:,:,:,indices] = 0
    else:
        # print('input shape is : ', input.shape)
        raise ValueError('input shape is not supported')
    return input

# 对除了要筛选的维度以外的所有维度计算范数
def get_batch_score(input1, input2 = None,  keep_frac = 0.5, select_method='norm'):
    if input2 is not None:
        if input1.shape[-1] != input2.shape[-1]:
            raise ValueError('input1 and input2 shape is not supported')
        if len(input2.shape) == 1:
            raise ValueError('input2 shape is not supported')
    kept_feature_size = int(input1.shape[-1] * keep_frac + 0.999)
    if len(input1.shape) == 1:
        raise ValueError('input1 shape is not supported')
    if select_method == 'norm':
        # print('input shape, batch size, feature len, kept: ',input.shape, batch_size, feature_len, kept_feature_size)
        input_flatted1 = input1.reshape(-1, input1.shape[-1])
        temp_input1_norm = torch.norm(input_flatted1, dim=0) # 对列求范数 
        sf_temp_input1_norm = torch.softmax(temp_input1_norm, dim=0)
        if input2 is not None:
            input_flatted2 = input2.reshape(-1, input2.shape[-1])
            temp_input2_norm = torch.norm(input_flatted2, dim=0) # 对列求范数   
            sf_temp_input2_norm = torch.softmax(temp_input2_norm, dim=0)
            score = sf_temp_input1_norm / input1.shape[-2] + sf_temp_input2_norm / input2.shape[-2] # （加权）
        else:
            score = sf_temp_input1_norm / input1.shape[-2]
        # 这里的index是
        gather_index = torch.argsort(score, descending=True)[..., :kept_feature_size]
        # gather_index = torch.argsort(score, descending=True)[kept_feature_size:]
        # print('gather_index shape : ',gather_index.shape)
        return gather_index
    elif select_method == 'rand':
        full_indices = torch.randperm(input1.size()[-1]).to(input1.device)
        gather_index = full_indices[kept_feature_size:]
        # print('gather_index shape : ',gather_index.shape)
        return gather_index

# ===========================back razor===========================

# def get_batch_score(input, batch_size, feature_len, kept_feature_size):
#     activation_mag = torch.abs(input)
#     threshold, _ = torch.kthvalue(activation_mag.flatten(1), kept_feature_size)
#     while len(threshold.shape) < len(activation_mag.shape):
#         threshold = threshold.unsqueeze(-1)
#     mask = activation_mag >= threshold

#     # print("mask density is {}".format(mask.float().mean()))
#     # idle mask
#     # mask = torch.ones_like(activation).to(torch.bool)
#     return mask

# dense tensor to sparse tensor
def denseToSparse(x):
    indices = torch.nonzero(x, as_tuple=True)
    values = x[indices]
    stacked = torch.stack(indices)
    sparse_x = torch.sparse_coo_tensor(stacked, values, x.size())
    return sparse_x

# sparse tensor to dense tensor
def sparseToDense(sparse_x):
    dense_tensor = sparse_x.to_dense()
    return dense_tensor

# 原来的稀疏方法，还不够简单
# # input should be viewed into [xxx, feature_len], then we give the output with masked zero.
# def select_columns(input, col_idx):
#     new_input = input.clone()
#     mask = torch.zeros(new_input.size(1), dtype=torch.bool)
#     mask[col_idx] = True
#     new_input[:, ~mask] = 0
#     return new_input

'''
流程：
input原维度
变为b f类型的维度
先在f上找出需要稀疏的维度 index
再将原矩阵稀疏为目标矩阵，但是0没有去掉
将目标矩阵转为稀疏矩阵类型，存起来。
稀疏矩阵回头可以恢复为目标矩阵，但是这里需要再view为原shape才可以作乘法。
'''

# test

import torch.nn.functional as F

if __name__ == "__main__":
    with torch.no_grad():
        # # test index column generation
        # input = torch.rand(3)
        # print("input is {}".format(input))
        # y = input.repeat(4,1)
        # print("y is {}".format(y))

        # test select columns 
        x = torch.randint(3, (3,3,3))
        print("x is {}".format(x))

        idx = torch.tensor([0,1])
        new_w = select_columns(x, idx)
        print("x is {}".format(x))
        print("new_w is {}".format(new_w))

        # # test dense to sparse
        # x = torch.randint(3, (3,3,3))
        # x[1,2] = 0
        # x[2,1] = 0
        # print(x)
        # dim_len = len(x.shape)
        # indices = torch.nonzero(x, as_tuple=True)

        # values = x[indices]
        # print("indices : ", indices)
        # print("values : ", values)
        # # indices = torch.tensor(indices)
        # # print("indices : ", indices)

        # stacked = torch.stack(indices)
        # print("stacked : ", stacked)

        # sparse_x = torch.sparse_coo_tensor(stacked, values, x.size())
        # dense_tensor = sparse_x.to_dense()
        # print(sparse_x)
        # print(dense_tensor)
        # print('type: ', type(sparse_x), type(dense_tensor), type(indices), type(values))

        # # test input2sparse
        # input = torch.rand(3,4)
        # weight = torch.rand(2,4)
        # bias = torch.rand(3,2)
        # print("input is {}".format(input))
        # keep_frac = 0.5
        # # gather_index = get_batch_score(input, 3, 4, kept_feature_size)
        # sparse, gather_index = input2sparse(input, keep_frac)
        # print("gather_index is {}".format(gather_index))
        # print("sparse is {}".format(sparse))
        # new_input = sparse2input(sparse, (3,4), gather_index)
        # print("new_input is {}".format(new_input))

        # def cln(t):
        #     if t is None:
        #         return None
        #     ct = t.clone().detach()
        #     ct.requires_grad_(True)
        #     return ct

        # cinput = cln(input)
        # cweight = cln(weight)
        # cbias = cln(bias)
        # grad_output = torch.rand(3,2)
        # grad_output.requires_grad_(True)
        # with torch.autograd.grad_mode.enable_grad():
        #     output = F.linear(cinput, cweight, bias=cbias)
        # # bias_grad_input, input_grad_input, weight_grad_input = output.grad_fn(grad_output)
        # input_grad_input, weight_grad_input, bias_grad_input = torch.autograd.grad(output, (cinput, cweight, cbias), grad_output)
        # print('backward grad_output : ',grad_output)
        # print('grad : ', input_grad_input, weight_grad_input, bias_grad_input)
        # print('backward shape : ',input_grad_input.shape,weight_grad_input.shape,bias_grad_input.shape)




# ================== back razor(not used) ==================
from pdb import set_trace
class Masker(object):
    def __init__(self, prune_ratio):
        self.prune_ratio = prune_ratio
    @torch.no_grad()
    def __call__(self, activation):
        num_small = int(np.clip(activation[0].numel() * self.prune_ratio, 1, activation[0].numel()))
        activation_mag = torch.abs(activation)
        threshold, _ = torch.kthvalue(activation_mag.flatten(1), num_small)
        while len(threshold.shape) < len(activation_mag.shape):
            threshold = threshold.unsqueeze(-1)
        mask = activation_mag >= threshold
        return mask

def sparsify(tensor, mask, with_batch_size=False):
    shape = tensor.shape
    shape = torch.tensor(shape)

    mask = mask.reshape(-1)
    sparse = tensor.reshape(-1)[mask]
    if with_batch_size:
        sparse = sparse.reshape(shape[0], -1)
    else:
        sparse = sparse.unsqueeze(0)

    # add bits to make it divisible by 8
    if mask.shape[0] % 8 != 0:
        add_bits = 8 - (mask.shape[0] % 8)
        mask = torch.cat([mask, torch.zeros(add_bits, dtype=mask.dtype, device=mask.device)], dim=0)

    # mask = packbit.packbits_padded(mask)

    # idle value
    # mask = torch.ones(1, device=tensor.device)
    # sparse = tensor

    return shape, mask, sparse

def unsparsify(shape, mask, sparse, with_batch_size=False):
    # mask = packbit.unpackbits_padded(mask).to(dtype=torch.bool)
    mask = mask.to(dtype=torch.bool)

    if with_batch_size:
        sparse = sparse.view(-1)
    else:
        sparse = sparse.squeeze(0)

    shape = torch.Size(shape)
    dense = torch.zeros(shape.numel(), device=sparse.device, dtype=sparse.dtype)
    dense[mask[:shape.numel()]] = sparse

    return dense.reshape(shape)

    # idle
    # return sparse

