import torch
import numpy as np
from memory_profiler import profile

# from config.training_config import train_config as TRAIN_CONFIG
# train_config = TRAIN_CONFIG()
# def profile_to_file():
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             with open(f"{train_config.output_dir}/{train_config.expr_name}.txt", 'a') as f:
#                 prof = profile(func, stream=f)
#                 return prof(*args, **kwargs)
#         return wrapper
#     return decorator

# return sparse
def get_sparse_input(input, gather_index):
    return get_selected_indices(input, gather_index)
    # sparse_input = denseToSparse(gathered_input)
    # return sparse_input
    # return gathered_input 

# 改为就地操作，因此现在计算forward要在稀疏化之前
# @profile_to_file()
def get_selected_indices(input, indices):
    mask = torch.ones(input.shape[-1], dtype=torch.bool)
    mask[indices] = False
    delete_index = mask.nonzero().squeeze()
    input = input.clone()
    if len(input.shape) == 1:
        raise ValueError('input shape is not supported')
    elif len(input.shape) == 2:
        input[:,delete_index] = 0
    elif len(input.shape) == 3:
        input[:,:,delete_index] = 0
    elif len(input.shape) == 4:
        input[:,:,:,delete_index] = 0
    elif len(input.shape) == 5:
        input[:,:,:,:,delete_index] = 0
    else:
        # print('input shape is : ', input.shape)
        raise ValueError('input shape is not supported')
    return input

# 对除了要筛选的维度以外的所有维度计算范数
# @profile_to_file()
def get_batch_score(input1, input2 = None,  keep_frac = 0.5, sparse_mode='norm'):
    if input2 is not None:
        if input1.shape[-1] != input2.shape[-1]:
            print('input1 shape : ',input1.shape)
            print('input2 shape : ',input2.shape)
            raise ValueError('input1 and input2 shape is not supported')
        if len(input2.shape) == 1:
            raise ValueError('input2 shape is not supported')
    kept_feature_size = int(input1.shape[-1] * keep_frac + 0.999)
    if len(input1.shape) == 1:
        raise ValueError('input1 shape is not supported')
    if sparse_mode == 'norm':
        # print('input shape, batch size, feature len, kept: ',input.shape, batch_size, feature_len, kept_feature_size)
        shape1 = input1.shape
        input1 = input1.reshape(-1, shape1[-1]) # eg：32，12，512，64-> xxx, 64, 64是列！

        # 根据讨论交流的结果，一范数和无穷范数应该是更好的选择
        # temp_input1_norm = torch.norm(input1, dim=0) # 对列求范数 
        # temp_input1_norm = torch.norm(input1, p=1, dim=0) # 对列求范数 
        temp_input1_norm, _ = torch.max(torch.abs(input1), dim=0) # 对列求无穷范数 
        # sf_temp_input1_norm = torch.softmax(temp_input1_norm, dim=0)
        if input2 is not None:
            shape2 = input2.shape
            input2 = input2.reshape(-1, shape2[-1])
            # temp_input2_norm = torch.norm(input2, dim=0) # 对列求范数 

            temp_input2_norm, _ = torch.max(torch.abs(input2), dim=0)# 对列求范数   
            score = temp_input1_norm + temp_input2_norm
            # sf_temp_input2_norm = torch.softmax(temp_input2_norm, dim=0)
            # score = sf_temp_input1_norm / shape1[-2] + sf_temp_input2_norm / shape2[-2] # （加权）
            # score = temp_input1_norm / input1.shape[-2] + temp_input2_norm / input2.shape[-2] # （加权）
        else:
            score = temp_input1_norm
            # score = sf_temp_input1_norm / shape1[-2]
            # score = temp_input1_norm / input1.shape[-2]
        # 这里的index是
        gather_index = torch.argsort(score, descending=True)[..., :kept_feature_size]
        # gather_index = torch.argsort(score, descending=True)[kept_feature_size:]
        # print('gather_index shape : ',gather_index.shape)
        result = gather_index.reshape(-1)
        del temp_input1_norm, score, gather_index, kept_feature_size, shape1, input1  # 删除原始变量
        if input2 is not None:
            del temp_input2_norm, shape2, input2
        # torch.cuda.empty_cache()  # 清空 CUDA 缓存
        return result
    elif sparse_mode == 'rand': # randAD
        full_indices = torch.randperm(input1.size()[-1]).to(input1.device)
        gather_index = full_indices[:kept_feature_size]
        # print('gather_index shape : ',gather_index.shape)
        result = gather_index.clone()  # 创建一个 gather_index 的副本
        del full_indices, gather_index, sparse_mode,   # 删除原始变量
        # torch.cuda.empty_cache()  # 清空 CUDA 缓存
        return result

# ===========================back razor===========================
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

def sparsify(tensor, mask, with_batch_size=False): # false的时候，所有batchsize都合并到同一维度输出。
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



# test


def idx_sparse(sp_out):
    fl_keep_col = sp_out.indices().shape[-1]/reduce(operator.mul, sp_out.size()[:-1]) + 0.999
    print('fl_keep_col : ',fl_keep_col)
    keep_col = int(fl_keep_col)
    print('keep_col : ',keep_col)
    return sp_out.indices()[-1,range(keep_col)],sp_out.size()

from functools import reduce
import operator
def idx_unsparse(new_sp_out):
    non_zero_cols = new_sp_out[0]
    tensor_size = new_sp_out[1]
    # 生成前三个维度的所有组合
    dims = [torch.arange(size) for size in tensor_size[:-1]]
    grid = torch.meshgrid(dims, indexing='ij')
    grid_flat = [g.flatten() for g in grid]
    # 复制每个组合以匹配non_zero_cols的长度
    repeats = non_zero_cols.size(0)
    grid_repeated = [g.repeat_interleave(repeats) for g in grid_flat]
    # 重复非零列索引以匹配前三维的每个位置
    repeated_cols = non_zero_cols.repeat(reduce(operator.mul, tensor_size[:-1]))
    print('non_zero_cols shape : ',non_zero_cols.shape)
    print('tensor_size : ',tensor_size)
    print('grid_repeated shape : ',grid_repeated[0].shape)
    print('repeated_cols shape : ',repeated_cols.shape)
    indices = torch.stack(grid_repeated + [repeated_cols])

    # 调整indices的形状以匹配原始稀疏张量indices的布局
    indices = indices.reshape(len(tensor_size), -1)
    return indices

import torch.nn.functional as F
import time
if __name__ == "__main__":
    start = time.perf_counter()
    input = torch.randn(32,12,512,512)
    idx = get_batch_score(input,None, 0.5, 'norm')
    print(idx)
    print('idx shape : ',idx.shape)
    output = get_sparse_input(input, idx)
    print('output shape : ',output.shape)
    sp_out = output.to_sparse()
    
    ds_out = sp_out.to_dense()
    print('sp_out indices shape : ',sp_out.indices().shape)
    print('sp_out values shape : ',sp_out.values().shape)
    print('ds_out shape : ',ds_out.shape)
    new_sp_out = idx_sparse(sp_out)
    ori_sp_out = idx_unsparse(new_sp_out)
    equal = torch.equal(sp_out.indices(), ori_sp_out)
    print(equal) 
    if not equal:
        result = sp_out.indices() - ori_sp_out
        print(result)
    end = time.perf_counter()
    print('Running time: %s Seconds'%(end-start))
    # shape = input.shape
    # input = input.reshape(-1, input.shape[-1])
    # print(input.shape)
    # # 根据讨论交流的结果，一范数和无穷范数应该是更好的选择
    # temp_input1_norm = torch.norm(input, dim=-1) # 对列求范数 
    # # temp_input1_norm = torch.norm(input1, p=1, dim=0) # 对列求范数 
    # # temp_input1_norm, _ = torch.max(torch.abs(input), dim=-1) # 对列求无穷范数 
    # print(temp_input1_norm.shape)
    # sf_temp_input_norm = torch.softmax(temp_input1_norm, dim=0)
    # print(sf_temp_input_norm.shape)
    # score = sf_temp_input_norm / shape[-2]
    # print(score.shape)