import torch
import numpy as np

# return sparse
def get_sparse_input(input, gather_index):
    gathered_input = get_selected_indices(input, gather_index)
    # sparse_input = denseToSparse(gathered_input)
    # return sparse_input
    return gathered_input # 这里假设后续用to sparse来替代

def get_selected_indices(ori_input, indices):
    # print('input shape : ', input.shape)
    # print('indices shape : ', indices.shape)
    input = ori_input.clone()
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
        input1.reshape(-1, input1.shape[-1])
        # 根据讨论交流的结果，一范数和无穷范数应该是更好的选择
        # temp_input1_norm = torch.norm(input1, dim=0) # 对列求范数 
        # temp_input1_norm = torch.norm(input1, p=1, dim=0) # 对列求范数 
        temp_input1_norm = torch.max(torch.abs(input1), dim=1) # 对列求无穷范数 
        # sf_temp_input1_norm = torch.softmax(temp_input1_norm, dim=0)
        if input2 is not None:
            shape2 = input2.shape
            input2.reshape(-1, input2.shape[-1])
            temp_input2_norm = torch.max(torch.abs(input2), dim=1)# 对列求范数   
            # sf_temp_input2_norm = torch.softmax(temp_input2_norm, dim=0)
            print('shape1: ',shape1)
            print('shape2: ',shape2)
            print('temp_input1_norm shape : ',temp_input1_norm.shape)
            print('temp_input2_norm shape : ',temp_input2_norm.shape)
            # print('sf_temp_input1_norm shape : ',sf_temp_input1_norm.shape)
            # print('sf_temp_input2_norm shape : ',sf_temp_input2_norm.shape)
            # score = sf_temp_input1_norm / shape1[-2] + sf_temp_input2_norm / shape2[-2] # （加权）
            score = temp_input1_norm / shape1[-2] + temp_input2_norm / shape2[-2] # （加权）
        else:
            # score = sf_temp_input1_norm / shape1[-2]
            score = temp_input1_norm / shape1[-2]
        # 这里的index是
        gather_index = torch.argsort(score, descending=True)[..., :kept_feature_size]
        # gather_index = torch.argsort(score, descending=True)[kept_feature_size:]
        # print('gather_index shape : ',gather_index.shape)
        return gather_index
    elif sparse_mode == 'rand': # randAD
        full_indices = torch.randperm(input1.size()[-1]).to(input1.device)
        gather_index = full_indices[kept_feature_size:]
        # print('gather_index shape : ',gather_index.shape)
        return gather_index

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
import torch.nn.functional as F
# if __name__ == "__main__":
    # if sparse_mode == 'norm':
    #     # print('input shape, batch size, feature len, kept: ',input.shape, batch_size, feature_len, kept_feature_size)
    #     input_flatted1 = input1.reshape(-1, input1.shape[-1])
    #     temp_input1_norm = torch.norm(input_flatted1, dim=0) # 对列求范数 
    #     sf_temp_input1_norm = torch.softmax(temp_input1_norm, dim=0)
    #     if input2 is not None:
    #         input_flatted2 = input2.reshape(-1, input2.shape[-1])
    #         temp_input2_norm = torch.norm(input_flatted2, dim=0) # 对列求范数   
    #         sf_temp_input2_norm = torch.softmax(temp_input2_norm, dim=0)
    #         score = sf_temp_input1_norm / input1.shape[-2] + sf_temp_input2_norm / input2.shape[-2] # （加权）
    #     else:
    #         score = sf_temp_input1_norm / input1.shape[-2]
    #     # 这里的index是
    #     gather_index = torch.argsort(score, descending=True)[..., :kept_feature_size]

