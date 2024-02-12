# import torch

# # 希望将sparse过程进行封装，这样可以直接调用，但是效果不好，内存占用太大。
# class Sparse(object):
#     def __init__(self):
#         pass

# class RandSparse(Sparse):
#     def __init__(self, prune_ratio):
#         super().__init__()
#         self.prune_ratio = prune_ratio
#     def get_sparse_index(self):
#         return self.sparse_index
#     def set_sparse_index(self, sparse_index):
#         self.sparse_index = sparse_index
#     def cal_sparse_index(self, input1, input2=None):
#         if input2 is not None:
#             if input1.shape[-1] != input2.shape[-1]:
#                 print('input1 shape : ',input1.shape)
#                 print('input2 shape : ',input2.shape)
#                 raise ValueError('input1 and input2 shape is not supported')
#             if len(input2.shape) == 1:
#                 raise ValueError('input2 shape is not supported')
#         kept_feature_size = int(input1.shape[-1] * self.prune_ratio + 0.999)
#         if len(input1.shape) == 1:
#             raise ValueError('input1 shape is not supported')
#         full_indices = torch.randperm(input1.size()[-1]).to(input1.device)
#         gather_index = full_indices[kept_feature_size:]
#         # print('gather_index shape : ',gather_index.shape)
#         return gather_index
#     def cal_sparse_input(self, input):
#         gathered_input = get_selected_indices(input, self.sparse_index)
#         # sparse_input = denseToSparse(gathered_input)
#         # return sparse_input
#         return gathered_input # 这里假设后续用to sparse来替代
    
# class NormSparse(Sparse):
#     def __init__(self, prune_ratio):
#         super().__init__()
#         self.prune_ratio = prune_ratio
#         self.sparse_index = None

#     def get_sparse_index(self):
#         return self.sparse_index
#     def set_sparse_index(self, sparse_index):
#         self.sparse_index = sparse_index

#     def cal_sparse_index(self, input1, input2=None):
#         if input2 is not None:
#             if input1.shape[-1] != input2.shape[-1]:
#                 print('input1 shape : ',input1.shape)
#                 print('input2 shape : ',input2.shape)
#                 raise ValueError('input1 and input2 shape is not supported')
#             if len(input2.shape) == 1:
#                 raise ValueError('input2 shape is not supported')
#         kept_feature_size = int(input1.shape[-1] * self.prune_ratio + 0.999)
#         if len(input1.shape) == 1:
#             raise ValueError('input1 shape is not supported')
#         input_flatted1 = input1.reshape(-1, input1.shape[-1])
#         temp_input1_norm = torch.norm(input_flatted1, dim=0) # 对列求范数 
#         sf_temp_input1_norm = torch.softmax(temp_input1_norm, dim=0)
#         if input2 is not None:
#             input_flatted2 = input2.reshape(-1, input2.shape[-1])
#             temp_input2_norm = torch.norm(input_flatted2, dim=0) # 对列求范数   
#             sf_temp_input2_norm = torch.softmax(temp_input2_norm, dim=0)
#             score = sf_temp_input1_norm / input1.shape[-2] + sf_temp_input2_norm / input2.shape[-2] # （加权）
#         else:
#             score = sf_temp_input1_norm / input1.shape[-2]
#         gather_index = torch.argsort(score, descending=True)[..., :kept_feature_size]
#         self.sparse_index = gather_index
#         return gather_index
#     def cal_sparse_input(self, input):
#         gathered_input = get_selected_indices(input, self.sparse_index)
#         # sparse_input = denseToSparse(gathered_input)
#         # return sparse_input
#         return gathered_input # 这里假设后续用to sparse来替代

    
# class_dict = {
#     'norm': NormSparse(0.5),
#     'rand': RandSparse(0.5)
# }

# def get_selected_indices(input, indices):
#     # print('input shape : ', input.shape)
#     # print('indices shape : ', indices.shape)
#     if len(input.shape) == 1:
#         input[indices] = 0
#     elif len(input.shape) == 2:
#         input[:,indices] = 0
#     elif len(input.shape) == 3:
#         input[:,:,indices] = 0
#     elif len(input.shape) == 4:
#         input[:,:,:,indices] = 0
#     elif len(input.shape) == 5:
#         input[:,:,:,:,indices] = 0
#     else:
#         # print('input shape is : ', input.shape)
#         raise ValueError('input shape is not supported')
#     return input
