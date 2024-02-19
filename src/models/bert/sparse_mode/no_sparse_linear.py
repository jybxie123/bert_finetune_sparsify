import torch
import torch.nn as nn
import torch.nn.functional as F

class OurNoSparseLinear(nn.Linear):
    # linear_idx is the index of the linear layer
    # step_idx is the index of iteration 
    # act_type: relu or silu
    # 我决定不为lienar_idx提供默认值，这样能避免出错。
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, linear_idx = None, act_type = None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        # print('OurNoSparseLinear step idx = 0')
        self.step_idx = 0
        self.linear_idx = linear_idx
        self.act_type = act_type
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        layer_output = noSparseLinear.apply(input, self.weight, self.bias)
        # cl.cal_zero_ratio(layer_output, self.linear_idx, self.step_idx, self.act_type)
        # print('Ourlinear step idx + 1')
        self.step_idx += 1
        return layer_output
        
    # 这个函数仅仅用作测试结构上是否已经完整，不需要在真实测试中使用。
    # def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = '', remove_duplicate: bool = True):
    #     # 在这里，我们将会重写 named_modules 的行为
    #     # 例如，我们可以过滤掉特定的层或者改变返回的名称

    #     modules = super().named_modules(memo, prefix)

    #     for name, module in modules:
    #         if module in self.__dict__.values():
    #             # 如果是，添加自定义前缀或更改名称
    #             name = 'our_linear_' + name if name else 'our_linear'
    #         yield name, module

# # self difination func: cal the zero ratio

class noSparseLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.matmul(weight.transpose(-2, -1))
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(-2,-1).matmul(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias
