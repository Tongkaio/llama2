import torch

# 创建一个原始张量
x = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
print(f"Original Tensor:\n{x}\n")
print(x.shape)

# 通过 expand 方法扩展张量
expanded_x = x[:,:,None,:].expand(1,2,4,3).reshape(1,2*4,3)
print(f"Expanded Tensor:\n{expanded_x}\n")

# 修改原始张量
x[0][0][0] = 100
print(f"Modified Original Tensor:\n{x}\n")
print(f"Expanded Tensor after modifying the original tensor:\n{expanded_x}\n")
