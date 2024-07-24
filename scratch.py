import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # 计算词向量元素两两分组以后，每组元素对应的旋转角度 
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    print("freqs = ", freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

if __name__ == '__main__':
    # self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
    dim = 4096
    n_heads = 32
    max_seq_len = 128
    freqs_cis = precompute_freqs_cis(dim // n_heads, max_seq_len * 2)
    print(torch.arange(0, dim // n_heads, 2))
    print(torch.arange(max_seq_len * 2))
    print(freqs_cis.shape)
    print(freqs_cis)
