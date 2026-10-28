import math
import torch
from infonce import info_nce
from _solution import info_nce as ref

torch.manual_seed(0)
N, D = 256, 64
a, b = torch.randn(N, D), torch.randn(N, D)

# 1) 与参考实现一致
assert torch.allclose(info_nce(a, b, 0.1), ref(a, b, 0.1), atol=1e-5), "和参考实现不一致"
print("[ok] 与参考实现一致")

# 2) 直觉测试
rand = info_nce(a, b, 1.0).item()
print(f"随机向量 loss = {rand:.3f}   (≈ log N = {math.log(N):.3f}，等于瞎猜)")
perfect = info_nce(a, a, 0.05).item()
print(f"完美配对 loss = {perfect:.4f}  (接近 0)")
print(f"放大模长不改变 loss: {info_nce(a*100, b, 0.1).item():.4f} == {info_nce(a, b, 0.1).item():.4f}")
for t in [0.01, 0.1, 1.0]:
    noisy = info_nce(a, a + 0.8 * torch.randn(N, D), t).item()
    print(f"τ={t:<5} 带噪配对 loss = {noisy:.3f}")
