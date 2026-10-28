# 参考答案——先自己写，再来看
import torch
import torch.nn.functional as F


def info_nce(z1, z2, temperature=0.1):
    z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(len(z1), device=z1.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
