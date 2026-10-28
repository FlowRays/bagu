import torch
import torch.nn.functional as F


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """对称 InfoNCE（CLIP 形式）。

    z1, z2: [N, D]，第 i 行互为正样本对；输入未归一化。
    返回标量 loss。
    """
    # TODO: 1) L2 归一化  2) 相似度矩阵 / temperature  3) labels  4) 两个方向的 CE 取平均
    raise NotImplementedError
