# 环境

本机（lambda1）有 8 张 RTX 6000 Ada（48 GB），今天的内容用单卡就够了，不需要去集群排队。

```bash
cd /storage/home/yz979/zheyuan/learn
source multimodal/env.sh     # 激活 uv 环境 + 缓存放在 /storage + 默认用 1 号卡
```

- 依赖见 [pyproject.toml](../pyproject.toml)，用 `uv sync` 安装（在 `multimodal/` 目录下执行）；torch 2.7.1 + cu128
- [env.sh](../env.sh)：设置 `HF_HOME`、`TORCH_HOME` 等环境变量，把缓存都放到 `/storage` 下（根分区已满，不能写）
- [prefetch.py](prefetch.py)：预下载 CIFAR-10（走 HuggingFace；torchvision 的官方源只有 60 KB/s）和 CLIP ViT-B/32 权重
- 换卡：`CUDA_VISIBLE_DEVICES=3 python ...`，先用 `nvidia-smi` 看哪张卡空闲

## 站点

笔记站点的仓库就是 `/storage/home/yz979/zheyuan/learn`（GitHub `FlowRays/bagu`，GitHub Pages）。
新增或删除文件后运行 `python3 tools/build_manifest.py` 更新目录；发布流程见仓库根目录的 `tools/publish.sh`。
