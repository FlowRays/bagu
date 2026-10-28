"""预下载后面模块用到的数据和权重，避免上课时等待。"""
import open_clip
from datasets import load_dataset
load_dataset("uoft-cs/cifar10")
print("cifar10 ok", flush=True)
open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
print("clip ok", flush=True)
