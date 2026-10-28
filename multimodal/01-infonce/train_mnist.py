"""MNIST 上的 SimCLR：不用标签学表征，再用 linear probe 评估。

用法: python train_mnist.py [--temperature 0.2] [--batch-size 512] [--epochs 10] [--use-solution]
"""
import argparse, math, time
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision

p = argparse.ArgumentParser()
p.add_argument("--temperature", type=float, default=0.2)
p.add_argument("--batch-size", type=int, default=512)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--use-solution", action="store_true")
p.add_argument("--tsne", action="store_true", help="保存 t-SNE 图")
args = p.parse_args()
if args.use_solution:
    from _solution import info_nce
else:
    from infonce import info_nce

torch.manual_seed(0)
dev = "cuda"
root = "/storage/home/yz979/zheyuan/learn/data"
tr = torchvision.datasets.MNIST(root, train=True, download=True)
te = torchvision.datasets.MNIST(root, train=False, download=True)
Xtr = (tr.data.float() / 255).unsqueeze(1).to(dev); Ytr = tr.targets.to(dev)
Xte = (te.data.float() / 255).unsqueeze(1).to(dev); Yte = te.targets.to(dev)


def augment(x):
    """逐样本随机仿射（旋转/缩放/平移）+ 随机遮挡，在 GPU 上批量完成。"""
    n = len(x)
    ang = (torch.rand(n, device=dev) - 0.5) * math.radians(50)
    sc = 0.75 + 0.4 * torch.rand(n, device=dev)
    tx, ty = (torch.rand(2, n, device=dev) - 0.5) * 0.4
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)
    # 随机遮挡一个 10x10 的块
    i, j = torch.randint(0, 18, (2, n), device=dev)
    yy, xx = torch.meshgrid(torch.arange(28, device=dev), torch.arange(28, device=dev), indexing="ij")
    mask = ((yy[None] >= i[:, None, None]) & (yy[None] < i[:, None, None] + 10) &
            (xx[None] >= j[:, None, None]) & (xx[None] < j[:, None, None] + 10))
    keep = torch.rand(n, device=dev) < 0.5
    return x * ~(mask & keep[:, None, None]).unsqueeze(1)


class Encoder(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        c = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
        self.net = nn.Sequential(c(1, 64), c(64, 64), nn.MaxPool2d(2), c(64, 128), c(128, 128),
                                 nn.MaxPool2d(2), c(128, dim), nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


enc = Encoder().to(dev)
proj = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128)).to(dev)  # projection head


@torch.no_grad()
def features(model, X):
    model.eval()
    out = torch.cat([model(X[i:i + 4096]) for i in range(0, len(X), 4096)])
    model.train()
    return out


def linear_probe(Ftr, Fte, epochs=100):
    """冻结特征，只训练一个线性层，测试集准确率衡量表征的线性可分性。"""
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
    clf = nn.Linear(Ftr.shape[1], 10).to(dev)
    opt = torch.optim.Adam(clf.parameters(), 1e-2)
    for _ in range(epochs):
        for idx in torch.randperm(len(Ftr), device=dev).split(4096):
            opt.zero_grad(); F.cross_entropy(clf(Ftr[idx]), Ytr[idx]).backward(); opt.step()
    return (clf(Fte).argmax(1) == Yte).float().mean().item()


acc0 = linear_probe(features(enc, Xtr), features(enc, Xte))
print(f"随机初始化编码器 probe acc = {acc0:.4f}")

opt = torch.optim.Adam([*enc.parameters(), *proj.parameters()], 1e-3)
t0 = time.time()
for ep in range(args.epochs):
    tot = 0.0
    batches = torch.randperm(len(Xtr), device=dev).split(args.batch_size)
    for idx in batches:
        x = Xtr[idx]
        loss = info_nce(proj(enc(augment(x))), proj(enc(augment(x))), args.temperature)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    print(f"epoch {ep+1:2d}  loss {tot/len(batches):.4f}  (log N = {math.log(args.batch_size):.2f})  {time.time()-t0:.0f}s")

Ftr, Fte = features(enc, Xtr), features(enc, Xte)
acc = linear_probe(Ftr, Fte)
print(f"SimCLR 编码器 probe acc = {acc:.4f}   (τ={args.temperature}, batch={args.batch_size})")

if args.tsne:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    sel = torch.randperm(len(Fte))[:3000]
    for name, model in [("random", Encoder().to(dev)), ("simclr", enc)]:
        emb = TSNE(init="pca").fit_transform(features(model, Xte[sel.to(dev)]).cpu().numpy())
        plt.figure(figsize=(5, 5)); plt.scatter(*emb.T, c=Yte[sel.to(dev)].cpu(), cmap="tab10", s=3)
        plt.title(name); plt.axis("off"); plt.savefig(f"tsne_{name}.png", dpi=120); plt.close()
    print("saved tsne_random.png / tsne_simclr.png")
