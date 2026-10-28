# 用法: source multimodal/env.sh   —— 所有缓存都放在 /storage 下，不写 / 和 home
export LEARN_ROOT=/storage/home/yz979/zheyuan/learn
export TMPDIR=/storage/home/yz979/tmp
export XDG_CACHE_HOME=$LEARN_ROOT/.cache
export UV_CACHE_DIR=/storage/home/yz979/.cache/uv
export HF_HOME=$LEARN_ROOT/.cache/hf
export TORCH_HOME=$LEARN_ROOT/.cache/torch
export MPLCONFIGDIR=$LEARN_ROOT/.cache/mpl
export PIP_CACHE_DIR=$LEARN_ROOT/.cache/pip
# 默认用一张空闲卡（1/3/5/7 空闲），需要时覆盖
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
source $LEARN_ROOT/multimodal/.venv/bin/activate 2>/dev/null || true
