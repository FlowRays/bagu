#!/usr/bin/env python3
"""把本地私密笔记加密成 private/*.enc, 只有密文进 git.

为什么是客户端加密而不是"输密码放行":
  GitHub Pages 是纯静态托管, 没有服务端, 任何 JS 校验都能被绕过、受保护的文件也能被直接 GET。
  所以唯一有真实意义的做法是仓库里只存密文, 密码用来派生密钥在浏览器里解密。

  局限: 密码熵决定安全上限。拿到密文的人可以离线爆破, PBKDF2 迭代次数只能拖慢不能杜绝。
  想真正安全就换长密码（>=16 字符或一句话）。

算法: PBKDF2-HMAC-SHA256(密码, 随机 salt, ITER) -> 256bit key -> AES-256-GCM
      和 index.html 里的 WebCrypto 解密一一对应。

用法:
    python3 tools/encrypt_private.py            # 从 ~/.notes_pass 读密码
    NOTES_PASS=xxx python3 tools/encrypt_private.py
"""
import base64
import json
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import hashlib

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "private"
ITER = 600_000          # 和 index.html 里必须一致; OWASP 2023 对 PBKDF2-SHA256 的建议值

# 单篇私密笔记 -> 输出名
FILES = {
    "notes/interview.md":      "interview",
    "notes/oj-sspoffer-48.md": "oj-sspoffer-48",
}
# 整本加密的知识库: 目录下所有 .md 都加密
PRIVATE_DIRS = ["ideas"]


def read_password() -> str:
    if os.environ.get("NOTES_PASS"):
        return os.environ["NOTES_PASS"]
    f = Path.home() / ".notes_pass"
    if f.exists():
        return f.read_text(encoding="utf-8").strip()
    sys.exit("没有密码: 请设置 NOTES_PASS 或写入 ~/.notes_pass (chmod 600)")


def main() -> None:
    pw = read_password()
    if len(pw) < 12:
        print(f"! 警告: 密码只有 {len(pw)} 个字符, 离线爆破成本很低。建议换成 >=16 字符。")
    OUT.mkdir(exist_ok=True)
    # 全部文件共用一个 salt, 这样浏览器只需派生一次密钥就能解开全部,
    # 才谈得上"输一次密码就够"。每个文件仍然用独立的 IV(GCM 的安全要求)。
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, ITER, dklen=32)
    targets = dict(FILES)
    for d in PRIVATE_DIRS:                      # 整本加密的知识库
        for f in sorted((ROOT / d).rglob("*.md")) if (ROOT / d).is_dir() else []:
            rel = f.relative_to(ROOT).as_posix()
            targets[rel] = rel[:-3].replace("/", "__")

    files = {}
    for src, name in targets.items():
        p = ROOT / src
        if not p.exists():
            print(f"跳过(不存在): {src}")
            continue
        iv = os.urandom(12)
        ct = AESGCM(key).encrypt(iv, p.read_bytes(), None)
        (OUT / f"{name}.enc").write_text(json.dumps({
            "iv": base64.b64encode(iv).decode(),
            "ct": base64.b64encode(ct).decode(),
        }), encoding="utf-8")
        files[src] = f"private/{name}.enc"
        print(f"  {src}  ->  private/{name}.enc  ({len(ct)/1024:.1f} KB 密文)")
    (OUT / "index.json").write_text(json.dumps({
        "kdf": "PBKDF2-SHA256", "iter": ITER, "cipher": "AES-256-GCM",
        "salt": base64.b64encode(salt).decode(),
        "files": files,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written: {OUT/'index.json'}  ({len(files)} 篇, 共用 salt)")


if __name__ == "__main__":
    main()
