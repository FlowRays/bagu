#!/usr/bin/env python3
"""扫描各知识库目录, 生成 notes.json, 供 index.html 构建侧边栏.

站点现在有三个知识库(base):
  - llm      : notes/    + 大纲 bagu.md       (LLM/VLM 八股)
  - embodied : embodied/ + 大纲 embodied.md   (VLA/WAM 具身八股)
  - papers   : papers/   + 索引 papers.md     (按专题读论文的笔记)

GitHub Pages 不提供目录列表, 所以目录树必须是静态文件.
被 .gitignore 排除的笔记(本地私密版)不写进清单, 否则公开站点会出现 404 链接;
它们由 index.html 的 PRIVATE 列表在本地探测后单独挂上。
新增/删除笔记后重新跑一次: python3 tools/build_manifest.py
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASES = [
    {"id": "llm",      "title": "LLM/VLM 八股笔记", "dir": "notes",    "root": "bagu.md"},
    {"id": "embodied", "title": "VLA/WAM 八股笔记", "dir": "embodied", "root": "embodied.md"},
    {"id": "papers",   "title": "论文笔记",         "dir": "papers",   "root": "papers.md"},
]


def ignored(d: Path) -> set[str]:
    """git 忽略的 .md 文件(相对 ROOT 的 posix 路径)."""
    paths = [p.relative_to(ROOT).as_posix() for p in d.rglob("*.md")]
    if not paths:
        return set()
    r = subprocess.run(["git", "check-ignore", "--stdin"], cwd=ROOT,
                       input="\n".join(paths), capture_output=True, text=True)
    return set(r.stdout.split())


def walk(d: Path, skip: set[str]) -> dict:
    rel = d.relative_to(ROOT).as_posix() + "/"
    files = sorted(p.relative_to(ROOT).as_posix() for p in d.glob("*.md")
                   if p.relative_to(ROOT).as_posix() not in skip)
    dirs = [walk(s, skip) for s in sorted(d.iterdir())
            if s.is_dir() and not s.name.startswith(".")]
    return {"path": rel, "files": files, "dirs": dirs}


def count(node: dict) -> int:
    n, stack = 0, [node]
    while stack:
        cur = stack.pop()
        n += len(cur["files"])
        stack += cur["dirs"]
    return n


if __name__ == "__main__":
    out = {"bases": []}
    for b in BASES:
        d = ROOT / b["dir"]
        if not d.is_dir():
            print(f"跳过(目录不存在): {b['dir']}/")
            continue
        skip = ignored(d)
        if skip:
            print(f"跳过(git 忽略的本地私密笔记): {', '.join(sorted(skip))}")
        tree = walk(d, skip)
        out["bases"].append({**{k: b[k] for k in ("id", "title", "root")}, "tree": tree})
        print(f"  {b['id']:<9} {b['dir']}/  {count(tree)} 篇")
    p = ROOT / "notes.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written: {p}  ({sum(count(x['tree']) for x in out['bases'])} 篇笔记)")
