#!/usr/bin/env python3
"""扫描 notes/ 生成 notes.json 目录清单, 供 index.html 构建侧边栏.

GitHub Pages 不提供目录列表, 所以目录树必须是静态文件.
被 .gitignore 排除的笔记(本地私密版)不写进清单, 否则公开站点会出现 404 链接;
它们由 index.html 的 PRIVATE 列表在本地探测后单独挂上。
新增/删除笔记后重新跑一次: python3 tools/build_manifest.py
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTES = ROOT / "notes"


def ignored() -> set[str]:
    """git 忽略的 .md 文件(相对 ROOT 的 posix 路径)."""
    paths = [p.relative_to(ROOT).as_posix() for p in NOTES.rglob("*.md")]
    if not paths:
        return set()
    r = subprocess.run(["git", "check-ignore", "--stdin"], cwd=ROOT,
                       input="\n".join(paths), capture_output=True, text=True)
    return set(r.stdout.split())


SKIP: set[str] = set()


def walk(d: Path) -> dict:
    rel = d.relative_to(ROOT).as_posix() + "/"
    files = sorted(p.relative_to(ROOT).as_posix() for p in d.glob("*.md")
                   if p.relative_to(ROOT).as_posix() not in SKIP)
    dirs = [walk(s) for s in sorted(d.iterdir())
            if s.is_dir() and not s.name.startswith(".")]
    return {"path": rel, "files": files, "dirs": dirs}


if __name__ == "__main__":
    SKIP = ignored()
    if SKIP:
        print("跳过(git 忽略的本地私密笔记):", ", ".join(sorted(SKIP)))
    tree = walk(NOTES)
    out = ROOT / "notes.json"
    out.write_text(json.dumps(tree, ensure_ascii=False, indent=1), encoding="utf-8")
    n = 0
    stack = [tree]
    while stack:
        node = stack.pop()
        n += len(node["files"])
        stack += node["dirs"]
    print(f"written: {out}  ({n} 篇笔记)")
