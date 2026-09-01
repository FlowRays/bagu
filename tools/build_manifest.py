#!/usr/bin/env python3
"""扫描 notes/ 生成 notes.json 目录清单, 供 index.html 构建侧边栏.

GitHub Pages 不提供目录列表, 所以目录树必须是静态文件.
新增/删除笔记后重新跑一次: python3 tools/build_manifest.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTES = ROOT / "notes"


def walk(d: Path) -> dict:
    rel = d.relative_to(ROOT).as_posix() + "/"
    files = sorted(p.relative_to(ROOT).as_posix() for p in d.glob("*.md"))
    dirs = [walk(s) for s in sorted(d.iterdir())
            if s.is_dir() and not s.name.startswith(".")]
    return {"path": rel, "files": files, "dirs": dirs}


if __name__ == "__main__":
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
