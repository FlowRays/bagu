#!/usr/bin/env python3
"""渲染检查: 用真实浏览器跑一遍全站, 查 KaTeX 报错 / 坏链 / 横向溢出.

用法: python3 tools/check_site.py [--port 8917]

为什么要真浏览器: 标题 id 由 KaTeX 渲染后的 textContent 生成, 标题里含公式时
id 不可预测(例如 `## 1. 两个下标 $v$ 和 $k$` 会变成 ...-vvv-和-kkk), 只能从
页面里读。链接锚点因此必须对着页面实际 id 校验, 猜不出来。

需要 playwright(带 chromium)。退出码非 0 表示有问题。
"""
import asyncio
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASES = [("notes", "bagu.md"), ("embodied", "embodied.md"),
         ("papers", "papers.md"), ("ideas", "ideas.md")]
# 本地私密 / 加密后才进仓库的文件, 不参与检查
SKIP = {"interview.md", "oj-sspoffer-48.md", "01-game-directions.md", "02-reading.md"}


def collect():
    docs, roots = [], set()
    for d, root in BASES:
        if (ROOT / d).is_dir():
            docs += sorted(str(p.relative_to(ROOT)) for p in (ROOT / d).rglob("*.md"))
        if (ROOT / root).exists():
            docs.append(root)
            roots.add(root)
    return [d for d in docs if Path(d).name not in SKIP], roots


def page_title(doc):
    return re.sub(r"^\d+-", "", Path(doc).name[:-3]) + " · 八股笔记"


def file_headings(doc):
    """文件里 code fence 之外的标题数, 用来交叉校验页面渲染是否漏了标题."""
    n, infence = 0, False
    for line in (ROOT / doc).read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            infence = not infence
        elif not infence and re.match(r"#{1,4}\s", line):
            n += 1
    return n


async def render(docs, roots, port):
    from playwright.async_api import async_playwright
    ids, bad = {}, 0
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        pg = await browser.new_page(viewport={"width": 1400, "height": 900})
        for doc in docs:
            # index.html 是 hash 路由, 只改 fragment 不重新加载, 必须 reload 并等
            # document.title 变成本篇, 否则拿到的是上一篇的 id
            await pg.goto(f"http://localhost:{port}/#{doc}")
            await pg.reload()
            await pg.wait_for_function("t => document.title === t",
                                       arg=page_title(doc), timeout=20000)
            ids[doc] = await pg.eval_on_selector_all(
                "#content h1,#content h2,#content h3,#content h4", "e=>e.map(x=>x.id)")
            errs = await pg.eval_on_selector_all("#content .err", "e=>e.map(x=>x.textContent)")
            if errs:
                bad += 1
                print(f"[KATEX]  {doc}")
                for e in errs[:5]:
                    print(f"         {e.strip()[:160]}")
            if await pg.evaluate('()=>{const c=document.getElementById("content");'
                                 'return c.scrollWidth>c.clientWidth}'):
                bad += 1
                print(f"[OVERFLOW] {doc}  内容横向溢出, 检查宽表格/长公式/代码块")
            # 根目录那几篇的 h1 由 viewer 注入, 数不上, 跳过交叉校验
            if doc not in roots and file_headings(doc) != len(ids[doc]):
                bad += 1
                print(f"[HEADINGS] {doc}: 文件 {file_headings(doc)} vs 页面 {len(ids[doc])}")
        await browser.close()
    return ids, bad


def check_links(ids):
    n = bad = 0
    for doc in ids:
        f = ROOT / doc
        for m in re.finditer(r"\[[^\]]*\]\(([^)\s]+)\)", f.read_text(encoding="utf-8")):
            href = m.group(1)
            if re.match(r"^(https?:|mailto:)", href):
                continue
            n += 1
            path, _, anchor = href.partition("#")
            target = (f.parent / path).resolve() if path else f
            if not target.exists():
                bad += 1
                print(f"[FILE]   {doc} -> {href}")
                continue
            if not anchor:
                continue
            have = ids.get(target.relative_to(ROOT).as_posix())
            if have is None:
                bad += 1
                print(f"[NOIDS]  {doc} -> {href}  (目标不在检查范围内)")
            elif anchor not in have:
                bad += 1
                print(f"[ANCHOR] {doc} -> {href}")
                # 标题重命名后最常见, 直接把页面里的真实 id 报出来
                cand = [i for i in have if i.split("-")[0] == anchor.split("-")[0]]
                if cand:
                    print(f"         实际是: {cand[0]}")
    return n, bad


def main():
    args = sys.argv[1:]
    port = 8917
    if "--port" in args:
        i = args.index("--port")
        port = int(args[i + 1])
        del args[i:i + 2]
    docs, roots = collect()
    if not docs:
        sys.exit("没找到任何笔记, 确认在仓库根目录下运行")

    server = subprocess.Popen([sys.executable, "-m", "http.server", str(port)],
                              cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://localhost:{port}/index.html", timeout=1)
                break
            except Exception:
                time.sleep(0.2)
        else:
            sys.exit(f"本地 http server 起不来 (端口 {port} 可能被占用, 换 --port)")
        ids, bad = asyncio.run(render(docs, roots, port))
    finally:
        server.terminate()

    n, link_bad = check_links(ids)
    total = bad + link_bad
    print(f"\n{len(docs)} 篇 / 标题 {sum(len(v) for v in ids.values())} 个 / "
          f"内部链接 {n} 个 —— 问题 {total} 处")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
