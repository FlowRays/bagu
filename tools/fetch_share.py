#!/usr/bin/env python3
"""抓取 ChatGPT 分享链接的对话全文, 输出 markdown.

用法: python3 tools/fetch_share.py <share_url> [-o out.md]
"""
import json
import re
import subprocess
import sys

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"


def fetch(url):
    html = subprocess.run(
        ["curl", "-sL", "--max-time", "60", "-A", UA, url],
        capture_output=True, text=True, check=True,
    ).stdout
    chunks = re.findall(r'\.enqueue\("((?:[^"\\]|\\.)*)"\)', html)
    if not chunks:
        sys.exit("页面里没有找到对话数据 (可能链接失效或被风控页拦截)")
    return "".join(json.loads('"' + c + '"') for c in chunks)


def decode_table(payload):
    """turbo-stream 引用表: 元素互相以下标引用, 还原成普通对象."""
    table = json.loads(payload.split("\n", 1)[0])
    memo = {}

    def deref(i):
        if not isinstance(i, int):
            return decode(i)
        if i < 0:
            return None
        if i not in memo:
            memo[i] = None
            memo[i] = decode(table[i])
        return memo[i]

    def decode(v):
        if isinstance(v, dict):
            return {str(deref(int(k[1:])) if k.startswith("_") else k): deref(val)
                    for k, val in v.items()}
        if isinstance(v, list):
            return [deref(x) for x in v]
        return v

    return deref(0)


def to_markdown(data, url):
    lines = [f"# {data.get('title', '未命名对话')}", "", f"> 来源: {url}", ""]
    for node in data["linear_conversation"]:
        msg = (node or {}).get("message")
        if not msg:
            continue
        role = msg.get("author", {}).get("role")
        content = msg.get("content", {})
        if role not in ("user", "assistant") or content.get("content_type") not in ("text", "multimodal_text"):
            continue
        parts = []
        for p in content.get("parts") or []:
            if isinstance(p, str) and p.strip():
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(f"[附件: {p.get('content_type', 'unknown')}]")
        if not parts:
            continue
        lines += [f"## {'User' if role == 'user' else 'Assistant'}", ""] + parts + [""]
    return "\n".join(lines)


def post_to_markdown(post, url):
    """chatgpt.com/s/t_... 格式: 单条消息的分享 (post + attachments)."""
    lines = [f"# {post.get('text') or post.get('og_title') or '分享消息'}", "", f"> 来源: {url}", ""]
    for att in post.get("attachments") or []:
        for m in att.get("messages") or []:
            role = (m.get("author") or {}).get("role", "?")
            for p in (m.get("content") or {}).get("parts") or []:
                if isinstance(p, str) and p.strip():
                    lines += [f"## {'User' if role == 'user' else 'Assistant'}", "", p, ""]
    return "\n".join(lines)


if __name__ == "__main__":
    args = sys.argv[1:]
    out = None
    if "-o" in args:
        i = args.index("-o")
        out = args[i + 1]
        del args[i:i + 2]
    if len(args) != 1:
        sys.exit(__doc__)
    url = args[0]
    loader = decode_table(fetch(url))["loaderData"]
    if "routes/share.$shareId.($action)" in loader:
        md = to_markdown(loader["routes/share.$shareId.($action)"]["serverResponse"]["data"], url)
    elif "routes/s.$postId" in loader:
        md = post_to_markdown(loader["routes/s.$postId"]["postWithProfile"]["post"], url)
    else:
        sys.exit(f"未知的分享页格式, loaderData keys: {list(loader)}")
    if out:
        open(out, "w", encoding="utf-8").write(md)
        print(f"written: {out} ({len(md)} chars)")
    else:
        print(md)
