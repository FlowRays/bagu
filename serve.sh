#!/usr/bin/env bash
# 笔记站点: bash serve.sh [port]  (默认 8901)
# 本地访问: ssh -L 8901:localhost:8901 lambda2 然后打开 http://localhost:8901
cd "$(dirname "$0")"
exec python3 -m http.server "${1:-8901}"
