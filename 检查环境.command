#!/bin/bash
# Finder double-click entry; all paths are resolved relative to this file.
set -u
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)" || exit 1
cd "$PROJECT_ROOT" || exit 1
export PATH="$HOME/.local/bin:$HOME/.docker/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    echo "缺少本项目的 Python 环境，请先按照 本地运行说明.md 重建 .venv。"
    if [[ -t 0 ]]; then read -r -p "按回车关闭窗口…"; fi
    exit 1
fi
"$PROJECT_ROOT/.venv/bin/python" -u "$PROJECT_ROOT/.launcher/launcher.py" --check "$@"
result=$?
if [[ -t 0 ]]; then
    echo
    read -r -p "按回车关闭窗口…"
fi
exit "$result"
