# 成果与验证

| 材料 | 内容 |
| --- | --- |
| `数据工程导读.md` | 数据建设、证据追踪与质量控制 |
| `成果与评测口径.md` | 实验结果、指标定义与数据设置 |
| `public-evaluation.json` | 指标摘要及来源哈希 |
| `dataset-summary.json` | 数据类别与分层划分统计 |
| `retrieval-audit.json` | 20 题的可回答性、证据计数和首条相关判定 |
| `routing-audit.json` | 200 条路由的分项评分 |
| `source-manifest.json` | 核心源码的相对路径、字节数与 SHA-256 |
| `verify_results.py` | 文件完整性检查与统计复算 |
| `项目介绍.tex` | 一页项目介绍的 LaTeX 源文件 |

运行 `python3 verify_results.py`，可核验数据配额、交付文件哈希及检索/路由统计。脚本仅依赖 Python 标准库，可从任意工作目录执行。
