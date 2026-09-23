# 核心代码导读

本目录提供数据构建、检索、工具路由、评测与前端交互的精选实现，适合按主题审阅。

## 实现索引

| 关注点 | 实现入口 | 工程贡献 |
| --- | --- | --- |
| 评测数据集构建 | `evals/datasets/build_industry_bench_100.py`、`industry_bench_100.schema.json` | Schema、证据定位、类别配额与分层划分 |
| 文档范围与混合检索 | `backend/app/service/fast_research.py` | 文档别名路由、词法排序、Dense/Lexical 融合及多文档候选分配 |
| 共享工具协议 | `backend/app/tool_routing_contract.py`、`evals/toolbench_run.py` | Gold 隐藏的 Prompt、严格解析、最小充分工具集及分项评分 |
| 执行与安全边界 | `backend/app/service/react_controller.py`、`action_gate.py`、`tool_executor.py` | 有界调用、授权范围、参数与执行状态管理 |
| 指标与统计 | `evals/metrics/` | 检索、引用、拒答与工具指标，以及配对 bootstrap |
| 冻结原子契约 | `evals/claim_atomic_contract.py`、`freeze_claim_atomic_decomposition.py` | 固定字段、独立可判定检查、审核状态及不可变哈希 |
| 字段校验适配 | `evals/model_field_claim_verifier_bench.py` | 无语义请求 ID、结构化 wire contract、确定性聚合和分层评分 |
| 前端证据与安全 | `frontend/src/pages/chat/component/research-detail/`、`frontend/src/components/markdown/` | 来源片段展开、键盘访问、HTML 与 URI 过滤 |

## 离线测试

在本目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_*.py' -v
```

**16/16 通过。** 测试覆盖检索、引用、拒答和工具指标，配对统计，工具协议解析、Gold 与 Prompt 隔离，以及参数和权限分项评分。运行需要 Python 3.9+ 标准库。

## 运行依赖

- 数据集构建器使用 `jsonschema`、`pypdf`，完整重建需要对应的原始报告和候选审核文件。
- 后端模块按主题摘选；运行完整服务需要完整应用工程及模型、数据库配置。
- 前端实现与安全测试按代码阅读材料提供，执行需完整 Node 工程和依赖。

源码保持项目实现，哈希索引见 `../02_成果与验证/source-manifest.json`。建议从数据 Schema、构建器和评分函数开始，再阅读检索策略与工具路由契约。

数据规范版本：`annotation_guidelines.md` 为早期 20 题基准的标注规则；当前 IndustryBench-100 的类别配额与 40/30/30 划分由 `build_industry_bench_100.py` 定义。
