# SLEGPM: 光谱引导的高性能子图匹配引擎

[![Rust Version](https://img.shields.io/badge/Rust-1.74%2B-orange)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#) [![CI](https://img.shields.io/badge/CI-GitHub%20Actions-green)](#)

SLEGPM（Spectral-Led Graph Pattern Matching）是一个以光谱特征与 Weisfeiler–Lehman 约简为核心的高性能子图匹配框架，通过离线/在线双阶段管线、缓存复用与并行化候选筛选在大规模图数据中快速定位复杂模式。它面向知识图谱理解、生物网络分析与安全情报等需要高准确率子图检索的团队，提供可扩展的 Rust 工程化实现与完整的测试基线。

## 安装指南

1. 安装 **Rust 1.74+**（推荐 `rustup` 管理工具）。
2. 克隆仓库：`git clone https://example.com/slegpm.git && cd slegpm`。
3. （可选）为 Python 数据脚本准备环境：`python -m venv .venv && .venv/Scripts/activate`，随后 `pip install -r datasets/requirements.txt`。
4. 准备数据集结构 `datasets/<名称>/data_graph.json`，如原始为 gpickle，可运行 `python datasets/preprocess.py` 自动转换。
5. 构建与安装：`cargo build --release`。
6. 设置日志级别（可选）：`set RUST_LOG=info` 或 `debug`。

## 使用示例

```powershell
# 针对全部数据集执行离线与在线匹配
cargo run --release

# 指定单个数据集，例如 yeast
cargo run --release -- yeast

# 运行基准测试评估光谱流水线
cargo bench

# 仅离线预处理（若未来扩展 CLI 参数）
cargo run --release -- --offline-only
```

## 功能亮点

- **双阶段流水线**：离线缓存 Laplacian 与光谱特征，在线阶段快速筛选模式。
- **多重候选过滤**：指纹比对、光谱距离、WL 迭代与 VF2 精确匹配层层把关。
- **并行化执行**：借助 Rayon 对候选子图评估与匹配进行数据并行。
- **自动模式采样**：在缺少样本时自动生成连通子图，降低准备成本。
- **可扩展缓存体系**：依据路径与参数哈希组织的持久缓存，支持跨运行复用。
- **测试与基准齐备**：包含端到端集成测试与 Criterion 性能基准。

## 依赖

- 核心：`anyhow`、`blake3`、`env_logger`、`log`、`indexmap`、`nalgebra`、`petgraph`、`once_cell`、`rand`、`rand_xoshiro`、`rayon`、`serde`、`serde_json`、`vf2`。
- 开发/测试：`approx`、`tempfile`、`criterion`（基准测试）。
- Python 辅助脚本：`networkx`、`matplotlib` 等（用于数据转换与可视化）。

## 测试

- 全量测试：`cargo test`
- 指定集成测试：`cargo test datasets_workflow`
- 基准评估：`cargo bench`

## 模块复杂度分析

| 模块                                               | 估计圈复杂度/等级 | 说明与建议                                                                                               |
| -------------------------------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------- |
| `src/main.rs`                                    | ≈ 12（中）       | CLI 流程控制涵盖参数解析、离线/在线循环与清理，逻辑清晰但函数较长，可拆分为独立命令处理函数。            |
| `src/lib.rs`                                     | ≈ 2（低）        | 主要负责模块导出，无复杂逻辑，无需调整。                                                                 |
| `src/graph/mod.rs`                               | ≈ 25（高）       | 集中处理 JSON 读写、子图构建、属性拆装，分支与错误路径多，建议拆分成更小的子模块并补充单元测试。         |
| `src/pipeline/preprocess.rs`                     | ≈ 18（中高）     | 缓存懒加载与错误传播较多，建议为 `LazyValue` 添加文档并考虑提取缓存策略到独立类型。                    |
| `src/pipeline/workflow.rs`                       | ≈ 28（高）       | 并行候选评估、统计累积与去重逻辑复杂，是性能关键路径，建议增加注释并探索将 `evaluate_candidate` 拆分。 |
| `src/cache.rs`                                   | ≈ 15（中）       | 文件系统操作与 JSON 序列化路径较长，建议补充缓存命中率统计与异常恢复策略。                               |
| `src/sampling.rs`                                | ≈ 17（中）       | 并行采样与去重逻辑合理，注意 RNG 种子管理；可通过配置对象减少硬编码。                                    |
| `tests/datasets_workflow.rs`                     | ≈ 20（中高）     | 集成测试覆盖完整流程，但断言与辅助函数较多，可拆分子测试或引入测试构建器。                               |
| `benches/spectral_pipeline.rs`                   | ≈ 8（低）        | 基准脚本结构简单，无需调整。                                                                             |
| `datasets/preprocess.py` & `datasets/utils.py` | ≈ 10/12（中）    | 数据转换脚本逻辑直观，可补充 CLI 参数与错误处理以增强健壮性。                                            |

## 在线阶段复杂度分析

记号说明：`n` 为目标图节点数、`m` 为目标图边数、`Δ` 为目标图平均度、`k` 为模式图节点数、`a` 为锚点数量（`AnchorSelector` 产出）、`c` 为进入评估的候选子图数、`s` 为光谱维度（`spectral_dim`），`t` 为 `wl_iterations`。

| 阶段 | 关键模块/函数 | 理论时间复杂度 | 说明 |
| --- | --- | --- | --- |
| 锚点选择与候选过滤 | `pipeline::candidate::{AnchorSelector, CandidateGenerator}` | `O(k log k + a · (n + m))` | 锚点评分排序为 `O(k log k)`；每个锚点在目标图中扫描节点并检查标签、权重与度数（使用 Rayon 并行），顺序上界为 `a · (n + m)`。 |
| 子图抽取 | `pipeline::candidate::SubgraphExtractor` | `O(c · min(2k, n) · Δ)` | 对每个候选锚点执行半径 `⌈log₂ k⌉` 的 BFS（限制节点数为 `2k`），再构造诱导子图。 |
| 指纹与光谱筛选 | `pipeline::workflow::evaluate_candidate` 中的指纹匹配、`GraphPreprocessor::from_instance`、`spectral_distance` | `O(c · (k + k² + T_谱(k, s)))` | 指纹计数与度多重集比较为 `O(k)`；归一化拉普拉斯与光谱向量构造在最坏情况下对应密集特征分解 `T_谱(k, s) = O(k³)`，若采用稀疏近似可降至 `O(k · s²)`。 |
| WL 细化 | `wl::weisfeiler_lehman_isomorphic` | `O(c · k · Δ · t)` | 每次迭代遍历节点与邻接列表，若颜色桶冲突会提前终止。 |
| VF2 精确匹配 | `petgraph::algo::is_isomorphic_matching` | `O(c · T_VF2(k))` | VF2 最坏为指数级（约 `O(k!)`），实际由候选规模与过滤强度决定；在 WL 失败时立即跳过 VF2。 |

将上述阶段合并，可给出在线阶段的顺序上界：

```
O(k log k + a · (n + m) + c · [k³ + k · Δ · t + T_VF2(k)])
```

其中 `k³` 来自光谱分解的最坏复杂度，若使用稀疏光谱求解则可替换为 `k · s²`。工程上通过加强候选过滤（降低 `c`）、控制光谱维度与 WL 迭代次数、缓存指纹结果等手段，可有效缩小在线阶段的常数因子。

## 优化建议

- 拆分 `workflow.rs` 中的核心逻辑，引入策略模式或管线描述，降低圈复杂度并提升可测试性。
- 对图读写模块实行单元测试覆盖，确保各种属性与定向/无向图场景行为稳定。
- 扩展 CLI：支持仅离线/仅在线、参数化光谱维度、并行线程数等选项。
- 引入持续集成流水线，自动执行 `fmt`/`clippy`/`test`/`bench`，并上传基准趋势。
- 强化文档：为数据集准备、缓存机制与扩展接口撰写专题教程，降低上手门槛。
- 探索 GPU/多节点方案或更高阶图嵌入，以优化超大规模图匹配性能。
