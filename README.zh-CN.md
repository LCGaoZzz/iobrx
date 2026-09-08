# iobrx

**保留 IOBRpy 的分析语义，让肿瘤微环境分析更快、更容易复现。**

[English](https://github.com/LCGaoZzz/iobrx/blob/main/README.md) · [教程与图集](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/README.md) · [兼容性说明](https://github.com/LCGaoZzz/iobrx/blob/main/docs/PORTABILITY.md)

iobrx 提供 pandas DataFrame 接口，通过 Rust、并行计算和 NumPy/pandas
向量化加速 IOBRpy。参考矩阵和基因签名继续使用 IOBRpy 的资源。

- 覆盖基因注释、Count 转 TPM、PCA/zscore/ssGSEA/联合签名评分，以及 CIBERSORT、EPIC、quanTIseq、MCP-counter、ESTIMATE。
- **不再强制要求 AVX-512**。已在没有 AVX-512 的 i9-13900KF 上实际运行；排序交给本机 NumPy 选择兼容实现。
- **12 本已执行的 Notebook**：11 个独立分析入口 + 1 个完整工作流，均含教程代码、结果和内嵌图；同时提供 PNG、PDF、SVG。
- 每张分析图经历初稿、第一轮版式调整、第二轮精修，采用白底、低饱和配色、细轴线和可编辑矢量文字。参见[逐图修改记录](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/FIGURE_REVIEW.md)。

![CIBERSORT 示例](https://raw.githubusercontent.com/LCGaoZzz/iobrx/main/tutorials/figures/07_cibersort.png)

## 安装与运行

已验证的环境为 **Linux x86-64 / WSL2、Python 3.11**。0.2.0 提供预编译
wheel，普通用户无需安装 Rust 或 C++ 编译器：

```bash
python -m pip install --only-binary=:all: iobrx==0.2.0
python -c "import iobrx; print(iobrx.backend_info())"
```

国内用户可在清华镜像同步后使用：

```bash
python -m pip install --only-binary=:all: -i https://pypi.tuna.tsinghua.edu.cn/simple iobrx==0.2.0
```

镜像尚未同步新版本时，请在第一条命令后添加 `--index-url https://pypi.org/simple`。
`--only-binary=:all:` 会在环境不支持时明确报错，避免意外触发源码编译。
数值依赖固定为已经验证的版本；已有环境存在版本冲突时，建议新建 Python 3.11 环境。

运行完整教程时，再获取对应版本的仓库与公开数据：

```bash
git clone --branch v0.2.0 https://github.com/LCGaoZzz/iobrx.git
cd iobrx
python -m pip install --only-binary=:all: "iobrx[tutorials,test]==0.2.0"
python -m jupyterlab tutorials
```

在 Omicos 环境中使用时，将上面的 `python` 换成该环境的 Python，并在
Jupyter 中选择同一环境的内核。仓库已附公开示例矩阵，安装依赖后可以离线运行教程。

也可直接使用固定版本容器：

```bash
docker run --rm ghcr.io/lcgaozzz/iobrx:0.2.0
docker run --rm -v "$PWD:/work" -w /work ghcr.io/lcgaozzz/iobrx:0.2.0 python analysis.py
```

容器内教程和数据位于 `/opt/iobrx/tutorials`，依赖已锁定，不会自动启动 Jupyter。
[GitHub Release](https://github.com/LCGaoZzz/iobrx/releases/tag/v0.2.0) 附 wheel、
源码包、SHA-256 校验值和用于严格复现的容器 digest。

```python
import numpy as np
import pandas as pd
import iobrx

iobrx.set_threads(8)
counts = pd.read_parquet("tutorials/data/eset_stad.parquet")
tpm = iobrx.count2tpm(counts, check_data=True, remove_version=True)
cib = iobrx.cibersort(tpm, perm=100, QN=False)
scores = iobrx.calculate_sig_score(
    np.log2(tpm + 1), "signature_collection", method="integration"
)
```

初次使用建议打开[完整工作流](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/12_complete_workflow.ipynb)，先理解不同方法
需要的输入尺度，再替换成自己的数据。

## 在 Omicos／Agent 中使用

[agent-harness](https://github.com/LCGaoZzz/iobrx/tree/main/agent-harness) 为 11 项分析提供统一的 JSON CLI、
可选的 stdio MCP 服务，以及可随 Omicos catalog 分发的 Agent／Skill。
运行前检查矩阵方向、声明的数据尺度和基因 ID；运行后保存参数、输入／输出
哈希、环境版本、实际后端和耗时。分析继续调用 iobrx 原有 API。

安装 iobrx 后，在仓库根目录运行：

```bash
python -m pip install ./agent-harness
iobrx-agent doctor
iobrx-agent run --request agent-harness/examples/signature_pca.json
```

详见 [Omicos 接入说明](https://github.com/LCGaoZzz/iobrx/blob/main/agent-harness/omicos/README.md)和
[实测记录](https://github.com/LCGaoZzz/iobrx/blob/main/agent-harness/VALIDATION.md)。配套 harness 目前从本仓库安装；
它尚未作为独立包发布到 PyPI。Omicos 共享 catalog 上线需要另走 admin 仓库的 PR／发布流程。

## 每项分析大约需要多久？

以下为 **i9-13900KF、WSL2 Ubuntu、Omicos Python 3.11、请求 8 线程**的实测值。
每项分析使用独立进程，先记录首次调用，再重复 3 次；表中主要耗时为这 3 次的中位数。
不计读取数据、分析前的输入准备和绘图；完整工作流计入自己的 TPM 转换和全部分析步骤。
“首次”指新进程内的首次分析调用，操作系统文件缓存可能已热。

<!-- BENCHMARK_TABLE_START -->
| 分析 / 教程 | 输入：特征 × 样本 | 中位耗时 | 首次调用 | 与 IOBRpy 一致性 |
| --- | --- | ---: | ---: | --- |
| [基因注释与重复条目处理](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/01_gene_annotation.ipynb) | 60,483 × 10 | **16.7 ms** | 36.0 ms | 逐位一致 |
| [Count 转 TPM](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/02_counts_to_tpm.ipynb) | 60,483 × 10 | **67.5 ms** | 111.6 ms | 逐位一致 |
| [PCA 基因签名评分](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/03_signature_pca.ipynb) | 872 × 348 | **244.1 ms** | 600.4 ms | 逐位一致 |
| [zscore 方法基因签名评分](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/04_signature_zscore.ipynb) | 872 × 348 | **78.6 ms** | 487.8 ms | 逐位一致 |
| [ssGSEA 基因集富集](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/05_signature_ssgsea.ipynb) | 872 × 348 | **86.7 ms** | 495.9 ms | 逐位一致 |
| [PCA、zscore 与 ssGSEA 联合评分](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/06_signature_integration.ipynb) | 872 × 348 | **336.0 ms** | 817.9 ms | 逐位一致 |
| [CIBERSORT 免疫细胞组成](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/07_cibersort.ipynb) | 48,058 × 10 | **26.37 s** | 26.99 s | 除 P-value 外逐位一致 |
| [EPIC 细胞比例与 mRNA 比例](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/08_epic.ipynb) | 48,058 × 10 | **6.8 ms** | 301.9 ms | 逐位一致 |
| [quanTIseq 免疫细胞反卷积](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/09_quantiseq.ipynb) | 48,058 × 10 | **40.9 ms** | 475.2 ms | 逐位一致 |
| [MCP-counter 细胞群丰度评分](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/10_mcpcounter.ipynb) | 48,058 × 10 | **1.1 ms** | 10.0 ms | 逐位一致 |
| [ESTIMATE 基质与免疫评分](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/11_estimate.ipynb) | 48,058 × 10 | **15.8 ms** | 31.0 ms | 逐位一致 |
| [完整的 TME 分析工作流](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/12_complete_workflow.ipynb) | 60,483 × 10 | **28.29 s** | 29.59 s | 按所含环节，同上各行 |
<!-- BENCHMARK_TABLE_END -->

CIBERSORT 使用 **100 次置换、`QN=False`**，输入为完整 STAD TPM 矩阵。
四个独立签名教程使用 872 × 348 的 IMvigor210 限定基因面板；完整工作流则在
STAD 全转录组上进行签名评分，因此不能把各行时间直接相加。硬件、样本量、
基因覆盖率和置换次数都会影响速度。

**“与 IOBRpy 一致性”一列不是计时**，而是官方一致性门
（[`tests/test_parity_official.py`](https://github.com/LCGaoZzz/iobrx/blob/main/tests/test_parity_official.py)）在同一批
数据上对该分析的断言：索引、标签与列名完全一致，且每个数值单元精确相等
（`max_abs_diff == 0.0`），对照对象是同环境内运行的 IOBRpy 原版实现
（[验证记录](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/results/validation.json)）。唯一例外是 CIBERSORT 的
P-value：原版用操作系统熵做置换种子，连它自己都无法逐次复现；iobrx 的
P-value 采用固定种子，跨运行、跨线程数稳定，公式与 `1/perm` 粒度与原版
相同。全部 11 项检查由 CI 在每次推送和 PR 上用
[已验证的依赖约束](https://github.com/LCGaoZzz/iobrx/blob/main/tests/constraints-validated.txt) 复跑；同一契约在历史
224 线程 Xeon 服务器上也成立（[BENCHMARKS.md](https://github.com/LCGaoZzz/iobrx/blob/main/BENCHMARKS.md)）。逐位一致
是上述环境中的实测结果，不是跨平台浮点保证（详见
[兼容性与精度说明](https://github.com/LCGaoZzz/iobrx/blob/main/docs/PORTABILITY.md)）。

[原始重复测量与环境](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/results/benchmark.json) ·
[测量方法](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/BENCHMARKS.md) · [历史服务器基准](https://github.com/LCGaoZzz/iobrx/blob/main/BENCHMARKS.md)

## 使用时最需要分清的几点

| 内容 | 正确理解 |
| --- | --- |
| `anno_eset(method="mean")` | 用均值对重复候选行排序并保留一行，并非对这些行求平均后合并 |
| `count2tpm` | 使用原始非负 counts；基因过滤和去重可能使输出列和略低于一百万 |
| `zscore` 签名方法 | 是上游命名，计算预处理表达的签名均值，不保证输出均值为 0、方差为 1 |
| `integration` | 拼接 PCA、zscore 和 ssGSEA 结果，不是把它们平均成一个分数 |
| CIBERSORT | 相对免疫组成；P-value 是整体拟合的置换统计量，不是细胞间差异检验 |
| EPIC / quanTIseq 的余项 | 未表征成分，不能自动解释成肿瘤纯度 |
| MCP-counter / ESTIMATE | 丰度或富集评分，不是细胞百分比 |
| ESTIMATE 平台参数 | 只有精确的 `"affymetrix"` 会触发上游纯度转换；`"affy"` 不会。本教程对 RNA-seq 使用 `"rnaseq"` 并只展示评分 |

所有热图标准化、余项合并和正表达值分布图仅用于展示，不改动原始分析输出。
教程没有虚构疗效、分组或生存标签。

## 兼容性与结果一致性

默认 `backend="auto"`。缺少原生扩展时，CIBERSORT 和签名分析可回退到
IOBRpy；原生 CIBERSORT 所需 BLAS 不兼容时也会回退。可通过
`backend="python"` 显式使用上游流程，或在导入前设置 `IOBRX_DISABLE_RUST=1`。

**CPU 指令集兼容不等于所有平台都已验证。** IOBRpy 的发行包仍限制了部分
系统和 Python 版本的便捷安装。Windows 推荐 WSL2；macOS、ARM 和原生 Windows
尚未完成全栈验证。从源码 `pip install .` 仍需要编译器，运行时回退不等于免编译安装。

11 项官方数据一致性检查比较的是同一环境内的 IOBRpy 与 iobrx。
CIBERSORT 权重、Correlation 和 RMSE 精确一致；原生实现使用固定种子的置换，
上游 Python 使用未固定种子的置换，因此 P-value 不作为逐值相等的检查对象。
这些检查已纳入 CI：每次推送与 PR 都会在 GitHub 托管 runner 上按
[已验证的依赖约束](https://github.com/LCGaoZzz/iobrx/blob/main/tests/constraints-validated.txt) 复跑全部 11 项。
跨 CPU、BLAS 或依赖版本的逐位相同不在保证范围内。

```bash
python -m pytest -q
IOBRX_TESTDATA=tutorials/data python -m pytest -q -m full
python scripts/execute_tutorials.py
python scripts/benchmark_tutorials.py
python scripts/validate_tutorials.py
```

iobrx 代码使用 [MIT](https://github.com/LCGaoZzz/iobrx/blob/main/LICENSE)；示例数据保留[上游来源及 GPL-3 条款](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/data/README.md)。
发表分析时请引用 IOBR/IOBRpy 及实际使用的方法论文。
