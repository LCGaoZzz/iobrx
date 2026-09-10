# iobrx

**保留 IOBRpy 的分析语义，让肿瘤微环境分析更快、更容易复现。**

[English](https://github.com/LCGaoZzz/iobrx/blob/main/README.md) · [教程与图集](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/README.md) · [兼容性说明](https://github.com/LCGaoZzz/iobrx/blob/main/docs/PORTABILITY.md) · [完整基准记录](https://github.com/LCGaoZzz/iobrx/blob/main/BENCHMARKS.md)

**iobrx 是基于原版 [IOBRpy](https://github.com/IOBR/IOBRpy) 构建、独立维护的加速与集成层。**
它在运行时依赖 IOBRpy，沿用上游的参考资源、基因签名和分析语义。
iobrx 的新增工作主要是 Rust 内核、并行与向量化加速、pandas 接口、
经过验证的教程，以及 Omicos Agent 接入。输出是普通的 DataFrame 与文件，
数值一致性按方法、参数和数据集验证，不能推广为每个新增流程、任意输入都逐位一致。
具体例外与本轮验证范围见 [0.3 验证记录](docs/VALIDATION_0.3.md)。

[原版 IOBRpy 仓库](https://github.com/IOBR/IOBRpy) · [IOBRpy 官方文档](https://iobr.github.io/IOBRpy/)

- **不再强制要求 AVX-512**。已在没有 AVX-512 的 i9-13900KF 上实际运行；排序交给本机 NumPy 选择兼容实现。
- **28 本已执行的 Notebook**：原有 12 本、11 本新增矩阵/文件教程，以及 5 本真实 FASTQ、BAM、HLA 教程，均含代码、结果和内嵌图；同时提供 PNG、PDF、SVG。
- 每张分析图经历初稿、第一轮版式调整、第二轮精修，采用白底、低饱和配色、细轴线和可编辑矢量文字。参见[逐图修改记录](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/FIGURE_REVIEW.md)。

**29 个公共 API** —— 27 个工作流函数（另含 `deconvolute_quantiseq` 别名与
`load_official` 数据助手），覆盖以下 IOBRpy 工作流类别：免疫反卷积
（CIBERSORT、BayesPrism、EPIC、quanTIseq、MCP-counter、ESTIMATE）、签名评分
（PCA / zscore / ssGSEA / integration）、TPM 转换与基因注释、免疫表型评分
IPS、配体-受体矩阵、NMF 与 TME 聚类、完整 `tme_profile` 链路，以及
FASTQ→TME 编排阶段（fastp / salmon / STAR / TRUST4 / SpecHLA）。

**Omicos 接入：** harness 现有 27 个带输入契约的分析标识（四种签名评分分别计数），其中 16 个为本轮新增适配器。HLA 与自定义参考 BayesPrism 仍通过 Python API 使用。[输入与边界](agent-harness/omicos/skills/iobrx/references/extended-workflows.md)。

**证据范围：** 下方 R3–R6 耗时沿用原科学智能体报告；部分原始脚本和日志尚未入库，不能仅凭本 PR 独立复现全部倍率。本轮数值测试、模拟工具测试与实际教程耗时分别记录。

**新增真实数据证据：** [FASTQ/BAM/HLA 复现步骤、重复计时与日志](benchmarks/real_tools/README.md)
及教程 24–28 使用公开测序 reads 和真实工具。4 线程 Salmon 在原版自身重跑时
也有差异；本次小规模 HLA 提取中 iobrx 更慢。这些结果不支持“每项都更快”或
“所有输入逐字节一致”的结论。

### 0.3.0 新增

- **新增 18 个工作流 API**（战役回合 R3–R6）：`nmf`、`merge_salmon`、
  `merge_star_count`、`prepare_salmon`、`log2_eset`、`ips`、`mouse2human`、
  `lr_cal`、`tme_cluster`、`bayesprism`、`tme_profile`、`fastq_qc`、
  `batch_salmon`、`batch_star_count`、`trust4`、`runall`、`spechla`、
  `hla_typing`，外加 `calculate_sig_score` 内部胶水的 30.3× 优化。
- **补齐独立 HLA reads 提取接口**：`extract_hla_read` 只从单个 BAM/CRAM
  提取 reads，默认使用已准备好的工具环境，不自动安装依赖。
- **4 个新 Rust 内核**：`lr_gene_valid_mask`（LR_cal 基因过滤）、
  `tme_kmeans`（k-means + KL 指数）、`merge_salmon_parse`（quant.sf 解析）、
  `bp_gibbs`（BayesPrism Gibbs 采样器，逐位复现 numpy 完整 RNG 链）。
- **`tme_profile` 端到端 10.59×**（对原版 CLI，冻结 STAD 数据）——瓶颈转移
  链 1.12× → 8.44× → 10.59× 的终点（[BENCHMARKS.md Part II](https://github.com/LCGaoZzz/iobrx/blob/main/BENCHMARKS.md)）。
- **原战役报告的 R6 基准**：14 个核心候选在统一冷启动口径下对原版 CLI 重测，全部
  parity 合同 PASS。原提交记录为 188 项默认测试通过；当前结果见[验证记录](docs/VALIDATION_0.3.md)。

![CIBERSORT 示例](https://raw.githubusercontent.com/LCGaoZzz/iobrx/main/tutorials/figures/07_cibersort.png)

## 新增教程与实际耗时

新增 11 本已执行 Notebook，包含代码、结果、内嵌图，以及两轮图形调整记录。
下表是本机 WSL／Omicos 解释器中最终一轮 Notebook 的单次 API 调用时间，
请求 2 线程，不包含输入准备。它们是小型教程的观测值，不是性能倍率基准；
NMF 的 BLAS 并行度不只由请求线程数控制。BayesPrism 使用缩短的演示采样链，
文件合并教程使用合成数据，均已在对应 Notebook 中说明。

| Tutorial | Input shape | API call time |
| --- | --- | --- |
| [13_ips](tutorials/13_ips.ipynb) | 48058 × 4 | 0.016 s |
| [14_lr_cal](tutorials/14_lr_cal.ipynb) | 48058 × 4 | 0.143 s |
| [15_nmf](tutorials/15_nmf.ipynb) | 10 × 22 | 0.465 s |
| [16_tme_cluster](tutorials/16_tme_cluster.ipynb) | 10 × 22 | 0.044 s |
| [17_log2_eset](tutorials/17_log2_eset.ipynb) | 48058 × 4 | 0.235 s |
| [18_mouse2human](tutorials/18_mouse2human.ipynb) | 4 × 3 | 0.020 s |
| [19_merge_salmon](tutorials/19_merge_salmon.ipynb) | 3 × 3 | 0.076 s |
| [20_prepare_salmon](tutorials/20_prepare_salmon.ipynb) | 3 × 4 | 0.014 s |
| [21_merge_star_count](tutorials/21_merge_star_count.ipynb) | 3 × 3 | 0.073 s |
| [22_tme_profile](tutorials/22_tme_profile.ipynb) | 48058 × 2 | 4.828 s |
| [23_bayesprism](tutorials/23_bayesprism.ipynb) | 128 × 3 | 0.351 s |

## 安装

已验证目标为 **Python 3.11、Linux x86-64／WSL2**，建议使用独立环境。
iobrx 的参考数据已内置在 wheel 中，默认安装即完整可用——不再依赖 IOBRpy，
也无需额外的约 90 MB 下载：

```bash
python -m pip install iobrx
# 清华镜像：python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple iobrx
```

可选 extras：

| Extra | 内容 | 适用场景 |
| --- | --- | --- |
| `iobrx[python]` | `iobrpy==0.2.1` | Python 回退后端、quanTIseq 与签名打分（它们复用上游 IOBRpy 模块）、官方 parity 测试。 |
| `iobrx[strict]` | `numpy==2.2.6`、`scipy==1.16.3`、`scikit-learn==1.7.2`、`gseapy==1.3.1` | 已验证的精确数值环境；官方 bit-exact parity 门与 CI 以此版本断言。 |
| `iobrx[test]`、`iobrx[tutorials]` | pytest／notebook 栈 | 开发与已执行教程。 |

默认安装使用版本范围，可与共享环境（如 Omicos 内核）共存而无需强制钉版。
已发布的参考数字以 `iobrx[strict]` 为验证依据；在更新的 numpy 上，native
求解器可能相差最后一个 ulp——细节与测量见
[BENCHMARKS.md](BENCHMARKS.md) 与 [docs/PORTABILITY.md](docs/PORTABILITY.md)。

### 大队列与线程

`n_threads=None`（默认）解析为 `min(8, os.cpu_count())`，是保守的桌面选择。
在更大机器上应显式传入物理核数——16–64 线程在全转录组输入上仍有实质增益：

```python
import iobrx
iobrx.set_threads(os.cpu_count())          # 进程级
iobrx.cibersort(eset, n_threads=32)        # 单次调用
```

每次 `cibersort()` 调用另有约 26 s 的固定一次性开销（签名加载 + BLAS 握手）。
多个队列共用同一签名矩阵时，应合并为一次调用后再拆分权重，而不是逐队列调用。

固定版本 wheel、源码包、校验文件与容器 digest 通过
[v0.3.0 发行页面](https://github.com/LCGaoZzz/iobrx/releases/tag/v0.3.0)交付。
源码开发需要 Cargo 和 C++17 编译器：

```bash
git clone https://github.com/LCGaoZzz/iobrx.git
cd iobrx
python -m pip install -c tests/constraints-validated.txt ".[test,python]"
python -c "import iobrx; print(iobrx.backend_info())"
python -m jupyterlab tutorials
```

Jupyter 内核应使用同一个解释器。这条源码安装命令会编译扩展，不能称为免编译
安装。核心分析容器使用 `ghcr.io/lcgaozzz/iobrx:0.3.0`；需要不可变版本时，
使用发行附件 `container-digest.txt` 中的 digest。

## 快速开始

```python
import numpy as np
import pandas as pd
import iobrx

iobrx.set_threads(8)
counts = pd.read_parquet("tutorials/data/eset_stad.parquet")  # 基因 × 样本
tpm = iobrx.count2tpm(counts, check_data=True, remove_version=True)
log_expression = np.log2(tpm + 1)

# 反卷积与签名评分（RNA-seq 反卷积示例使用线性 TPM）
cib = iobrx.cibersort(tpm, perm=100, QN=False)          # Rust NuSVR 核
epic = iobrx.epic(tpm)["cellFractions"]
qnt = iobrx.quantiseq(tpm, tumor=True, rmgenes="default")
scores = iobrx.calculate_sig_score(
    log_expression, "signature_collection", method="integration"
)

# 0.3.0 新移植 API
lr = iobrx.lr_cal(eset="tpm_symbol.csv", output_file="lr.csv",
                  data_type="tpm", id_type="symbol", cancer_type="pancan")
ips = iobrx.ips(eset="tpm_symbol.csv", output_file="ips.csv")
clusters = iobrx.tme_cluster(df=pd.read_csv("tme_transposed.csv"), id="sample")
iobrx.bayesprism(bulk="bulk_counts.csv", out_dir="bp_out", n_threads=8)
#   加 backend="rust" 启用逐位一致的 Rust Gibbs 内核

# 完整 TME 分析链，单进程跑完（对原版 CLI 10.59×）
iobrx.tme_profile(input="TPM.csv", output="tme_out", threads=16)

# FASTQ → TME 编排（外部工具需在 PATH 上，见下文）
iobrx.runall(mode="salmon", outdir="run_out", fastq="raw_fastq_dir",
             threads=16, resume=True,
             unknown=["--index", "references/salmon"])
```

`runall(resume=True)` 只有在产出表格的步骤成功完成、输出哈希匹配时才会复用结果。
失败或写入中断留下的半成品会重新执行；已成功的上游步骤仍可复用。
运行状态格式 2 增加了逐步骤记录；更早、没有这些记录的目录需要新建 `outdir`，
已有格式 2 状态可以继续使用。输入、参数、参考文件或记录的计算产物被修改后，
需要使用新目录；增删改笔记和图表不会阻断恢复。
进程被强制终止留下的未验证计算产物不会直接当作成功结果复用。

初次使用建议打开[完整工作流 Notebook](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/12_complete_workflow.ipynb)
（仓库自带公开示例数据，安装后无需下载），先理解不同方法需要的输入尺度，
再替换成自己的数据。四个独立签名教程使用公开的 IMvigor210 演示面板
（872 特征 × 348 样本）。

## 29 个 API 一览

| API | 一句话说明 |
| --- | --- |
| `cibersort` | CIBERSORT LM22 免疫反卷积（NuSVR）—— Rust 核，置换固定种子 |
| `calculate_sig_score` | 每样本签名评分：`pca` / `zscore` / `ssgsea` / `integration` |
| `count2tpm` | 原始 count 矩阵 → TPM（向量化，逐位一致） |
| `quantiseq` | quanTIseq TIL10 反卷积，HGNC 别名映射进程内缓存 |
| `deconvolute_quantiseq` | 保留原版函数名的别名 |
| `epic` | EPIC 细胞比例与 mRNA 比例 |
| `mcpcounter` | MCP-counter 细胞群丰度评分 |
| `estimate_score` | ESTIMATE 基质/免疫评分与肿瘤纯度 |
| `anno_eset` | 探针 / Ensembl ID 聚合到基因 symbol |
| `bayesprism` | BayesPrism 反卷积 —— python-fast 默认，Rust Gibbs 可选，确定性已修复 |
| `tme_profile` | 完整 TME 分析链（签名评分 + 6 种反卷积 + LR_cal）单进程跑完 |
| `nmf` | NMF 聚类，轮廓系数选 k |
| `tme_cluster` | TME k-means 聚类，KL 指数选最优 k（Rust 核） |
| `lr_cal` | 配体-受体对表达矩阵（log2 TPM 取小；Rust 基因过滤核） |
| `ips` | 免疫表型评分 Immunophenoscore（Charoentong 2017 四区块） |
| `merge_salmon` | 合并 Salmon `quant.sf` 目录 → TPM / count 矩阵（Rust 解析引擎） |
| `merge_star_count` | 合并 STAR `ReadsPerGene.out.tab` → 单一 count 矩阵 |
| `prepare_salmon` | Salmon TPM → 去重 symbol / ENSG / ENST 矩阵 |
| `log2_eset` | 对基因 × 样本矩阵做 `log2(x+1)` |
| `mouse2human` | 小鼠 → 人类基因 symbol 转换 |
| `fastq_qc` | fastp + MultiQC 质控（外部工具阶段） |
| `batch_salmon` | 双端 FASTQ 批量 Salmon 定量 |
| `batch_star_count` | 批量 STAR 两遍比对 + GeneCounts |
| `trust4` | TRUST4 TCR/BCR 重建 + 加速的免疫后处理 |
| `spechla` | SpecHLA 单样本全分辨率 HLA 分型 |
| `extract_hla_read` | 从单个 BAM/CRAM 提取 HLA 相关 FASTQ，不执行分型 |
| `hla_typing` | 从 BAM 目录批量 HLA 分型 |
| `runall` | FASTQ → TME 端到端编排器（salmon / star 两条链） |
| `load_official` | 解析 / 下载 IOBR 公开示例数据 |

## 三条路线、一个选择 —— 每个 API 如何加速

每个移植模块都沿三条路线评估过：继续调**原版 iobrpy CLI**、写**纯
Python 快路径**、或做 **Rust 内核**。最终选择 = 在通过 bit-exact 合同的
前提下墙钟最快的那条。19 个移植 API 的 speedup 为 **原战役报告的 R6 基准**统一口径
（冷启动、每次运行新子进程、双臂同窗交替、取中位数——
`research/bench_r6/results.json`，[BENCHMARKS.md §II.6](https://github.com/LCGaoZzz/iobrx/blob/main/BENCHMARKS.md)）；
编排阶段的比值来自 R5 真实数据运行（[§II.8](https://github.com/LCGaoZzz/iobrx/blob/main/BENCHMARKS.md)）；R3 之前的
老模块沿用官方 gate 数字（[Part I](https://github.com/LCGaoZzz/iobrx/blob/main/BENCHMARKS.md)）。

| 模块 | 原版路线 | 纯 Python 路线 | Rust 路线 | 最终选择 | speedup | parity 合同 |
| --- | --- | --- | --- | --- | --- | --- |
| cibersort | iobrpy CLI | python 回退 | Rust NuSVR 核（内置 sklearn-1.7.2 libsvm） | **rust (auto)** | 官方 gate：端到端 34.1×，最高 **53.6×**（held-out）——见 BENCHMARKS Part I | bit-exact，未播种 P-value 列除外 |
| calculate_sig_score | iobrpy CLI | python | Rust ssGSEA/PCA 核 + 胶水优化 | **rust (auto)** | 胶水步 **30.3×**（R5）；阶段 gate 6.6–11.5×（Part I） | bit-exact |
| count2tpm / epic / estimate / mcpcounter / quantiseq / anno_eset | iobrpy CLI | python | rust / 向量化 | **rust-向量化 (auto)** | 见 BENCHMARKS Part I（count2tpm 最高 48.8×、quantiseq 最高 16.9×、mcpcounter 4.2–5.0×、anno_eset 3.5–3.8×、estimate 2.5–3.4×、epic 1.6–1.8×） | bit-exact |
| nmf | iobrpy CLI | **python（选）** | 无 Rust 核（两侧共用 sklearn NMF） | **python** | **1.21×**（R6） | bit-exact：3 文件 sha256（clusters / top_features / pca_plot.png） |
| merge_salmon | iobrpy CLI | python 顺序解析（R6 1.86×） | Rust 读取解析引擎 | **rust (auto)** | **2.41×**（R6） | 列对齐后 token 级一致（上游 `as_completed` 列序非确定；移植侧列序确定） |
| merge_star_count | iobrpy CLI | **python（选）** | 无 | **python** | **1.75×**（R6） | 列对齐 token 级一致 + 统计行 **bug-compat**（上游"前 4 全局统计行不清除"缺陷原样保留） |
| prepare_salmon | iobrpy CLI | **python（选）** | 无 | **python** | **2.37×**（R6） | bit-exact（sha256） |
| log2_eset | iobrpy CLI | **python（选）** | 无 | **python** | **2.29×**（R6） | bit-exact（sha256） |
| ips | iobrpy CLI | **python（选）** | 无 | **python** | **3.25×**（R6） | bit-exact（sha256） |
| mouse2human | iobrpy CLI | **python（选）** | 无 | **python** | **3.76×**（R6） | bit-exact（sha256） |
| lr_cal | iobrpy CLI | python 回退 | Rust 基因过滤核 | **rust (auto)** | **8.48×**（R6） | bit-exact（sha256） |
| tme_cluster | iobrpy CLI | python RNG / k-means 循环 | Rust k-means 核 | **rust (auto)** | **7.17×**（R6） | bit-exact（sha256） |
| bayesprism | iobrpy CLI | **python-fast（auto 默认）** | Rust Gibbs 核（opt-in） | **python auto；`backend="rust"` 可选** | python **2.62×** / rust **3.74×**（R6） | bit-exact：hs0 三文件 sha256；Rust 核逐位复现 numpy 完整 RNG 链 |
| tme_profile | iobrpy CLI（9 子步串行） | — | reuse_fast v3（sig 胶水 + Rust LR_cal） | **reuse_fast（`cibersort_backend="original"`）** | **10.59×**（R6） | 9 个输出：7 个逐字节一致 + 2 个剥除未播种 cibersort P-value 列后一致 |
| fastq_qc | iobrpy CLI | **python（选）** | 无（工具本体做功） | **python** | ≈1.0×（R5 真实数据；启动层轻 14.9×） | bit-exact，fastp 内部 HTML 抖动除外 |
| batch_salmon | iobrpy CLI | **python（选）** | 无 | **python** | ≈1.0×/样本（R5 真实数据；启动层轻 14.8×） | bit-exact，运行元数据时间戳除外 |
| batch_star_count | iobrpy CLI | **python（选）** | 无 | **python** | ≈1×（R5 真实数据，声明的 16 vs 32 线程偏差下；启动层轻 10.6×） | BAM 记录流 + 计数表 bit-exact；header @PG/@CO 携带声明的线程数 |
| trust4 | iobrpy CLI | **python（选，后处理加速）** | 无 | **python** | ≈1.0×（R5 真实数据；stub 启动轻 3.3×） | bit-exact：12/12 文件逐字节一致（含后处理输出） |
| runall | iobrpy CLI | **python（选）** | 无 | **python** | **1.02×**（R5 真实数据） | bit-exact，已归档的工具抖动类除外 |
| spechla | iobrpy CLI | **python（选）** | 无 | **python** | **1.01×**（R5 真实数据） | bit-exact，samtools @PG 随机 ID 抖动除外 |
| hla_typing | iobrpy CLI | **python（选）** | 无 | **python** | **1.05×**（R5 真实数据） | bit-exact，同上 |
| extract_hla_read | iobrpy CLI | 复用现有提取函数 | 无 | **python** | 未测性能倍率 | 已用模拟脚本对照命令与输出契约 |

## 为什么各模块加速不同：一个地板模型

上表所有数字服从同一条规律：**speedup ≈ min(1/(1−p), 地板上限)** ——
p 是原版墙钟中"可压缩热点"的占比，每类模块有自己的不可压缩地板。这是
战役 R1–R6 逐轮测量出的结论（[BENCHMARKS.md Part II](https://github.com/LCGaoZzz/iobrx/blob/main/BENCHMARKS.md)），
不是事后叙事：

1. **Rust 内核只有在 p ≥ 0.6 时才带来 ≥5×。** `lr_cal` 的逐基因 pandas
   过滤占原版墙钟 p=0.81 → **8.48×**；`tme_cluster` 由纯 Python k-means
   主导 → **7.17×**。BayesPrism 的 Gibbs 采样只占 p≈0.24 → Rust 核落在
   **3.74×**（对 python 路线 1.56×），尽管构造上逐位一致，也到不了 5×。
2. **瓶颈转移（多阶段 Amdahl）。** `tme_profile`：**1.12×**（合进程、
   原版代码——未优化的 sig 胶水占墙钟 ~84%）→ **8.44×**（30.3× 胶水优化
   后地板转移到 cibersort + LR_cal ≈76%）→ **10.59×**（LR_cal 换 Rust，
   子步 4.10 s → 0.22 s ≈18.6×；地板变为合同绑定的原版 CIBERSORT 求解器
   58.6% + sig Rust 链 31%）。下一层地板 ~9.3 s（~16.6×）——**不打破
   cibersort bit-exact 合同，≥12× 实际不可达。**
3. **写出天花板（io_merge 类 ~2.4× 封顶）。** `merge_salmon` 的 Rust 解析
   单独测 **10.32×**，但字节级输出合同（pandas `to_csv` 序列化 + gzip）把
   端到端封顶在 ~2.5×：实测 2.41–2.48×，已达同窗理论上限的 ~96%。Rust 化
   读取突破不了写出地板。
4. **import + pandas IO 地板（small 模块 ~2–3.8×）。** 原版 CLI 每次调用
   先付 ~1.3–1.4 s 的 `iobrpy.main` import（总墙钟才 1.3–1.9 s）。消除它
   之后 pandas `read_csv`/`to_csv` 成为新地板：纯 IO 模块落在 2.29–2.37×
   （`log2_eset`、`prepare_salmon`），有净计算的落在 3.25–3.76×（`ips`、
   `mouse2human`），`nmf` 收敛到 1.21×——两侧跑的是同一个 sklearn 求解器
   （共享内核地板）。
5. **编排天花板（speedup ≈ 1）。** `fastq_qc` / `batch_salmon` /
   `batch_star_count` / `trust4` / `runall` / `spechla` / `hla_typing` 的
   墙钟由外部二进制本体主导。真实数据比值 0.99–1.25×（STAR 的 1.25× 完全
   归因于声明过的 32→16 线程偏差，STAR 自己的日志里有 mapping speed
   457 vs 594 M reads/h 佐证）。移植价值在 API 一致、单进程组合（不再依赖
   console-script/PATH）、断点续跑与并行调度、字节级产物一致——以及用
   stub 工具可测的 Python 启动层 **3.3–14.9×** 减重。

## bit-exact 合同与已知非确定项

**bit-exact** 指：与同环境运行的原版 iobrpy 0.2.0 相比，索引/列/dtype/
NaN 掩码完全一致且每个数值单元 `max_abs_diff == 0.0`，或输出文件 sha256
相等。合同由 `tests/test_parity_*.py` 强制（默认 188 个测试通过；官方数据
门在 CI 用冻结 fixtures 复跑）。

所有已知非确定项都是**上游**性质，用原版对原版的对照实验证明，并各有明文
合同条款：

- **CIBERSORT P-value 列** —— 原版置换种子取自操作系统熵（未播种
  `SeedSequence()`），*原版自己*都无法逐次复现该列。iobrx 固定种子：跨
  运行、跨线程数稳定，公式与 `1/perm` 粒度不变。所有含该列的字节合同都
  明文豁免它（cibersort 输出；tme_profile 的 `cibersort_results.csv` /
  `deconvo_merged.csv`）。
- **BayesPrism 状态序** —— 原版的细胞状态迭代顺序跟随每进程的
  `PYTHONHASHSEED`，偶发翻转离散 Gibbs 抽样。iobrx 默认
  `state_order="sorted"` 消除该依赖（theta / theta_cv 对
  `PYTHONHASHSEED=0` 的原版 100% bit-exact；Z_tumor 仅 ULP 级差异，最大
  绝对误差 1.42e-14，0 个离散翻转）；`state_order="legacy"` +
  `PYTHONHASHSEED=0` 精确复现冻结金标准 sha（R6 parity 所用配置）。
- **`as_completed` 列序** —— 原版 `merge_salmon` / `merge_star_count` 的
  输出列序跨运行不定（R6 三次重复观测到三种列序）。iobrx 输出确定的
  sorted 列序；parity 合同按列名对齐后做 token 级 / DataFrame 位级比较。
- **samtools @PG / @RG 头 ID** —— spechla / hla_typing 的
  `<sample>.realign.sort.bam` 继承 `samtools merge` 的随机头 ID（如
  `bwa-7A10F178`）；原版重跑原版呈现同样抖动。比对记录流与其余全部产物
  按字节比较。
- **fastp HTML 抖动** —— `fastp.html` 中 duplication rate 第 7 位有效数字
  在 fastp 二进制内部逐运行变化（原版 CLI 重跑复现了移植侧的数字、反而与
  自己的冻结基线不同）；清洗后的 FASTQ 逐字节一致。
- **运行元数据** —— 日志时间戳（salmon/STAR/TRUST4 日志、multiqc
  uuid/creation date）为声明过的运行元数据；**所有数据产物一律无归一化
  sha256 比较**。
- **bug-compat 保留** —— 会体现在输出字节里的上游缺陷被刻意保留，例如
  `merge_star_count` 原版从不清除的前 4 行全局统计行、`lr_cal` count
  分支的行为。

逐位一致是**在指定输入与钉版依赖（numpy<2.3、scikit-learn<1.8，见安装节）
下的实测结果**，不是跨平台浮点保证（[详情](https://github.com/LCGaoZzz/iobrx/blob/main/docs/PORTABILITY.md)）。

## 外部工具（需自备）

iobrx 只调度重型二进制，不捆绑它们。请自行安装并放入 `PATH`（或用各阶段
的 `*_bin` 参数覆盖）；iobrx 保留上游命令参数。结果一致性仍取决于工具自身
是否确定、以及比较了哪些产物，详见新增真实对照中的 Salmon 波动与比较范围。

| 阶段 | 外部工具 |
| --- | --- |
| `fastq_qc` | fastp、MultiQC |
| `batch_salmon` | salmon |
| `batch_star_count` | STAR（+ samtools） |
| `trust4` | TRUST4（`run-trust4`） |
| `spechla` / `hla_typing` | SpecHLA 工具链：samtools、bwa/bowtie2、bcftools、freebayes、vcflib、blastn、bamUtil（`bam`） |
| `extract_hla_read` | SpecHLA 提取资源、samtools、bamUtil（`bam`）；输入需排序并建立索引 |
| `runall` | 所选 salmon/star 链的全部工具 |

所有纯计算 API（反卷积、签名评分、TPM、注释、IPS、LR_cal、NMF/TME 聚类、
合并类）**不需要任何外部工具**——只需 iobrpy（参考数据 + 回退）与钉版的
科学计算栈。

已有 BAM/CRAM 时，可以只提取 HLA reads：

```python
iobrx.extract_hla_read("sample1", "sample1.bam", "hg38", "hla_reads")
```

这个新接口在两个后端中均默认 `auto_install=False`；任务确实需要安装工具时，
才显式传入 `auto_install=True`。它目前通过 Python API 使用。

## 在 Omicos／Agent 中使用

[agent-harness](https://github.com/LCGaoZzz/iobrx/tree/main/agent-harness) 为 27 个分析标识提供统一的 JSON CLI、
可选的 stdio MCP 服务，以及可随 Omicos catalog 分发的 Agent／Skill。
运行前检查矩阵方向、声明的数据尺度和基因 ID；运行后保存参数、输入／输出
元数据、环境版本、实际后端和耗时。分析继续调用 iobrx 原有 API。
已知请求可以直接运行；能力查询、预检和环境诊断均按需使用。智能体也可以
直接调用 Python API 完成自定义分析。默认不做全量哈希，SHA-256 审计可显式开启；
普通结果检查只确认文件是否存在，不会把后续编辑误判为当时的分析失败。

安装 iobrx 后，在仓库根目录运行：

```bash
python -m pip install ./agent-harness
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
相同。全部一致性门禁由 CI 在每次推送和 PR 上用
[已验证的依赖约束](https://github.com/LCGaoZzz/iobrx/blob/main/tests/constraints-validated.txt) 复跑；同一契约在历史
224 线程 Xeon 服务器（[BENCHMARKS.md](https://github.com/LCGaoZzz/iobrx/blob/main/BENCHMARKS.md) Part I）与 R6 盲测
（Part II）中同样成立。逐位一致是上述环境中的实测结果，不是跨平台浮点保证（详见
[兼容性与精度说明](https://github.com/LCGaoZzz/iobrx/blob/main/docs/PORTABILITY.md)）。

[原始重复测量与环境](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/results/benchmark.json) ·
[测量方法](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/BENCHMARKS.md) ·
[历史服务器基准](https://github.com/LCGaoZzz/iobrx/blob/main/BENCHMARKS.md) ·
[完整性能记录：加速栈 + 移植战役](https://github.com/LCGaoZzz/iobrx/blob/main/BENCHMARKS.md)

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
| `bayesprism` | 需要伪 bulk 计数 + scRNA 参考；输出为细胞类型/状态比例与置信区间 |
| `lr_cal` | 配体-受体对矩阵取两基因 log2 TPM 的较小值，不是相互作用强度 |
| `ips` | 0–1 的免疫表型评分，四区块加权和（Charoentong 2017） |
| ESTIMATE 平台参数 | 只有精确的 `"affymetrix"` 会触发上游纯度转换；`"affy"` 不会。本教程对 RNA-seq 使用 `"rnaseq"` 并只展示评分 |

所有热图标准化、余项合并和正表达值分布图仅用于展示，不改动原始分析输出。
教程没有虚构疗效、分组或生存标签。

## 兼容性与结果一致性

```python
iobrx.backend_info()  # 原生扩展可用性、排序分发、捆绑 BLAS

# 可选：显式走原版 IOBRpy 工作流
cib = iobrx.cibersort(tpm, perm=100, QN=False, backend="python")
# backend="rust" 需要原生支持，不可用时明确报错
```

凡是存在加速路线的函数默认 `backend="auto"`（每个模块最终 shipped 的路线
见上文路线表）。在**导入前**设置 `IOBRX_DISABLE_RUST=1` 可全进程禁用原生
加速。缺少原生扩展不影响公共分析 API（需已安装 IOBRpy 及其依赖）；缺少
兼容 OpenBLAS 时 CIBERSORT/PCA 自动回退。

**CPU 指令集兼容不等于所有平台都已验证。** IOBRpy 的发行包仍限制了部分
系统和 Python 版本的便捷安装。Windows 推荐 WSL2；macOS、ARM 和原生 Windows
尚未完成全栈验证。从源码 `pip install .` 仍需要编译器，运行时回退不等于免编译安装。

全部 parity 门在 CI 每次推送与 PR 上按
[已验证的依赖约束](https://github.com/LCGaoZzz/iobrx/blob/main/tests/constraints-validated.txt)复跑；断言即精确相等
——标签与每个数值单元，或文件 sha256——对照同环境原版，附上述明文非确定
性条款（CIBERSORT 置换：原生实现固定种子、上游未固定，P-value 不作逐值
相等对象）。跨 CPU、BLAS 或依赖版本的逐位相同不在保证范围内。详见
[精度与回退说明](https://github.com/LCGaoZzz/iobrx/blob/main/docs/PORTABILITY.md)。

## 复现、测试与贡献

```bash
python -m pytest -q                      # 188 个冒烟 + parity + 可移植性测试
IOBRX_TESTDATA=tutorials/data python -m pytest -q -m full   # 官方数据门
python scripts/execute_tutorials.py      # 全部 12 本，新内核，内嵌图
python scripts/benchmark_tutorials.py    # 每项分析首次 + 3 次重复
python scripts/validate_tutorials.py     # 输出、导出与数据校验和
```

教程可在 GitHub 上直接阅读；交互运行：`python -m jupyterlab tutorials`。
绘图辅助函数在 [`tutorials/_common.py`](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/_common.py)；每个分析调用
在 Notebook 中保持可见。[逐图修改记录](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/FIGURE_REVIEW.md)保留了
两轮精修的草稿与终稿。

贡献请保留 parity 门，并为数值行为改动添加针对性回归测试。`bench/` 下的
文件是冻结的历史工件。报告问题时请附包版本、backend 信息、输入形状与
表达尺度。

## 致谢、引用与许可

感谢 [IOBRpy 开发者](https://github.com/IOBR/IOBRpy)、
[IOBR 团队](https://github.com/IOBR/IOBR)及各分析方法的原作者，
为本项目提供上游分析流程和参考资源。

发表分析时请引用 IOBR/IOBRpy 及实际使用的方法原文——包括 CIBERSORT
（Newman 等）、BayesPrism、quanTIseq、EPIC、MCP-counter、ESTIMATE、IPS
（Charoentong 等 2017）、TRUST4、SpecHLA，以及运行编排阶段时的
fastp/salmon/STAR 工具论文；参见
[IOBRpy 预印本](https://doi.org/10.64898/2026.07.17.739055)和
[官方引用指南](https://iobr.github.io/IOBRpy/Citation.html)。预印本报告的是
原版实现*内部*的 CIBERSORT 线程扩展——与本文的移植加速是不同测量轴（见
[BENCHMARKS.md §II.9](https://github.com/LCGaoZzz/iobrx/blob/main/BENCHMARKS.md)）。

iobrx 代码使用 [MIT](https://github.com/LCGaoZzz/iobrx/blob/main/LICENSE)；内置（vendored）源码保留其
[第三方声明](https://github.com/LCGaoZzz/iobrx/blob/main/rust/vendor/THIRD_PARTY_NOTICES.md)；公开示例数据保留
[上游来源及 GPL-3 条款](https://github.com/LCGaoZzz/iobrx/blob/main/tutorials/data/README.md)。
