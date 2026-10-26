# FC Tutorial 现有功能与迁移清单

盘点日期：2026-10-25。依据本机实际文件，未访问远端 OSF，也未重新运行拟合。

## 1 找到的材料

较新的本地项目位于：

```text
C:\Users\Owner\Documents\ChatGPT\fc_tutorial
```

本次主要核对的本地 OSF 整理版位于：

```text
output/osf_share_20261018_organized/
├── README.md
├── tutorials/
│   ├── 01_tirt.ipynb
│   ├── 02_gtum.ipynb
│   ├── 03_2plm_rank.ipynb
│   ├── 04_ggum_rank.ipynb
│   ├── 05_mgppm.ipynb
│   ├── forced_choice_extensions.ipynb
│   └── resources/
│       ├── data/empirical/
│       ├── data/examples/
│       ├── tutorial_helpers.py
│       ├── probit_helpers.py
│       ├── workflow_helpers.py
│       └── requirements.txt
└── computational_performance/
    ├── models/
    ├── data_preparation/
    ├── configurations/
    ├── scripts/
    └── sources/
```

这是共享材料的**本地快照**，不是对 OSF 当前远端内容的确认。它比当前网络盘教程副本中的旧 NumPyro 快照更适合作为此次包功能盘点的依据。目录名中的日期也不代表每个文件的最后修订日期。

额外核对了本地项目的 `src/fc_pymc/`：已有数据准备、五个模型、先验配置、预测、log-likelihood、拟合与诊断辅助模块。这些是重构参考，不能据此宣称 PyFCIRT 已实现同等能力。

## 2 核心模型：有代码与保存输出的功能

- **TIRT，`01_tirt.ipynb`。** 二元与等级成对反应，logit/probit 路径。包括特质相关、先验与识别说明、抽样、链诊断、后验预测、PPP、预测评估和估计展示。这里的 Categorical 用于成对等级反应时，不能视为完整 block 排名似然。
- **GTUM，`02_gtum.ipynb`。** 二元和等级成对反应、unfolding 均值结构及 block 效应，包含相同的分析流程。完整排名事件需要另行定义联合效用及参数关系。
- **2PLM-RANK，`03_2plm_rank.ipynb`。** 完整排名；statement endorsement odds 通过逐步选择构成排名概率，并进入 Categorical。作为后续排名类模型参考，不包含 graded 分支。
- **GGUM-RANK，`04_ggum_rank.ipynb`。** GGUM endorsement 机制加逐步选择排名概率，完整排名 Categorical。与 2PLM-RANK 可以核对并共享数学上相同的逐步选择操作，statement 响应机制仍单独实现。
- **MGPPM，`05_mgppm.ipynb`。** 排名与等级对两个分支，通过 score-weighted 类别效用和 Categorical 表达。选为首版模型，使两种响应的完整流程都能被验收；与逐步选择排名的机制分别处理。

Notebook 中可见四链抽样、`pm.compute_log_likelihood`、后验预测、PPP 与 PSIS-LOO 的代码和输出。原抽样设置是教程分析设置，不直接当作包默认值或日常测试规模。支持范围主要由现有示例体现，**不能由数组参数化推断已经验证任意 block size 或混合族联合拟合**。

五个核心 notebook 的代码 cell 数分别为 23、23、14、14、25，均保存了非空 execution_count，未发现 error 类型输出。扩展 notebook 为 59 个代码 cell，同样未发现 error 类型输出。这是本次 JSON 静态检查结果；不等于重新执行通过、全部模型恢复通过或每条可选分支都运行过。

## 3 扩展 notebook 已有的示例

`forced_choice_extensions.ipynb` 含：

- **BRB IRT**：成对作答及 block 层相关结构的示例。
- **JRT-TIRT**：反应与反应时的联合示例。
- **Thurstonian D-diffusion IRT**：扩散模型示例，涉及 HSSM。
- **M-TCIR**：潜在反应质量／类别结构，使用 Potential，并手动组织预测量。
- **ZG-MUPP**：额外 unfolding/偏好响应机制的成对示例。

每节可见数据、模型、拟合、诊断、预测、拟合检查及参数估计的组织。这些属于后续独立族或组件候选，不能直接作为首版完整排名族导出。Potential 的目标密度也不会自动提供逐观测 likelihood 与生成机制。

## 4 可以优先继承的工作流

- **设计和 ID 对齐。** 已有规范数据准备、完整排名及比较转换思路；迁入 FCDesign/FCData 后统一保留标签，转换函数不参与统计定义。
- **PyMC + ArviZ 流程。** 原教程由 PyMC 发起 NumPyro/JAX NUTS，使用 DataTree 结果、诊断和预测。新包沿用生态接口，采样器设为显式选项。
- **模型概率。** 五个 `computational_performance/models/*.py` 文件给出直接 PyMC 声明，可用于公式、符号及数组的交叉审查。它们依赖外部准备变量，不能原封不动作为公共包函数。
- **FC 专属检查。** 迁移 PPP 的 discrepancy 定义及按 trait/person 展示的思路；为完整排名重做观测单位和分组映射。通用统计量与绘图调用 ArviZ。
- **数据与例子。** 已有 TIRT、2PLM-RANK、MGPPM 的经验输入，以及 GTUM、GGUM-RANK、等级反应和扩展的准备数据。纳入发行包前逐项检查授权、来源和编码；先用小型自生成数据完成教程。
- **模拟。** 已有 Python 扩展示例生成脚本；性能研究的核心数据生成主要依赖原作者 R 工具。包的核心模拟应单独提供 Python 实现，不引入运行时 R 依赖。

## 5 需要重新设计或审查的部分

本包当前确定的输入与来源教程有两点差别：默认使用选择顺序列表，逐 statement 长表保留为规范视图／兼容入口；graded 只迁移二元题组定义，不迁移 triplet 内多次 graded 比较。参考 notebook 中 triplet graded 的存在不代表它在本包首轮范围内。

1. **Thurston 完整排名概率。** 从相互依赖的效用差推导联合事件；审查原 comparison 阈值是否能写成 statement 参数差，以及 block 随机项是否具有一致的效用解释。没有这个定义就不能只改 observed 分布。
2. **混合大小与混合族。** 原示例多按固定 triplet 布局排列参数；新包要从全局 statement 身份和观测映射构造计算组，共享 theta，并验证联合识别。
3. **先验。** 原教程的固定 profile 与识别设置是参考；新包须支持显式覆盖、最终先验预览和冲突检查。不能把历史不同 prior profile 的结果混作当前默认配置的验证。
4. **兼容性辅助。** `workflow_helpers.py` 中存在针对 PyMC 6.3.1 变换变量的 likelihood 克隆处理。迁移时先在包支持的环境复现问题，再决定是否需要局部兼容代码；不直接复制内部属性操作作为长期接口。
5. **数据保存与新预测。** 教程保留的图、数组和输出不等于包的保存／重建协议；新被试、新 statement 预测、题目协变量均按新需求建设。
6. **代码与数据授权。** 新仓库已有 MIT 许可证；原作者的 R/Stan 文件、外部数据和其他第三方资源仍遵循各自授权，不能因放入新仓库而改变许可证。

## 6 不迁入运行库的材料

Stan 源码、原作者 R 包、十次重复 benchmark 控制器、外部环境安装器、远程 GPU 调度、论文产物和大型结果目录保留为独立研究参考。包只保留所需的数学测试、受控恢复测试和小型合法示例。

共享教程 requirements 记录 Python 3.12、PyMC 6.3.1、PyTensor 3.3.0、ArviZ 1.3.0、NumPyro 0.20.0 和 JAX 0.9.0.1。这是来源环境记录，**不是 PyFCIRT 的最终兼容范围**；包的依赖边界由安装和执行测试确定。
