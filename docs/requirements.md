# PyFCIRT 功能与需求清单

Status: 2026-10-25. This document records requirements, including future scope; it is not an implementation checklist. The package, data API and MGPPM analysis modules are implemented. Posterior MCMC, full empirical tutorials and population recovery are optional manual validation, outside the formal test suite. See the [development roadmap](roadmap.md) for current scope and the [migration inventory](tutorial_migration.md) for the original tutorial capabilities.

## 阅读方式与开发顺序

**模型类别与功能层级分别管理。** 第一版选择 **MGPPM**，同时完成排名和二元 graded 的常规分析。随后 2PLM-RANK、GGUM-RANK、TIRT、GTUM 复用同一流程；五个核心模型接齐后再推进局部先验、混合族和协变量。2PLM-RANK 保持排名类模型，不强加 graded。第二批模型候选单独管理。

首版流程包括：列表／NA 与 graded 输入、先验、拟合、诊断、估计、预测、PPP、逐观测 likelihood、PSIS-LOO、DIC、WAIC、已有及**新被试评分**、保存读取和完整教程。排名与 graded 分别验收，不仅是能读入数据。V1-11 固定指标参考实现与数值对照要求，详见开发计划及 [相对拟合规格](model_comparison.md)。

## 1 项目目标与范围

- **R01 易用的 FC 分析包。** 项目名 PyFCIRT，发行名、导入名、仓库名为 `pyfcirt`。普通用户通过少量对象完成设计、读入、建模、拟合、诊断、预测和评分。
- **R02 充分使用生态。** PyMC 创建计算图、抽样和预测；ArviZ 与 xarray 保存带标签结果，提供通用统计诊断和绘图。包集中解决 FC 特有的数据语义、似然、识别、评分与检查。
- **R03 可维护、可扩展。** 使用标准 Python 打包、`src/` 布局、类型标注、少量组合对象及直接可读的模型模块。借鉴 Bambi 的规格、先验和构建边界；不照搬其内部实现，也不提前建设公式语言、多后端编译器、继承树或自动插件注册系统。
- **R04 独立产品边界。** FC Tutorial 为数学、代码、示例和测试参考；PyFCIRT 独立开发。不纳入 Stan 实现、独立 `validation/` 库、论文 benchmark、远程计算控制器或投稿准备工作。包内输入校验及统计正确性测试仍属必要功能。

## 2 数据与设计

- **R05 两张基础表。** 默认 block 作答表为 `person_id, block_id, response`；设计表为 `block_id, position, statement_id, trait, keying`。排名 response 为按偏好排序的列表：`[2,3,1]` 表示第 2 条 → 第 3 条 → 第 1 条；PICK／MOLE 使用同长列表中的 None／NA 表示未知位置。内部规范为 statement-rank 长表，也保留显式长表输入。graded 则用单个类别标量。任务、编码和量尺明确声明，不从数值猜含义。
- **R06 FCDesign 与 FCData。** `FCDesign` 管理 statement、题组成员、参考位置、维度与方向；`FCData` 将作答和设计组合起来。两者不保存先验、活动计算图或估计结果。
- **R07 便捷地定义测验。** 同时提供表格入口和 `from_mapping`：用户给出维度到 statement、block 到 statement 的对应关系即可。两种入口得到同一份规范设计，可导出并查看维度覆盖、题组大小、重复或未使用 statement。
- **R08 不同 block size。** 默认检查所有 block 大小相同；显式开启后允许不同大小。大小从设计成员计算，不能从缺失作答推断。模型须进一步声明经过验证的大小范围。
- **R09 身份与约束。** 全局 statement ID 区别于题组内位置；同一 statement 的维度及 keying 一致。keying 控制参数方向，不反转观测名次。首版单个 statement 对应一个 trait。
- **R10 可选题目特征。** `statement_features` 一行一个 statement，通过 ID 对齐；只在需要协变量时提供。数据层保存数值和来源，模型层选择预测列、变换和统计用途。
- **R11 观测、NA 与 graded。** PICK／MOLE 保留全部选择集合，汇总兼容排名的概率；真实未作答／未施测通过可选状态表与结构性 NA 分开，不静默删项、设并列、填最低名次或交给 PyMC 自动插补。基础兼容 graded，但只允许 K=2：类别顺序、方向、标签和 C 显式声明，NA 表示整次作答缺失。完整 rank 入口仍拒绝不完整／并列。一般部分排名、triplet 分解的 graded、随机呈现和多维加载另行分期。细节见 [graded 协议](graded_response.md)。

## 3 模型与 block 组合

- **R12 统一 Model。** 用户指定默认 family，并按 block 覆盖。规格、构图、拟合分开；`build()` 返回包含原生 `pm.Model` 的 `BuiltModel`，`fit()` 委托通用推断入口。
- **R13 联合拟合。** 全测验共享一套 person-trait 结构和特质相关参数，各 block 使用各自的响应机制。在同一图中按兼容的族配置、大小、格式与形状批处理。
- **R14 参数共享。** 同族中复用 statement 时，兼容的参数只创建一次。共享身份、最终先验和联合识别约束必须一致；不兼容时明确报错。首版不隐含建立跨族 statement 参数关系。
- **R15 排名与观测事件。** 每个有效 person-block 对应一个事件：rank 为完整排名；pick／mole 对未知顺序边缘化；graded 为二元题组的一次偏好程度类别，使用独立的类别概率机制。完整排名编码基于参考顺序稳定往返；默认选择顺序列表与逐 statement 的 rank 向量明确区分。允许正确的原生类别分布或有 `logp` 与生成方法的实现，不要求始终存储全部 K! 概率。
- **R16 Thurston 类联合似然。** 按用户要求，完整排名分支使用 statement 效用诱导的联合排名概率，保留比较间依赖。原教程的成对 Bernoulli／等级反应不直接等同于该模型，须重新核定均值、协方差、参数映射、识别与数值算法。
- **R17 模型覆盖。** 五个核心系列为 TIRT、GTUM、2PLM-RANK、GGUM-RANK、MGPPM。首轮 graded 目标为 TIRT、GTUM、MGPPM 的 K=2 分支；2PLM-RANK／GGUM-RANK 先声明排名类格式。具体“族 × 格式 × 大小 × 量尺 × 采样器”只登记已验证组合，不因共用 FCData 就宣称所有族支持 graded；扩展模型单独分期。

## 4 先验与题目参数模型

- **R18 用户先验。** 轻量 `Prior` 沿用 PyMC 分布名称和参数语义。全局特质结构与族内题目参数分开配置，不静默自动缩放先验。
- **R19 局部覆盖。** 支持按 block、按 statement 指定兼容的题目参数先验。顺序为：族内置默认值 → 族实例设置 → block 设置 → statement 设置。先解析最终设置，再检查共享参数冲突；预览中显示实际生效的设置及来源。
- **R20 识别优先。** 明确参数角色、支持集、维度及先验作用于自由量还是变换后参数。固定锚点、零和约束、相关矩阵等联合结构不能被逐项覆盖静默破坏。
- **R21 参数协变量。** 后续用 `ParameterModel` 描述题目参数的生成过程。例如由 Likert 估计值、文本或设计特征预测 FC 的载荷大小，并允许题目残差；预测变量、link、系数与残差先验显式配置。
- **R22 一条生成路径。** 同一参数选择直接先验、参数回归、固定标定或工厂中的一种；不能既回归生成又叠加一套重复的直接先验。保留训练变换、尺度、方向及来源。
- **R23 分开处理 Likert 的用途。** 先支持把已获得的 Likert 结果当固定协变量；外部结果构造信息先验、传播外部后验不确定性、联合 Likert–FC 标定分别设计和验证。不默认两种格式的参数相等。

## 5 分析流程与结果

- **R24 可检查的拟合。** 保存解析后的模型规格和实际抽样设置；默认采样器显式选择。PyMC 原生 NUTS 为起点，通过测试后提供 PyMC 发起的 NumPyro/JAX 路径，GPU 为可选运行环境。
- **R25 原生结果。** FitResult 保留标准 DataTree、chain/draw、原始标签、观测映射及元数据。R-hat、ESS、MCSE、图和 PSIS-LOO 复用当前 ArviZ；WAIC 按旧版 ArviZ 0.22.0 实现，DIC 按 R2jags 默认实现，保留原链信息与指标定义。
- **R26 按需预测与相对拟合。** 支持任务对应的预测、FC discrepancy、PPP，以及 PSIS-LOO、DIC、WAIC。共同 log-likelihood 只含响应似然，以实际 person-block 事件为单位；排名和 graded 各用相应概率。DIC 按 R2jags 默认逐链 pV，WAIC 按旧版 ArviZ 的 ddof=0 与 SE 约定；同一组固定抽样逐项数值对照是首版验收条件，不能以独立 MCMC 结果近似相同代替。模型比较要求同一数据、观测事件和目标。
- **R27 基础评分。** 每个核心模型支持已有摘要及已标定题目的新被试评分，输出 trait 后验均值、区间和标签。Fixed calibration only; calibration uncertainty is not propagated. 评分覆盖模型已支持的 PICK／MOLE 及二元 graded，后者保留类别程度信息。新作答默认不重新标定题目。新设计／statement 参数预测另行开发，详见 [评分设计](scoring.md)。
- **R28 模拟与示例。** 提供由显式参数生成规范 FCData 的 Python 模拟入口、小型示例与完整分析教程。数学参考测试保持独立，避免生成器和拟合器共同出错而互相验证。
- **R29 保存和读取。** 保存原生后验、规范表、规格、先验、约束、编码及运行信息；读取结果不隐式恢复可执行图。显式重建后检查兼容性。自定义构建代码由用户另行保存，不 pickle 图或函数。

## 6 可复用组件、测试与文档

- **R30 自定义模型入口。** 保留数据/编码、共享特质、参数生成、响应机制、结果与诊断之间的边界。新族说明大小、格式、参数角色、约束及预测能力；在 PyMC 上创建自身组件，复用通用流程。
- **R31 原生 PyMC 入口。** 用户可访问原生图添加高级结构。自定义先验工厂和参数组件在后续实现；任意图修改不能自动承诺保存后重建或全部诊断功能。
- **R32 分层测试。** 固定参数、归一化、梯度、编码、标签、先验和约束先做确定性检查；prior predictive 在构图后独立检查，不等 MCMC。Posterior MCMC is manual-only; formal tests use mocks and fixed or saved draws. Recovery is optional research validation; reference fixtures verify metrics. 慢任务有固定预算和事先规定判据，不无限重跑随机失败；逐模块文件与判据见 [测试清单](testing.md)。
- **R33 文档和维护。** Sphinx、autodoc/autosummary、numpydoc 生成 API；MyST-NB 执行教程。安装、数据语义、模型方程、识别和使用决策人工组织。每个正式模型有一个完整教程，另有混合大小与混合族示例。
- **R34 包的质量。** CI 检查风格、适当的模型测试、干净环境 wheel 安装、文档和受影响教程；依赖按运行、绘图、存储、文档、开发分组。只承诺经过测试的环境组合。

## 7 以后逐项纳入的教程能力

教程中 TIRT／GTUM 的 graded 先迁移其二元题组定义，MGPPM 的 graded pair 进入基础目标。triplet 内多次 graded 比较不在当前范围。BRB、JRT-TIRT、Thurstonian D-diffusion、M-TCIR、ZG-MUPP 属于第二批模型；反应时、质量标签等额外观测先定义协议，HSSM 等依赖按需安装。

v0.1 实现 MGPPM 排名及二元 graded 从输入到评估、评分和保存的完整分析，包含 DIC／WAIC 数值参考验收。其余模型逐个达到相同流程；列表、新被试评分和 PICK／MOLE 的 NA 保持基础范围。
