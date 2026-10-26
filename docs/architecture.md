# PyFCIRT 包架构草案

日期：2026-10-25  
设计目标：易用、可安装、可维护、可扩展的 forced-choice 心理测量分析包。  
项目名称：**PyFCIRT**；发行名、仓库名和导入名统一为 `pyfcirt`，推荐使用 `import pyfcirt as fc`。仓库：[qipengchen/pyfcirt](https://github.com/qipengchen/pyfcirt)。

采用 `src/` 布局，以 block 作为观测与似然组成单位，以测量族组织统计实现。公开 `Model` 入口将不同大小、不同测量族的 block 组合进同一个原生 PyMC 模型，共享被试特质；少量组合对象连接规格、原生图和 ArviZ 结果。新增测量族主要增加其源码、测试与教程，通用采样和持久化不按模型名称分派。

开发将模型类别与功能分开：首版以 **MGPPM** 同时完成排名和二元 graded 的完整 tutorial 分析，随后其他族复用流程。默认列表、PICK／MOLE 的 NA、新被试评分及 DIC／WAIC 属于首版，指标需要固定抽样的参考数值验收；局部先验、跨族混合和协变量随后推进。

这个包承担数据处理、建模、拟合、评分和预测。Stan 实现、跨后端比较、论文 benchmark 和独立研究 `validation/` 不纳入包仓库的设计。Formal tests cover input validation, numerical likelihoods, graph construction, and mocked sampling. Posterior MCMC checks are optional manual runs outside `tests/`.

## 1 采用哪些 Python 生态设计

### 标准打包布局

采用 PyPA 描述的 `src/` 布局，让测试通过安装后的包导入代码，减少仓库文件意外进入导入路径和漏打包的问题。构建、依赖及工具配置集中在 `pyproject.toml`。CI 同时检查 editable 开发安装和实际 wheel 安装。[PyPA src 布局](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)、[pyproject.toml 指南](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)

### 具体借鉴 Bambi 的哪些设计

Bambi 与 PyFCIRT 的共同点是：为特定统计问题提供方便的建模接口，底层仍由 PyMC 构图和推断。Bambi 的 `Model` 暴露 `build`、`fit`、`prior_predictive`、`predict` 等操作，后端 `PyMCModel` 负责生成原生 PyMC 模型并执行推断。PyFCIRT 可以借鉴这种职责分离。[Bambi Model](https://bambinos.github.io/bambi/api/Model.html)、[Bambi PyMC 后端源码](https://github.com/bambinos/bambi/blob/main/bambi/backend/pymc.py)

具体落实为四点：

1. **统计规格与计算图分开。** `Model` 接收默认测量族和按 block 的覆盖配置；完整的族分配、先验、约束和观测映射进入 `ModelSpec`。联合构建器与各族组件在同一个 `pm.Model` 中定义共享特质和响应，最终返回 `BuiltModel`。
2. **构建、拟合和预测分别调用。** 先检查模型，再采样，再按需执行预测或计算逐观测 likelihood。各步骤共享清楚的输入输出，避免一个入口承担整套分析。
3. **默认先验允许明确覆盖。** 每个模型在 `_config.py` 中声明先验和识别配置，拒绝未知参数名，并保存解析后的完整设置。Bambi 提供先验覆盖和自动缩放；FC 潜变量的尺度与识别紧密相关，因此 PyFCIRT 的默认值须逐模型论证和测试，不直接套用回归中的自动缩放。
4. **保留底层生态的访问入口。** `BuiltModel.model` 允许使用 PyMC 的原生功能；后验保留带维度、坐标的标准结果，供 ArviZ 继续分析。包装层只增加 FC 所需语义。

首版增加一个轻量的 `Model` 组合入口，并借鉴 Bambi 的测量族规格与组件构建边界；不建立公共继承树、自动注册表或多后端框架。Bambi 的公开 `Model` 使用一个 family，按 block 组合多个测量族是 PyFCIRT 自己实现的扩展。FC 的响应机制与识别约束由族模块表达，不将它们强行化约为普通回归的 link 或公式。

上述目录、`ModelSpec`／`BuiltModel`／`FitResult` 是 PyFCIRT 的设计选择，借鉴的是 Bambi 的边界和使用流程，并非照搬其内部结构。实施时核对所参考的 Bambi 版本；若实际复用源码，需另行遵守其许可证与署名要求。

### 其他相近实践

PyMC-Marketing 的 ModelBuilder 区分模型配置、采样配置、构建和保存加载，可作为职责边界的另一项参考。[PyMC-Marketing ModelBuilder](https://www.pymc-marketing.io/en/stable/api/generated/pymc_marketing.model_builder.ModelBuilder.html)

PyMC-Marketing 的 MaxDiff 数据准备接口已有与本项目相近的实践：从 respondent、task、item 长表生成带 mask 的数组和索引映射。可以参考这种预处理边界，但 MaxDiff 的概率模型不能据此视为心理测量 FC 模型。[MaxDiff 数据适配](https://www.pymc-marketing.io/en/1.1.0/api/generated/pymc_marketing.customer_choice.maxdiff.prepare_maxdiff_data.html)

### ArviZ 结果协议与版本基线

新包按 PyMC 6、PyTensor 3、ArviZ 1 的方向设计，结果容器采用符合 ArviZ schema 的 `xarray.DataTree`。PyMC 6 的发行说明明确记录了从 `InferenceData` 到 `DataTree` 的迁移；通用组、维度和坐标遵循 ArviZ 规范。[PyMC 6 发行说明](https://github.com/pymc-devs/pymc/releases/tag/v6.0.0)、[ArviZ schema](https://python.arviz.org/en/stable/schema/schema.html)

依赖范围是实施起点，正式发布前须在实际环境中测试并确定下界。首版只维护一套主要版本组合。若以后需要支持 PyMC 5 与 ArviZ 0.x，再集中增加兼容层；各模型不分别维护两套结果格式。

### PyMC 和 ArviZ 的复用边界

PyFCIRT 的公共功能围绕 FC 数据与统计语义组织；通用计算能力沿用生态接口：

- **构图与推断：** 使用原生 `pm.Model`、PyTensor 运算以及 PyMC 的采样和预测接口。首版明确固定默认采样器，其他采样器逐项测试后再声明支持，不自建采样器分派框架。[PyMC sample](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html)
- **结果与标签：** 保存原生 `DataTree`，使用命名维度和坐标保留被试、条目、区块与 trait 标签，便于直接交给 ArviZ／xarray 处理。[ArviZ schema](https://python.arviz.org/en/stable/schema/schema.html)
- **诊断与绘图：** R-hat、ESS、MCSE、后验图和 PSIS-LOO 使用当前 ArviZ。DIC 按 R2jags 默认逐链算法独立实现；WAIC 按旧版 ArviZ 0.22.0 实现。两者复用标准 likelihood，并按固定样本数值对照；细节见 [评估规格](model_comparison.md)。
- **预测与存储：** 复用 PyMC 的先验／后验预测，以及 xarray 的标准存储。模型负责可生成的响应机制、兼容的形状坐标和正确的观测单位；这些统计约定不能由通用工具代替。[PyMC 后验预测](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample_posterior_predictive.html)

可用性的目标是：同一份规范数据使用不同模型时有一致流程；用户仍可直接访问原生 PyMC 模型和标准后验结果；新增模型主要增加自己的实现、测试和教程。

## 2 整体结构和依赖方向

```text
FCDesign + block 作答表（排名列表／graded 类别）
      |
      v
FCData                  规范化输入、标签与完整性检查
      |
      v
Model                   默认测量族 + 按 block 覆盖 + 先验与约束
      |
      +---- 共享 theta、trait 坐标与特质相关结构（只定义一次）
      |
      +---- 各测量族组件：定义参数、按大小/格式分组、构建观测
      |
      v
BuiltModel              一个原生 pm.Model + ModelSpec + FCData
      |
      v
inference.fit           一次联合采样，返回 FitResult
      |
      +---- predictive  生成响应、输出顺序列表／graded 类别
      +---- scoring     FC 特有评分
      +---- results     保留标准后验，按需重组块级 log likelihood
      +---- io          保存与读取
      +---- ArviZ       通用诊断、模型评估和绘图
```

`data` 不导入测量族；`model.py` 组织共享潜变量与组件构建；各族实现约定的准备和构建能力。`inference`、`results`、`predictive`、`io` 读取公共规格和组件清单，不按 TIRT、GGUM 等名称分支。模型模块不相互导入私有实现。

block size 由设计决定，测量族由统计规格决定。一次 person-block 响应贡献一项完整观测概率；相同族、大小和兼容配置的 block 可批量计算，不要求每个 block 单独创建模型或独立采样。详见 [block 组合设计](block_models.md)。

只在数学机制完全相同且已有验证时提取公共概率函数，不按相似名称共享实现。

## 3 建议目录结构

以下是逐步实现的目标结构。树中的 TIRT 是模块组织示例，第一版实际创建 `models/mgppm/`，分别处理排名和二元 graded 的类别机制与识别。只为真实职责创建文件，逐模块任务见开发计划。

```text
pyfcirt/
├── pyproject.toml                     # 构建、元数据、依赖、pytest 和 Ruff
├── README.md                          # 安装、快速示例、支持能力与文档链接
├── LICENSE                            # 项目确定的许可证
├── CHANGELOG.md                       # 用户可见变化与迁移说明
├── CONTRIBUTING.md                    # 开发环境、测试和模型贡献流程
├── .gitignore
├── .github/
│   └── workflows/
│       ├── ci.yml                     # 格式、测试与 wheel 安装检查
│       ├── docs.yml                   # API 文档构建、教程执行与站点发布
│       └── release.yml                # 后续发布流程，使用受控的 release 触发
├── src/
│   └── pyfcirt/
│       ├── __init__.py                # 少量公共入口和版本信息
│       ├── py.typed                   # 发布类型信息
│       ├── model.py                   # 公开 Model 与共享潜变量的联合构建
│       ├── priors.py                  # Prior 描述、按 ID 覆盖和工厂适配
│       ├── parameters.py              # 参数生成、协变量回归及共享约束
│       ├── _model.py                  # ModelSpec、BuiltModel 与组件元数据
│       ├── data/
│       │   ├── __init__.py            # 导出 FCDesign、FCData
│       │   ├── _schema.py             # 长表字段、FCDesign 和 FCData
│       │   ├── _validation.py         # 通用记录完整性检查
│       │   ├── _indexing.py           # 标签映射、坐标和索引约定
│       │   └── _encoding.py           # 排名模式编码/解码，支持不同 block size
│       ├── models/
│       │   ├── __init__.py            # 显式导出测量族，无自动扫描
│       │   └── tirt/
│       │       ├── __init__.py        # 导出 TIRT、模拟和配置类型
│       │       ├── model.py           # TIRT 规格及其组件构建
│       │       ├── _config.py         # TIRTPriors、TIRTIdentification
│       │       ├── _data.py           # TIRT 分组、排序事件与约束索引
│       │       ├── _likelihood.py     # 完整排序概率与对应生成机制
│       │       └── simulation.py      # 按指定参数生成 FCData
│       ├── inference.py              # 通用 fit 和采样记录
│       ├── results.py                # FitResult、观测映射和块级 likelihood 视图
│       ├── model_comparison.py       # DIC／WAIC 的参考兼容计算与元数据
│       ├── predictive.py             # 通用 prior/posterior predictive
│       ├── checks.py                 # FC 专属 discrepancy 与 PPP
│       ├── scoring.py                # 已有被试摘要与新被试评分
│       └── io.py                     # 保存、读取及兼容检查
├── tests/
│   ├── conftest.py                    # 小型共享 fixture 和固定种子
│   ├── data/
│   │   ├── test_schema.py
│   │   ├── test_indexing.py
│   │   └── test_encoding.py
│   ├── models/
│   │   └── test_mgppm.py              # 首版：likelihood、梯度、约束、模拟
│   ├── integration/
│   │   └── test_mgppm_workflow.py     # Supplied-trace checks; no sampling
│   ├── recovery/
│   │   └── test_mgppm_recovery.py     # Deterministic design checks; no sampling
│   ├── test_model.py                  # 共享参数、构图与 fit 参数转发
│   ├── test_priors.py                 # 首版已实现的先验配置与约束
│   ├── test_prior_predictive.py       # 构图后、后验拟合前的独立前向检查
│   ├── test_predictive.py
│   ├── test_checks.py                 # discrepancy／PPP 的固定输入检查
│   ├── test_model_comparison.py       # DIC／WAIC 固定抽样参考对照
│   ├── test_scoring.py
│   ├── test_io.py
│   └── reference/                    # 隔离环境直接比较 R2jags／旧 ArviZ
└── docs/
    ├── conf.py                       # Sphinx、API 生成和 notebook 执行配置
    ├── index.md
    ├── architecture.md               # 职责边界、扩展方式与维护约定
    ├── quickstart.md
    ├── data_format.md
    ├── block_models.md                # 混合大小、测量族与完整块级似然
    ├── core_api.md                    # 核心对象、默认大小规则和自定义先验
    ├── parameter_models.md            # 局部先验、参数协变量和扩展边界
    ├── models/
    │   └── tirt.md                    # 方程、先验、识别、适用格式与限制
    ├── tutorials/
    │   └── tirt_workflow.ipynb         # 完整工作流；教程的单一源码
    ├── guides/
    │   ├── prediction.md
    │   └── persistence.md
    ├── contributing/
    │   └── adding_model.md
    └── api/
        ├── index.rst                 # 手写公开对象列表，驱动 autosummary
        └── generated/                # 自动生成，不手改、不提交
```

`models/tirt/` 将模型专属文件放在一起，避免新增模型时同时修改全局 adapters、likelihoods 和 simulations 中的多个分派函数。`data/` 只共享真正通用的数据约定。

单个模型初期很小时，可以先将 `_config.py` 和 `_data.py` 中的少量代码放在 `model.py`，出现独立职责后再拆开。上面的完整组织方式适合已经具有非平凡识别和排名编码的模型，无须为目录齐全而创建空文件。

`tests/`、`docs/` 是开发仓库内容。完整教程集中在 `docs/tutorials/`，不再维护另一份相同的 `examples/` notebook。用户安装的 wheel 主要包含 `pyfcirt/` 运行库、类型标记及发行元数据，不包含测试、大型输出或研究脚本。类型标记遵循 [PEP 561](https://peps.python.org/pep-0561/)。

这里展示目标结构；按实际职责逐步创建文件，不为目录齐全增加空模块。

## 4 数据层负责什么

默认输入采用一行一次 person-block 的作答表与题组设计表。`FCDesign` 保存设计，`FCData` 组合设计与作答，以普通 DataFrame 为底层，不继承 DataFrame。排名类内部转规范长表，graded 保留单次类别事件；字段与快捷入口见 [输入协议](data_format.md)。

- `responses`：默认字段为 `person_id, block_id, response`。排名 response 是选择顺序列表，如 `[2,3,1]`；PICK／MOLE 同长列表保留 None／NA。显式兼容 `statement_id, rank` 长表，二者不混淆。graded 为量尺中的单个类别标量。
- `design`：普通单维 statement 使用 `block_id`、`position`、`statement_id`、`trait`、`keying` 的设计长表；内部拆分 statement 信息与题组成员信息，避免重复维护维度定义。
- `statement_features`：可选，一行一个 statement ID。协变量值属于设计数据，列选择、link、变换及参数回归属于统计规格。

题组通过 statement 间接关联维度，不要求每个题组只有一个维度。`keying` 表示内容方向，不直接反转原始排名。除表格入口外，拟提供按维度分组的 statement 字典和按题组列出的 statement 列表，自动生成同一种设计表。

共同规则包括键的唯一性、未知标签、选择集合、任务合法性、编码、量尺和 NA。rank／pick／mole／graded 由设计默认值与按 block 覆盖确定。前三者以兼容排名概率处理未知顺序，保留完整选择集合；graded 仅 K=2，C、类别顺序与方向显式声明，使用相应族的类别机制。结构性 NA 不触发 PyMC 自动插补；真实未答／未施测由可选状态表标明。详见 [输入协议](data_format.md)与 [graded 协议](graded_response.md)。

同一 statement 可以出现在多个区块；statement 身份与呈现位置分开。`FCData` 不强制模型使用同一种内部数组，也不根据当前文件恰好出现的记录猜测全部区块结构。

对象构造时复制并规范化输入，避免外部修改原 DataFrame 后使已建模型失效。若提供只读语义，需要受控访问或复制；`dataclass(frozen=True)` 本身不会冻结底层 DataFrame 或数组。

测量族专属限制由 `models/<model>/_data.py` 检查。block size 从成员关系计算。默认检查等大小；显式设置 `allow_variable_block_size=True` 后接受不同大小，且继续检查其余结构约定。测量族分配保存在 `Model` 规格中，不重复到设计表或作答行。支持的大小、格式、trait 连接结构和 statement 复用方式由对应族明确检查。

## 5 三个内部组合对象连接公共流程

### ModelSpec

保存可序列化的统计规格：

- 联合模型格式版本，以及每种测量族的统计实现版本。
- 全部已解析的先验及其来源、显式局部覆盖、识别约束和参数生成方式。
- 参数回归使用的特征、设计列顺序与训练变换；特征值和数据指纹随输入保存。
- 每个 block 的测量族分配、已解析配置、响应格式、likelihood 类型与预测目标。
- 各计算组的 observed RV、编码版本及稳定 obs_id 到 person-block 的映射。
- 观测单位与潜变量条件化约定，完整排名统一为 person-block 事件。
- 变量角色，例如 trait 变量、响应变量和题目参数变量的名称。
- 数据编码版本及规范化输入的指纹。

规格只使用可保存的数据结构。族模块定义可配置的参数角色、支持集、尺度和识别约束；轻量 `Prior` 保存 PyMC 分布名称和参数，构建时才创建随机变量，不建立另一套分布语言。全局先验放在 `Model(priors=...)`，族参数先验放在 `TIRT(priors=...)` 等规格中，block 与 statement 通过显式 ID 规则覆盖；参数回归由独立 `ParameterModel` 规格定义；自定义工厂另存来源和配置，构建代码由用户保留。覆盖、共享冲突和工厂边界见 [核心对象与先验配置](core_api.md)。

### BuiltModel

组合原生 PyMC 模型、完整规格和规范化输入。`Model.build()` 返回它，图中包含全部组及其共享特质；通用流程通过组件清单获取元数据。用户仍可直接访问 `.model` 中的原生 `pm.Model`。

### FitResult

组合原生后验容器、模型规格、规范化输入与有效采样设置。原始后验只保存一份，摘要按需计算；活动计算图不承担长期文件格式的职责。

以下是边界示意，省略字段校验和序列化实现：

```python
from dataclasses import dataclass

import pymc as pm
import xarray as xr

from pyfcirt.data import FCData


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_version: str
    settings: dict
    components: list[dict]
    observation_map: list[dict]
    variable_roles: dict
    observation_unit: str
    likelihood_kind: str
    data_fingerprint: str


@dataclass(frozen=True)
class BuiltModel:
    model: pm.Model
    spec: ModelSpec
    data: FCData


@dataclass
class FitResult:
    idata: xr.DataTree
    spec: ModelSpec
    data: FCData
    sampler_settings: dict
```

`ModelSpec` 位于 `_model.py`，`FitResult` 位于 `results.py`，二者只依赖公共数据结构。字典字段实施时应补充类型、复制与 JSON 兼容检查。

## 6 测量族模块提供的能力

每个族提供轻量规格对象，例如 `TIRT(priors=..., identification=...)`。用户先验覆盖族默认值；固定参数和识别约束单独管理，不能靠指定先验静默覆盖。它描述的是测量机制，不是一个已经独立拟合的全测验模型，也不继承 PyMC 分布或 Bambi 内部类型。

组件构建的最小职责包括：

1. 检查所分配 block 的大小、响应格式、设计与识别条件，准备数组及分组。
2. 接收联合模型中已有的共享 theta 和坐标，为该族管理的 statement 解析一次参数生成方式，可使用直接先验、参数回归或受控工厂；识别约束仍由族定义。
3. 在同一个 PyMC 上下文中，按兼容大小和配置批量创建 observed RV，返回变量名称、类别编码及 person-block 映射。
4. 提供同一统计机制下的生成函数，供模拟及先验／后验预测复用。

这些职责可通过短函数和小对象实现，不先建立复杂基类。自定义族以对象参与同一构建流程，遵循参数角色、观测映射和生成能力契约；可复用公共参数生成和数学含义一致的响应组件，见 [参数与自定义组件](parameter_models.md)。构建过程不启动 MCMC、不写文件、不修改全局 PyMC 或 PyTensor 设置，也不为每个计算组重新创建 theta 或重复的 statement 参数。

完整排名采用模式类别观测。小类别空间可使用 `pm.Categorical`；需要自定义已观测类别的 logp 时，提供对应生成机制，并保持相同支持集和编码。`_likelihood.py` 使用数值稳定、可检验梯度的 PyTensor 图；Thurston 类完整排序概率须按相应联合效用模型实现。[PyMC CustomDist](https://www.pymc.io/projects/docs/en/stable/api/distributions/custom.html)

比较对可以作为内部差分计算，但不能把相关的边际比较概率相乘后称为完整排序似然。排名类别与大小的映射、区域概率的精度和梯度、模型识别及性能要求见 [block 组合设计](block_models.md)。

`simulation.py` 保留显式参数到 `FCData` 的模拟入口。生成和拟合可以复用机制，正确性测试仍使用独立数学参考，避免共同错误相互抵消。

## 7 通用拟合和分析接口

`inference.fit(built, **sample_kwargs)` 调用 PyMC，记录实际使用的 seed、采样器、chains、tune 和 draws，返回 `FitResult`。这里不出现 `if model_id == ...`。

采样参数尽量沿用 PyMC 的名称；只为 FC 特有行为增加参数。默认采样器显式选择，防止安装额外依赖后悄悄改变执行路径。其他采样器只有通过对应测试才列为支持能力。

运行记录保存规范化后的基本类型元数据，而非原样序列化 `sample_kwargs`。整数 seed 直接保存；RNG 对象记录抽样前的生成算法和可序列化状态。自定义 step、callback 等对象记录类型与构建标识，构造代码由用户保存；无法完整编码时明确标记自动复现的限制。

后验的逐观测 likelihood 可以由显式步骤请求，例如调用 PyMC 的 `compute_log_likelihood`。不同计算组的观测通过稳定 obs_id 按需合并成统一 person-block 视图，避免将多个组误当成不同统计单位或重复计数。不要在拟合调用中自动执行全部预测、LOO、拟合指标和大规模派生计算；由分析者按需请求。

`Model.fit()` 是 `Model.build()` 与通用 `inference.fit()` 的便捷组合；构建后修改统计配置须重新构建，不在旧图上悄悄换族。使用方式草案：

```python
import arviz_stats as azs
import pyfcirt as fc

from pyfcirt.models import TIRT, TwoPLRank

design = fc.FCDesign.from_frame(design_table, response_format="rank")
data = fc.FCData.from_frames(responses=responses, design=design)
model = fc.Model(
    data,
    family=TIRT(priors=priors, identification=identification),
    block_families={"B3": TwoPLRank()},
)
built = model.build()

fit = fc.fit(
    built,
    chains=4,
    tune=1000,
    draws=1000,
    random_seed=2026,
    nuts_sampler="pymc",
)

summary = azs.summary(fit.idata, var_names=["theta"])
ppc = fc.posterior_predictive(built, fit, random_seed=2027)
fc.save_result(fit, "analysis-result")
```

参数数值仅用于展示调用方式，实际抽样量依据模型诊断决定。示例中的 `pyfcirt` API 是设计稿；实施后应由可执行示例和 API 测试维护。

ArviZ 1 生态的统计接口可直接使用 `arviz_stats`，绘图使用 `arviz_plots`。包只增加 FC 专属结果，例如按被试和 trait 标签组织的评分摘要。通用 R-hat、ESS、MCSE 和后验图由 ArviZ 提供。[ArviZ summary](https://python.arviz.org/projects/stats/en/stable/api/generated/arviz_stats.summary.html)

## 8 预测和新被试评分的边界

`predictive.py` 提供分开的 prior 与 posterior predictive 入口。prior predictive 从 BuiltModel 直接生成，不需要 FitResult，在后验采样前检查参数支持、识别、响应、量尺和标签；测试单列于 `tests/test_prior_predictive.py`。posterior predictive 才接收后验结果并检查变量与规格匹配。两者均按任务解码为顺序列表或 graded 原类别；排名类另可导出长表。结果使用标准组并明确来源，不覆盖原拟合。

PyMC 6 调整了预测时变量重采样的语义，包装层应按目标版本明确设置 `sample_vars` 等参数，而不是把 `var_names` 误当成重采样控制。[PyMC 6 预测接口变更](https://github.com/pymc-devs/pymc/releases/tag/v6.0.0)

`scoring.py` 从基础阶段就包括已有摘要和新被试评分。trait 变量名由 ModelSpec 指定，标签由 xarray 保留；评分覆盖模型声明的排名类格式及二元 graded，保留偏好程度和 NA 语义。Scoring uses one fixed calibration point, records its source, and does not propagate calibration uncertainty; see [scoring](scoring.md).

每个核心族提供接受已给定题目参数的响应计算组件，通用评分层用它为新被试构建 theta 与响应，复用数学而不重复整个标定器。不能将 `pm.set_data()` 本身当作全部评分算法。新设计预测和新 statement 参数预测另外开发；确需模型特定准备时再增加 `prediction.py` 等短模块。

## 9 新增 GGUM RANK 测量族的具体改动

新增族先达到单族拟合、其声明支持的基础任务和新被试评分；支持 graded 的族另检验 K=2 类别机制。进入混合层后再验证跨族组合。其响应组件使用上下文提供的共享 theta；代码集中在族目录：

```text
src/pyfcirt/models/ggum_rank/
├── __init__.py                # 导出 GGUMRank 和配置类型
├── model.py                   # 族规格、参数与组内观测构建
├── _config.py                 # GGUMRankPriors、GGUMRankIdentification
├── _data.py                   # 分组、索引及能力检查
├── _likelihood.py             # 完整排名 logp 与生成机制
└── simulation.py              # 显式参数到 FCData
```

### 必须增加或更新

1. **实现族目录。** 从公开方程定义概率、先验和识别，明确支持的大小与格式。使用已有共享特质，返回 observed RV 与完整的观测映射，不重新建立全测验模型。
2. **新增 `tests/models/test_ggum_rank.py`。** 检查固定参数 logp、梯度、全部合法模式归一化、边界、形状、约束及模拟；保留小型独立参考。
3. **新增或扩展工作流测试。** 单族阶段即检查小 MCMC、支持任务及 NA、预测、PPP、逐观测 likelihood、评分与保存，和完整教程对齐；进入混合层再增加跨族联合检查。不能只检查采样运行成功。
4. **新增相关恢复测试。** 先处理识别不定性，再检查参数、已有及新被试评分的恢复；混合层采用明确的已知参数场景。多种子慢测试使用 marker。
5. **更新 `src/pyfcirt/models/__init__.py`。** 显式导出 `GGUMRank`，不自动扫描目录或修改通用采样器。
6. **新增模型说明与可执行教程。** 解释方程、假设、识别、支持范围和失败模式；教程导入包函数，公开对象补齐类型标注与 docstring。
7. **验证保存读取。** 覆盖每个 block 的族分配、先验、约束、类别编码、标签和观测映射，检查显式重建后的兼容性。
8. **更新文档导航、API 列表和 CHANGELOG。** 将新族加入共享契约测试，严格构建文档并执行对应教程。

### 根据真实需求才修改

新格式需要更新协议、编码版本和测试。数学相同的机制可提取私有函数并保留回归检查。新被试评分和声明的基础任务进入核心族验收；TIRT／GTUM／MGPPM 的二元 graded 在基础目标内，其他族不能自动宣称支持。额外格式与依赖按需求加入。

### 通常不需要修改

`inference.py`、`results.py`、`predictive.py`、`io.py` 和顶层 `__init__.py`。联合构建器通过一致的组件能力组合新族，通用流程通过组件清单和观测映射工作。若缺少公共元数据，需要演进契约并验证已有族，不添加按族名称分派的通用流程。

## 10 保存和读取

建议保存一个目录：

```text
analysis-result/
├── posterior.nc              # 原生 DataTree，保留 chain、draw 和各结果组
├── model.json                # 族分配、任务、配置、编码版本、组件与观测映射
├── sampler.json              # 规范化运行记录与软件版本，不含原始 Python 对象
├── responses.json            # 默认 block 作答，保留列表、类别、NA 和类型
├── blocks.json               # 同样保留表结构和标签
├── statement_features.json   # 可选特征表，保留字段、标签和来源元数据
└── administration.json       # 可选施测状态表，保留真实未答／未施测信息
```

表结构可用 pandas table-oriented JSON 保存，再按 schema 恢复列表、原标签、类别及 NA。输入 encoding、参考顺序、任务、graded 量尺、完整选择集合和可选状态表一并保存；不将顺序列表误读为 rank 向量。NetCDF 依赖通过 `io` extra 声明，大规模需要再考虑 Zarr。后验存储使用 xarray／ArviZ 生态。

`load_result()` 只恢复结果、规格、输入和标签，不隐式加载可执行图。要继续预测，用户根据已保存的各 block 族分配与已解析设置显式重建 `Model`，再执行规格、数据指纹、组件、编码及变量兼容检查。`io.py` 不承担测量族重建的可执行分派。

自定义修改 `.model` 后，普通模型规格不能自动描述任意计算图。用户需要记录其构建代码和修改后的规格；包不能保证自动重建未记录的自定义图。

格式版本与模型统计实现版本分别管理。未知主格式版本给出明确错误；仅在存在实际迁移需求时增加迁移函数。保存默认不覆盖已有结果，使用临时位置完成写入后再提交完整目录，避免留下半个可加载结果。

## 11 打包配置和依赖管理

建议选一个构建后端，首版使用 Hatchling。`pyproject.toml` 的起始形态如下；依赖下界和完整元数据须在发布准备时补齐并验证。

```toml
[build-system]
requires = ["hatchling>=1.26,<2"]
build-backend = "hatchling.build"

[project]
name = "pyfcirt"
version = "0.1.0"
description = "Bayesian forced-choice IRT models built with PyMC"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
  "pymc>=6,<7",
  "pytensor>=3,<4",
  "arviz>=1,<2",
  "arviz-stats>=1,<2",
  "numpy",
  "pandas",
  "xarray",
]

[project.optional-dependencies]
io = ["h5netcdf", "h5py"]
plots = ["arviz-plots[matplotlib]>=1,<2"]
test = ["pytest"]
release = ["build", "twine"]
docs = ["sphinx", "myst-nb", "numpydoc", "ipykernel"]
lint = ["ruff"]

[tool.hatch.build.targets.wheel]
packages = ["src/pyfcirt"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--import-mode=importlib --strict-markers"
markers = [
  "slow: subprocess and supplied-trace checks without MCMC",
  "reference: direct checks in isolated R2jags / legacy ArviZ environments",
]
```

项目直接导入的第三方库应列为直接依赖，即使 PyMC 暂时也会安装它。通用开发工具、文档和可选绘图存储依赖单独声明。为了可复现开发与 CI，可以提交环境锁文件；发布库的依赖范围仍由兼容测试决定，不把维护者环境的全部精确锁定版本强加给用户。

版本元数据由 `pyproject.toml` 单一提供，运行时可通过 `importlib.metadata` 读取。顶层导入只公开少量稳定对象，不运行采样、不读本地结果、不访问网络。联合模型入口为 `pyfcirt.Model`，测量族通过 `pyfcirt.models` 提供；底层 `_data.py` 和 `_likelihood.py` 属于可调整的私有实现。[Python 发行元数据接口](https://docs.python.org/3/library/importlib.metadata.html)

开发与构建方式：

```shell
python -m pip install -e ".[test,docs,lint,io,plots,release]"
python -m pytest -m "not slow and not reference"
python -m pytest -m "slow and not reference"
python -m pytest -m reference
python -m build
python -m twine check dist/*
```

正式发布前补齐确定的许可证、作者、项目链接及 Python 分类信息。正式发行时使用已确定的 `pyfcirt` 名称创建项目。

## 12 测试和发布检查

首版的必要／暂缓测试、Bambi 源码依据和执行层次以 [测试计划](testing.md) 为准。只新增 pytest 作为测试工具；不强制 pytest-mock／pytest-cov 等插件。参数传递可替代采样，真实 NUTS 仅通过独立手动脚本运行，并复用拟合结果；正式 pytest 不包含后验 MCMC。没有实际采样的用例不能标记为推断验证。

DIC／WAIC 通过标准库 subprocess 直接调用隔离环境的 R2jags／ArviZ 0.22.0，对同一固定输入做数值比较。它们不进入运行依赖，也不要求重新运行 JAGS 模型；参考测试首次实现、相关变更和发布前真实执行，普通环境可明确跳过。小型固定公式检查仍在日常 CI 运行。

Formal checks cover fixed-parameter likelihoods/gradients, encoding, labels, identification, generation, prediction and observation probabilities. Posterior MCMC and recovery are optional manual checks outside pytest.大型论文模拟与外部 Stan 对照仍属于独立研究工作。

日常 CI 运行快速测试和小 MCMC；修改模型时还运行相关恢复测试。完整多种子恢复可以作为独立慢测试任务，不必让格式检查等待完整实验。诊断、误差容差和随机种子预先定义，失败保留证据，不用无限重跑取得通过结果。

维护检查包括：

- Ruff 格式和静态检查；公共函数具有类型标注和 NumPy 风格 docstring。
- 受支持 Python 版本及依赖组合的测试矩阵；首版可以从少量明确组合开始。
- 构建 wheel 和 sdist，在干净环境安装 wheel，从仓库外导入并运行关键工作流测试。
- 检查 `py.typed` 和必要资源实际包含在 wheel 中，开发路径和未声明依赖没有掩盖漏打包。
- 文档构建及模型示例执行；每个测量族有完整示例，另有混合大小及混合族的联合工作流。
- 共享特质和 statement 参数只创建一次；每个有效 person-block 在 likelihood 中出现一次，分组与逐块参考一致。
- 保存读取后，后验数值、组、维度、标签、约束设置和字段类型一致；错误或不兼容输入有清楚消息。

`src/`、包外测试目录、editable 安装及 `importlib` 导入模式也符合 pytest 的新项目实践。[pytest 集成建议](https://docs.pytest.org/en/stable/explanation/goodpractices.html)

统计诊断、图表、PSIS-LOO 和标准存储使用 PyMC／ArviZ／xarray。当前 ArviZ 已移除 WAIC，因此按用户指定增加短的旧版参考兼容实现，DIC 同样保持小而独立；它们不扩张为另一套通用统计框架。[ArviZ 迁移说明](https://python.arviz.org/en/stable/user_guide/migration_guide.html)

## 13 文档自动生成方案

采用 **Sphinx + autodoc／autosummary + numpydoc + MyST-NB**。API 参考可以从源码生成，教程可从 notebook 执行生成；解释统计含义和使用决策的内容仍需要作者编写。普通 `#` 注释通常不会成为 API 文档，公开对象应使用 docstring。[Sphinx autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)、[autosummary](https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html)

### 哪些维护在源码中

公开函数和类的 docstring 使用 NumPy 风格，说明参数、返回值、维度与坐标、异常、必要的统计语义和简短示例。类型标注提供程序结构信息，docstring 解释使用者需要理解的约定。完整模型方程放在模型说明页，API 文档链接过去，避免多处维护同一推导。

`docs/api/index.rst` 只列出稳定公开对象，Sphinx 生成对应页面。例如：

```rst
API reference
=============

.. autosummary::
   :toctree: generated/

   pyfcirt.FCDesign
   pyfcirt.FCData
   pyfcirt.Prior
   pyfcirt.ParameterModel
   pyfcirt.fit
   pyfcirt.Model
   pyfcirt.models.TIRT
```

新增模型时，主要补好源码 docstring，并将公开入口加入列表。`docs/api/generated/` 在构建时产生，不手改、不提交。[numpydoc](https://numpydoc.readthedocs.io/en/latest/)

### 哪些由作者编写

安装与快速入门、长表数据协议、模型假设与识别、诊断与失败模式、预测边界和贡献指南需要人工组织。生成工具无法从函数签名推断模型为什么成立、何时不适用，或用户应如何作出分析决策。

每个模型在 `docs/tutorials/` 保留一个完整 notebook，导入已测试的模型与数据函数。MyST-NB 执行并将其纳入文档站点，同一份源码也可供用户下载运行；不再复制一份教程到 Markdown。[MyST-NB 执行与缓存](https://myst-nb.readthedocs.io/en/stable/computation/execute.html)

### 最小配置与构建

`docs/conf.py` 的起点如下，主题和导航再按需补充：

```python
project = "PyFCIRT"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "myst_nb",
]
autosummary_generate = True
numpydoc_show_class_members = False
exclude_patterns = ["_build"]

nb_execution_mode = "cache"
nb_execution_allow_errors = False
nb_execution_raise_on_error = True
nb_execution_timeout = 300
```

`index.md` 的 toctree 纳入快速入门、模型页、教程及 `api/index`。构建前安装包与 `docs` extra，并确认 notebook 的 Python kernel 使用同一环境。autodoc 导入实际安装的包；不通过修改 `sys.path` 或模拟核心依赖来掩盖安装问题。构建命令：

```shell
python -m sphinx -n -W --keep-going -b html docs docs/_build/html
```

本地编辑时可复用缓存；固定随机种子和环境版本以便复查。MyST-NB 的缓存主要跟踪 notebook 代码与元数据，不能自动证明包源码、依赖或外部数据未变。模型或公共流程变更时，CI 强制执行受影响教程；发布检查在干净环境中强制执行全部正式教程：

```shell
python -m sphinx -n -W --keep-going -b html -D nb_execution_mode=force docs docs/_build/html
```

300 秒是示意性的单 cell 超时预算，实施时按实际运行设置；教程使用经过诊断的受控示例，大型恢复与 benchmark 另行运行。预期错误示例须逐 cell 明确标记，意外执行错误和构建警告应使 CI 失败。未来包仓库忽略 `docs/_build/`、`docs/api/generated/` 和 `.jupyter_cache/` 等生成物；源码、配置和教程纳入版本控制。

## 14 开发协作与仓库归属

当前仓库由个人账号 `qipengchen` 持有，合作者通过各自账号参与。仓库归属不会改变 Python 导入名；`import pyfcirt` 与 GitHub 用户名无关。

若项目已确定为课题组长期共同维护的资源，可直接创建课题组的 GitHub Organization，由组织持有 `pyfcirt` 仓库，成员使用各自个人账号登录并分配权限。无需为课题组再注册一个多人共用的个人账号。组织可使用 GitHub Free，具体功能随仓库可见性和方案而异。[GitHub 账号与组织](https://docs.github.com/en/get-started/learning-about-github/types-of-github-accounts)

先在个人账号开发，后续也可以转入组织，提交历史、issue、PR 和 releases 等随仓库保留。迁移时更新项目链接、本地远程地址以及文档部署或发布集成，逐项确认可用。[GitHub 仓库转移](https://docs.github.com/en/repositories/creating-and-managing-repositories/transferring-a-repository)

日常协作保持简单：

- 通过小而完整的改动推进功能，提交说明解释行为变化；多人协作时用 PR 讨论和检查。
- CI 覆盖格式、快速测试、小型推断流程及必要文档执行；模型变更运行相关恢复测试。
- 保留安装、贡献和版本变化说明，使后来加入的维护者能搭建环境、定位代码、验证改动。
- 每加入一个新模型，检查新增代码能否集中在模型模块，以及旧模型的使用方式是否保持兼容。确有公共契约变化时，统一更新说明与现有模型测试。

## 15 首版实施顺序

具体阶段、范围和验收以 [开发计划](roadmap.md) 为准；全部需求见 [需求清单](requirements.md)。核对本地 OSF 共享材料后的调整依据见 [教程迁移清单](tutorial_migration.md)。

1. v0.1 按 V1-01–V1-16 搭建模块，先实现 **MGPPM 的排名与二元 graded**；score、截距识别和先验分别定义。构图后先完成 V1-07a prior predictive，再接后验采样，测试落点见 [MGPPM 测试清单](testing.md)。
2. 两分支分别完成诊断、估计、预测、PPP、逐观测 likelihood、PSIS-LOO／DIC／WAIC、评分、保存续接及完整 notebook。V1-11 包含 R2jags 和 ArviZ 0.22.0 的固定抽样数值验收。
3. 再逐个接入 2PLM-RANK、GGUM-RANK、TIRT、GTUM，复用并检验相同流程；不强加未定义的 graded。Thurston 数学审查可提前开展。
4. 五个核心模型达到共同分析流程后，再推进局部先验、跨族联合、参数协变量与自定义组件。
5. 第二批扩展模型单独排期，沿用同一能力契约，不把新增模型本身当成高级功能。

支持模型、响应格式和新被试评分的清单只登记已经实现并测试的能力。本文作为架构草案，具体 API 名称在实施和使用测试后确定。
