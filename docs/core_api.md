# 核心对象与先验配置草案

Implementation status: FCDesign, FCData, MGPPM, Model/Prior, prior prediction and the fitting/result/prediction interfaces are implemented. See the [data guide](data.md) and [model guide](model.md). The broader configurations below remain design proposals; local priors, covariates and mixed-family fitting are not implemented.

本阶段只明确架构，不实现运行库。核心公开对象为 `FCDesign`、`FCData`、`Model`；另用一个轻量 `Prior` 描述先验。以下 API 为待实现方案，参数名称随统计参数定义和使用测试再确定。

## FCDesign 管理测验结构

保存 statement 信息、题组成员关系、维度映射、方向、响应格式，以及可选的 statement 特征表；不保存被试作答、测量族、先验或 PyMC 计算图。基础格式包括 `rank`、`pick`、`mole`、`graded`，通过默认 `response_format` 与 `block_response_formats` 按 block 覆盖。graded 要求 K=2 及显式 `graded_scale`；观测任务与测量族独立，模型另行检查支持范围。

入口为 `from_frame()` 与 `from_mapping()`，归一化到同一种设计。提供 `to_frame()` 和设计摘要，用户可以检查完整配置和默认值。

默认 `allow_variable_block_size=False`，检查所有 block 大小相同。显式设置为 `True` 后允许不同大小，但不跳过其他结构检查。大小始终从设计表的实际成员计算，不能从有缺失的作答表推断，也不要求用户再输入一个可能不一致的 blocksize。

拟提供 `block_sizes` 表示 block ID 到大小的映射；`block_size` 在等大小时为该整数，在变长设计时为 `None`，不以最大值代替。变长模式也接受恰好等大小的设计。

```python
design = fc.FCDesign.from_frame(
    design_table,
    response_format="rank",
    allow_variable_block_size=True,
)
```

对两个入口采用相同规则。设计构造时复制输入；statement 身份、局部位置、维度和方向一致性由数据层检查。某种 block size 是否被选定测量族支持，由 Model 进一步检查。

## FCData 管理设计与作答

组合 FCDesign 与默认的一行一次 person-block 作答表，字段为 `person_id, block_id, response`：

```python
data = fc.FCData.from_frames(responses=responses, design=design)
```

默认 `encoding="selection_order"`。排名类 response 是长度 K 的位置列表：`[2,3,1]` 表示第 2 条 → 第 3 条 → 第 1 条；PICK 为 `[2,None,None]`，四项 MOLE 可为 `[2,None,None,4]`。元素是设计 position，列表槽位是名次；均保留全部选择集合。graded 则是量尺中的单个类别，不将标量当作选择顺序。

排名类内部转为 `person_id, block_id, statement_id, rank` 长表；也接受显式 `encoding="statement_rank"` 的长表输入。graded 保留 person-block 类别事件，不能写进 rank 列。任务及编码均明确解析，不根据数据内容猜；类别 C 来自量尺而不是样本中已出现的取值数。

检查未知 ID、重复作答、非法列表、重复／越界 position、NA 位置及 graded 类别。排名 NA 规范化为可空整数，不直接交给 PyMC 自动插补；graded 只在显式缺失状态下允许整次类别缺失。量尺、方向和 NA 详见 [数据协议](data_format.md)与 [graded 协议](graded_response.md)。

可选 `administration` 表通过 person/block/status 区分 answered、missing、not_administered；提供时与作答交叉校验。普通两表入口仅描述已提供的有效响应，不能据此推断全部施测状态。全 NA 且没有明确状态的记录须报告含义不明。规则与例子见 [数据协议](data_format.md)。

保留用户标签、原始表、设计和可选状态表，不预先绑定模型的稠密数组或类别长度。模型准备阶段才生成整数索引、任务事件编码、兼容排名约束和分组。一个 person-block 是一次观测，长表行数不等于独立观测数。

## Model 管理统计规格与联合构建

接收 `FCData`、默认测量族、按 block 的测量族覆盖，以及全局、block 和 statement 先验设置。族规格还可声明参数回归或自定义参数生成组件。共同 theta、trait 顺序、特质相关结构只定义一次；各族在同一原生 PyMC 模型中增加参数与观测。

`family` 的作用是给未单独指定的 block 提供默认值，`block_families` 的指定优先。等大小不等于同族，不同大小也不要求使用不同族。先解析 block 分配，再按族配置、大小、格式及计算形状分组。

构造阶段校验静态配置；`build()` 解析最终先验与约束、检查能力和识别条件并创建计算图，返回 `BuiltModel`；`fit()` 委托同一个通用采样函数。尚无真实 likelihood 实现的族不能通过空图或回退到成对 Bernoulli 来伪装可拟合。

复制并保留输入规格。构建后不原地替换族、先验或设计；修改配置时创建新的 Model，避免规格和已有图失配。缓存的图属于一个确定规格，不作为长期保存格式。

## 基础评分

已有被试摘要和新被试评分都是每个核心族的基础能力；新被试输入沿用默认列表或 graded 类别及其任务规则，通过已标定题目推断新 trait。New-person scoring uses fixed calibration and does not propagate calibration uncertainty; see [scoring](scoring.md).数据接口与评分可共用，具体族及其支持格式必须有独立检查。

## Prior 使用 PyMC 分布语义

普通用户使用轻量、可保存的声明：

```python
fc.Prior("LogNormal", mu=0.0, sigma=0.5)
fc.Prior("HalfNormal", sigma=1.0)
```

分布名称和参数遵循 PyMC，Prior 自身不创建随机变量。Model 构建时在正确上下文中创建 RV，并管理名称、维度、坐标和参数共享。首版不建立另一套分布语言，不接受任意代码字符串，也不默认按数据或 block size 自动缩放用户先验。

这借鉴 Bambi 将先验描述与实际构图分开的方式；Bambi 的 Prior 支持 PyMC 分布名称、参数及自定义分布工厂。[Bambi Prior 源码](https://github.com/bambinos/bambi/blob/main/bambi/priors/prior.py)

用户选择先验时区分以下作用域：

- `Model(priors=...)`：只配置全局潜变量结构中允许修改的先验，例如特质相关结构。
- `TIRT(priors=...)` 等族规格：配置该实例覆盖的 statement／block 参数。
- `Model(block_priors=...)`：按明确 block ID 选择可配置参数。
- `Model(statement_priors=...)`：按全局 statement ID 选择可逐项配置的参数。

不使用含糊的全局 `a`、`b` 参数名去同时覆盖不同测量族，也不把按 block 的先验复制给全局 theta。按 ID 使用固定的显式覆盖顺序，不增加通配符或正则匹配。具体例子见 [参数协变量与自定义组件](parameter_models.md)。

## 参数角色与识别约束

每个族声明自己可配置的参数角色、含义、支持集、维度和先验施加的尺度。例如正向载荷大小与带符号载荷是不同角色；如果模型使用 `keying * loading_magnitude`，先验施加在正的 magnitude 上。

先验作用于原始参数、自由坐标还是变换后的参数，必须在该族文档中明确。例如零和位置参数不能一边承诺独立 Normal 边际、一边未经说明地减均值。结构性参数、固定锚点和识别约束由专门的 identification 配置定义，不允许靠指定先验静默覆盖。

Model 为参数结构补齐必要的 `dims`、shape 或分布维数，并检查用户设置是否冲突。对相关结构等特殊参数，只接受与其约束匹配的分布和参数化；不是所有 PyMC 分布都能用于所有角色。

未知参数角色、错误分布参数、不匹配的支持集和维度，应给出含族、block 或 statement 标签的错误。复杂工厂的支持集不能只凭一个样本证明；须结合声明、构图检查与模型测试验证。

## 覆盖与共享规则

可逐项配置的直接先验按“族内置默认值 → 族实例用户设置 → block 设置 → statement 设置”解析，保存最终结果。`block_families` 覆盖的是该 block 的完整族规格；未在该规格中给出的先验使用该族默认值，不从被替换的默认实例暗中继承。

全局结构先验只解析一次。共享 statement 参数先确定身份，应用最具体的覆盖，再检查所有使用位置的一致性；按大小分组不会重复创建先验。若最终仍有冲突，明确报错，不按遍历顺序决定。联合约束参数不能默认拆成独立逐项先验。

构建前提供最终先验的可读预览，列出参数角色、适用标签、分布、超参数、约束及来源。预览应包含所有默认值，保存时写入 ModelSpec。具体预览方法名称在实施时确定。

## 题目参数协变量与自定义组件

族规格可用 `parameter_models` 为某个参数角色指定预测特征、link、残差和回归先验。设计保存特征值，Model 保存参数模型；一个参数只有一条生成路径，不能同时使用独立先验和另一个回归生成变量。

自定义模型通过受控的参数生成和 block 响应接口复用数据、编码及标准结果。声明参数角色、观测映射、完整 logp 与生成能力后，可继续使用通用拟合和诊断；不要求自动插件注册或复杂继承。完整边界见 [参数协变量与自定义组件](parameter_models.md)。

## 自定义与层级先验的高级入口

为需要自定义分布或层级结构的用户保留 PyMC 工厂入口，例如拟议的 `fc.Prior(factory=...)`。工厂在活跃 PyMC 上下文中接收受控的参数名称与维度，返回符合参数角色的随机变量；超先验也由用户用 PyMC 明确表达。

工厂按唯一参数／参数组调用一次，不按观测或 block size 反复调用。新增超参数使用受控命名空间，避免组间同名冲突。精确的工厂签名和高级支持范围在实现、测试前再确定。

不接受从其他计算图中提前创建的命名 RV 直接塞入配置。高级用户仍可访问 `BuiltModel.model`；任意自定义图的改动不能自动由普通配置恢复。

声明式 Prior 保存分布名称、数值、标签和最终解析信息。工厂保存可识别的来源及配置元数据，用户需另行保留可导入的构建代码；不 pickle 函数、不执行从结果文件读取的任意代码。保存后验与自动重建自定义图是不同能力。

## 组合使用示意

以下只展示配置层次，假设设计中已经包含 B4；示例超参数不是经验证的推荐默认值，参数角色名也仍属草案：

```python
import pyfcirt as fc
from pyfcirt.models import TIRT, TwoPLRank

design = fc.FCDesign.from_frame(
    design_table,
    response_format="rank",
    allow_variable_block_size=True,
)
data = fc.FCData.from_frames(responses=responses, design=design)

model = fc.Model(
    data,
    family=TIRT(
        priors={
            "loading_magnitude": fc.Prior("LogNormal", mu=0.0, sigma=0.5),
        },
    ),
    block_families={"B4": TwoPLRank()},
)
```

默认等大小时省略 `allow_variable_block_size`。后续基本框架的职责是表示、校验和解析这些配置；真实拟合仍须由已经通过 likelihood、梯度、预测与恢复检查的测量族提供。
