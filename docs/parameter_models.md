# 参数先验 协变量与自定义组件草案

本页扩展既有核心对象设计，只描述架构，不实现运行库。不同 block 和 statement 的先验、题目参数回归及自定义测量族，最终都解析到同一个 PyMC 联合模型及标准结果契约。

## 按 block 与 statement 指定先验

在默认族先验之外，Model 接受显式的 ID 映射，不使用变量名通配符或正则规则。示意接口如下，假设设计中存在 B1 与 S3，参数角色及超参数均是示例：

```python
model = fc.Model(
    data,
    family=TIRT(),
    block_priors={
        "B1": {
            "loading_magnitude": fc.Prior("HalfNormal", sigma=1.0),
        },
    },
    statement_priors={
        "S3": {
            "loading_magnitude": fc.Prior("LogNormal", mu=0.0, sigma=0.3),
        },
    },
)
```

可逐元素配置的参数按以下顺序解析：族内置默认值、族实例的用户设置、block 设置、statement 设置。更具体的设置优先，并在先验预览和 ModelSpec 中记录来源。全局 theta 结构仍由 `Model(priors=...)` 单独配置，不受这些局部规则影响。

block 规则对真正的 block 参数选择该 block，对 statement 参数选择其成员；它不会把原本共享的 a_i 自动改成 block 特定的 a_ib。statement 规则按全局 ID 解析，与题组内局部 position 无关。

同一 statement 在多个 block 复用时，参数只有一份。先应用最终的 statement 覆盖，再检查各使用位置的配置是否一致；没有更具体覆盖来消除冲突时，两个 block 不能为同一共享参数留下不同先验。需要情境特定参数时，应由模型明确声明 statement-by-block 结构。

单个 statement 的先验只能用于该族允许逐项配置的角色。相关矩阵、零和向量及其他联合约束的参数，不能未经定义就拆成独立先验。固定锚点和 identification 也不能被局部先验覆盖。

## Statement 特征单独保存

为 `FCDesign` 增加可选的 `statement_features` 表，一行一个全局 statement ID，其他列为用户提供的特征。例如 Likert 参数估计、题目长度、社会赞许性或文本表示。普通用户仍可只提供作答和设计两张表；需要协变量时再增加特征表。

```python
design = fc.FCDesign.from_frame(
    design_table,
    statement_features=features,
    response_format="rank",
)
```

按 ID 对齐，不按表行位置拼接。一个 statement 跨 block 使用时只维护一份特征。只有参数模型明确选择的列进入预测矩阵，不把所有数值列自动当作协变量。

数据层检查重复 ID、类型和结构，Model 检查所需 statement 是否具有完整的预测特征。默认不因缺失特征静默删除作答；变换、分类编码和插补如果引入，均需明确配置和保存。

特征值属于数据，选择哪些特征、采用什么 link 及先验属于模型规格。所用列、列顺序、训练时的中心化或其他变换随规格保存，新 statement 预测使用同一套变换。

## 题目参数可以有自己的回归模型

新增轻量 `ParameterModel` 规格，负责生成一个声明的参数角色。测量族不再将每个参数硬编码为某一种独立先验，而是接收解析后的参数值。

以正的载荷大小为例，可以指定：

\[
\log a_i^{FC}=\beta_0+\beta_1 x_i+u_i,
\qquad u_i\sim\mathcal N(0,\tau^2).
\]

x_i 可以是经过方向、尺度及有效性处理的 Likert 参数特征。回归系数由数据估计；u_i 保留预测特征无法解释的 statement 差异。这里没有默认假定 FC 参数等于 Likert 参数，log link 也不代表 FC 排名响应使用 logit link。

独立的配置示例：

```python
family = TIRT(
    parameter_models={
        "loading_magnitude": fc.ParameterModel(
            predictors=["likert_log_loading", "text_length_z"],
            link="log",
        ),
    },
)

model = fc.Model(data, family=family)
```

名称是设计草案。该参数模型包含截距和逐 statement 残差；回归系数、截距及残差尺度均需有可配置且可预览的先验，不能隐式成为无先验的自由参数。去掉残差将变成确定性参数预测，是另一项需明确指定的假设。

这是借鉴 Bambi 将参数的预测器、link 和响应分布分开的思路，扩展到 FC 的题目参数层。[Bambi 基本组件](https://bambinos.github.io/bambi/notebooks/how_bambi_works.html)、[Bambi 分布参数回归](https://bambinos.github.io/bambi/notebooks/distributional_models.html)

先从显式预测列与设计矩阵开始，不为此自行编写公式解析语言。未来需要公式语法时复用成熟解析器。各族声明哪些参数角色支持预测，以及 link、残差尺度和识别约束如何作用。

## 一个参数只有一条生成路径

一个参数可以来自直接先验、参数回归、合法的固定配置或自定义工厂。在同一模型规格中必须唯一，不能既创建 a_i 的独立先验，再创建另一个回归生成的 a_i，或给同一变量额外乘一次未声明的先验。

`parameter_models` 替代对应角色原有的直接先验生成方式。用户再为该角色指定 `block_priors` 或 `statement_priors` 时，默认报冲突；若希望改变回归系数、残差或分组超参数，应在参数模型中明确配置相应角色。相关高级局部规则待有真实需求后增加。

参数回归先生成与识别约束相容的自由参数或潜在值，再按族定义形成实际参数；例如受零和约束的位置参数，其回归含义和变换不能由通用层自行猜测。载荷方向、全局特质尺度等识别要求保持有效。

## Likert 结果的三种用法

- **固定预测特征：** 将既有估计作为 x_i，学习 FC 参数回归。普通入口先支持这一方式，推断条件于提供的特征值，不自动传播它们的估计误差。
- **外部信息先验或固定标定：** 用 Likert 结果指定 FC 参数先验中心，或在有充分依据时固定参数。这是不同模型，不能与协变量回归混称；迁移尺度和允许偏离程度需明确。
- **不确定特征或联合标定：** 显式表示外部估计的误差模型，或联合拟合 Likert 与 FC 响应。需要处理相关性及样本关系，属于后续统计扩展，不是将 SE 多存一列就已完成。

如使用外部后验样本，须保留联合抽样关系，不能把每个 draw 当作新的独立标定观测。预测目标、信息流方向和是否允许 FC 数据更新外部标定都要说明。

Likert 与 FC 参数的跨格式对应并非无条件成立；Morillo 等的研究检验了这种不变性及其偏离。因此，本包默认把这些量视为有来源和尺度的特征，不自动视为已知的 FC 真值。[Morillo 等，2019](https://journals.copmadrid.org/jwop/art/jwop2019a11)

特征记录应说明来源、参数化、方向、估计样本及必要的变换。不能用待预测 FC 作答估出的参数再作为其外生特征；模型评估中，训练与验证数据的特征构建也须符合预测目标。新 statement 的参数预测与新被试评分是不同能力。

## 给自定义模型保留的组件边界

复用集中在明确的四层：

1. **数据准备与响应空间。** 标签映射、不同大小分组、排名编码和解码由公共数据工具提供。
2. **参数生成。** 直接先验、参数回归或 PyMC 工厂生成有角色和标签的参数，不在 likelihood 里重复定义。
3. **block 响应机制。** 测量族使用参数与共享 theta 计算完整 block 概率，并提供对应的生成过程。
4. **标准结果。** 组件清单、观测映射和变量角色供通用拟合、预测、评分、保存读取及 ArviZ 使用。

不同族只有在数学含义完全相同时才复用概率组件。相同的“ranking”名称或相同的 Categorical 外壳不足以证明 kernel 相同。

自定义族以对象传给 `Model`，采用明确的准备和构建接口，不必继承庞大基类，也不要求进入字符串注册表。构建上下文提供现有 theta、坐标、受控命名及参数共享信息；扩展不能悄悄创建第二份 theta 或改变观测单位。

族须声明支持的大小和格式、参数角色与识别约束；返回 observed RV 名称、变量角色、类别编码及 person-block 观测映射。响应机制可以提供完整概率向量供 Categorical 使用，或提供适配 CustomDist 的 logp 和生成方法。

用户因此可以只替换参数生成或 block 概率中的某一部分，继续使用原有数据和诊断工具。提供只有 logp 的扩展时，只能声明相应拟合能力，不能假装已有先验／后验预测；完整支持仍要求生成机制、梯度与统计正确性检查。

回归系数、残差及实际题目参数保留在原生后验中，变量角色和标签进入组件清单，ArviZ 可以继续汇总和检查。观测级 log likelihood 仍以完整 person-block 为单位，不因增加协变量改变。默认不保存巨大逐人逐题组概率张量。

公开受控组件接口，保留原生 PyMC 图作为高级入口；不提供任意修改图后仍保证规格和保存读取正确的万能 hook。自定义代码需由使用者保留可导入来源。

## 研究参考与实现顺序

FC 中加入协变量已有研究先例，例如 Lee 与 Smith 的 BRB 模型在实证分析中纳入 gender 协变量。本页的参数回归接口是架构建议，不表示该研究直接采用了这里的 Likert 预测方案。[Lee 与 Smith，2020](https://doi.org/10.1177/0013164419871659)

Zotero 中对应的条目为 Lee 与 Smith（2020），item key `4PMVTZNN`，BibTeX key `lee_bayesian_2020`；两种 key 用途不同。跨格式不变性文献及 Bambi 文档使用上面的原始来源链接。

先实现并测试三个核心对象、直接先验和组件输出契约，再增加显式 block／statement 选择。参数回归先验证固定特征模式，最后再考虑不确定特征、联合标定与更复杂的扩展 API。当前只保留这些清楚边界，不提前实现通用插件框架。
