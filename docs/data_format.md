# PyFCIRT 输入数据设计草案

Implemented data-container usage is documented in the [English data guide](data.md). This page retains the original design rationale; model-fitting examples remain planned.

采用“block 作答表＋题组设计表”的默认入口。每个作答单元用选择顺序列表表达排名，内部再规范化为长表／事件记录。`FCDesign` 表示 statement、题组和维度的关系，`FCData` 组合设计与作答；不继承 DataFrame，也不让用户直接管理 PyTensor 数组。

本页先明确数据含义。以下 API 是待实现的设计示例。基础任务包括完整排名 `rank`、最偏好选择 `pick`、最偏好／最不偏好选择 `mole`，以及仅限二元题组的 `graded`。任务在设计中显式声明，分别校验。

## 默认输入：选择顺序列表

默认作答表一行是一次 person-block 响应，字段为 `person_id, block_id, response`。排名类的 response 是列表，**列表第一个元素最偏好**。元素指设计表的局部 `position`，不是名次、全局 statement ID 或某个模型的类别编号。

- `[1, 2, 3]` 表示第 1 条最偏好，其次第 2 条，最后第 3 条。
- `[2, 3, 1]` 表示第 2 条 → 第 3 条 → 第 1 条；转为逐 statement 的 rank 后为 `[3, 1, 2]`。这两个列表不能混用。
- 列表长度等于该 block 的 K，每个完整排名恰好包含 `1..K`。不同大小用各自长度，不补到全测验最大长度。

```python
responses = pd.DataFrame(
    {
        "person_id": ["P01", "P01"],
        "block_id": ["B1", "B2"],
        "response": [[2, 3, 1], [1, 3, 2]],
    }
)
data = fc.FCData.from_frames(responses=responses, design=design)
```

该入口默认 `encoding="selection_order"`；类型和嵌套列表会被复制、规范化，修改原列表不应改变已构造的数据。可接收整数列表／等价一维数组；不根据数值大小猜它是顺序还是类别编号。K≥10 时仍然使用列表，无需拼接数字字符串。

PICK／MOLE 保留相同长度，在**未知名次的位置**填写 None／NA：

- 三项 PICK：`[2, None, None]`，只有第 2 条被选为最偏好。
- 四项 MOLE：`[2, None, None, 4]`，第 2 条最偏好、第 4 条最不偏好，其他两条的顺序未知。
- 三项 MOLE：`[2, None, 1]`，剩余第 3 条的位置可由两端逻辑确定；原输入仍保留 NA。

这里列表元素是 statement 的位置，**不是把未选中 statement 从选择集合删除**。任务由设计决定，不从 NA 形状猜 PICK／MOLE。规范化时再转换为下面的 statement-rank 长表。

JSON 保存真正的数组；CSV 单元格使用明确 JSON 数组，如 `"[2, null, null, 4]"`，由导入器安全解析。Python 内使用 None／`pd.NA`，JSON 内使用 null，不执行单元格中的代码。旧的 `123` 等字符串只通过明确的导入适配转换，不作为默认格式。

graded 的 response 是一个类别标量，而非排序列表；同一张 block 作答表可以依设计的任务包含两种类型。相应规格见 [二元题组 graded 输入](graded_response.md)。

## 兼容入口与内部视图：statement-rank 长表

完整排名中，每一行是“一个被试在一个题组中，对一个 statement 给出的名次”。规范字段为 `person_id`、`block_id`、`statement_id`、`rank`，统一约定 **1 表示最偏好**。

```text
person_id,block_id,statement_id,rank
P01,B1,S1,3
P01,B1,S2,1
P01,B1,S3,2
P01,B2,S4,1
P01,B2,S5,3
P01,B2,S6,2
```

以上是格式示例。每个有效的完整排名须包含该题组的全部 statement，名次恰好为 `1..K`。行顺序没有统计含义；不能靠 DataFrame 当前排序判断呈现位置或偏好顺序。

显式传入 `encoding="statement_rank"` 时可直接读取这种四列长表，转换结果与默认列表入口一致；不靠表格形状或数字内容静默猜编码。此视图用于 rank／PICK／MOLE；graded 保留单次题组类别记录，不能复制成两个 statement 的名次。

`rank` 越小越偏好只是输入约定，不决定模型的数学机制。响应格式须显式声明；模型是否使用逐步选择、模式概率或其他 likelihood，由模型实现决定。完整题组排名是一次块级观测，不能将这些长表行自动当作独立观测计算 LOO。

`rank=0` 不是缺失编码。`rank` 使用 pandas 可空整数，接收空单元格、`pd.NA` 或 `NaN` 后统一规范化。完整排名任务仍要求 `1..K`；PICK／MOLE 按下面的任务规则接受结构性 NA。真实未作答与未施测不靠 NA 或缺行自动判定。

## PICK、MOLE 与结构性 NA

三种排名任务在规范化后的长表中共用上述四列，**NA 不使 statement 离开选择集合**。基础 PICK 定义为选一个最偏好项；基础 MOLE 定义为选一个最偏好项和一个最不偏好项。

- `rank`：K 个 statement 的 rank 恰为 `1..K`，不允许 NA 或并列。
- `pick`：选中的 statement 为 1，其余 K−1 项为 NA。已知所选项优于所有未选中项，后者内部次序未观测。
- `mole`：最偏好项为 1，最不偏好项为 K，其余 K−2 项为 NA。两端必须不同；中间项的内部次序未观测。

例如，四条 statement 的 MOLE 输入：

```text
person_id,block_id,statement_id,rank
P01,B1,S1,1
P01,B1,S2,
P01,B1,S3,
P01,B1,S4,4
```

这表示 S1 最偏好、S4 最不偏好；S2 与 S3 保持在四项选择集合内。兼容完整排名为 `S1>S2>S3>S4` 与 `S1>S3>S2>S4`，观测概率是两者之和。三项 MOLE 的中间位置已经由两端确定；保留原始 NA，在 adapter 中记录逻辑推断，不将它当成额外独立作答。

任务是设计属性，支持默认格式与按 block 覆盖。参数名为待实现草案：

```python
design = fc.FCDesign.from_frame(
    design_table,
    response_format="rank",
    block_response_formats={"B2": "pick", "B3": "mole"},
    allow_variable_block_size=True,
)
```

首版同一个 block 对所有被试使用同一任务；若以后支持被试特定任务，再增加明确的观测级任务字段。导入作者数据中的最不偏好编码（例如固定写为 3）时，转换器须在明确原编码含义后转为本包的 K，不直接把原编码当名次。

对于 rank／PICK／MOLE 响应 `y`、格式 `t` 和完整排名 `pi`，adapter 构造与观测相容的集合 C(y,t)：

```text
P(y | theta, item_parameters, t)
    = sum_{pi in C(y,t)} P(pi | theta, item_parameters)
logP(y) = logsumexp(logP(pi), pi in C(y,t))
```

可以使用与这个求和严格等价的直接公式；无需始终枚举 K! 模式。模型层收到合法的观测事件编码与兼容集合／约束，不把含 NA 的 rank 数组直接交给 PyMC 自动插补。

## 真实未作答与未施测

一次有效的 PICK／MOLE 响应已经包含结构性 NA，这与整次任务未回答不同。全 NA 或完全缺失的 person-block 不能仅凭表格判断是未施测、漏答还是文件不完整。

需要表达这些情况时，提供可选 `administration` 状态表：`person_id, block_id, status`，状态为 `answered`、`missing` 或 `not_administered`。它不是每次使用都必需的第三张表：普通两表入口仅包括已完成的 person-block；需要表示所有被试的施测全集时才提供状态表。状态与响应相互校验，`answered` 必须满足所属任务规则，缺失记录不能夹带已知选择。

真实整块缺失或未施测不贡献响应 likelihood；相应 trait 推断依赖其余有效作答和先验。这个处理对应可忽略缺失假设，不自动建模非随机漏答。完整 rank 任务中只缺少部分名次属于另一种部分观测，基础入口明确拒绝，不能冒充 PICK／MOLE。

排名类的规范长表仍保留每个已作答 block 的全部 statement 行。若提供只列被选项的稀疏导入入口，必须先声明任务，转换器再按设计补齐选择集合；不能从缺少的行猜任务、block size 或未施测状态。graded 的 NA 类别表示整次响应未获得，不存在排名未知带来的结构性 NA。提供状态表时，其缺失／未施测事件随 FCData 保存和导出。

## 题组设计表

普通单维 statement 使用一张方便录入的表：一行表示“某个 statement 在某个题组中的成员关系”，同一行带上它的维度及计分方向。

```text
block_id,position,statement_id,trait,keying
B1,1,S1,extraversion,1
B1,2,S2,conscientiousness,1
B1,3,S3,agreeableness,1
B2,1,S4,extraversion,-1
B2,2,S5,conscientiousness,1
B2,3,S6,agreeableness,1
```

- `statement_id` 是跨题组稳定的身份；`position` 是设计表中的题组局部位置，不能当作 statement ID。
- `trait` 使用可读标签，维度按 statement 定义。一个题组可以包含来自多个维度的 statement，无需再给题组指定单一维度。
- `keying` 使用 `+1/-1` 表示相对于维度的内容方向，不是用户提供的区分度数值，也不用于自动反转原始名次。方向如何进入概率模型，由对应模型说明和检验。
- 同一 statement 出现于多个题组时，其维度、方向和文本元数据必须一致。若需要题组特定版本或独立参数，应使用不同 statement ID；具体模型仍可拒绝尚不支持的重复使用设计。

这里的 `position` 是参考设计位置。以后若支持被试间随机呈现顺序，应另存实际呈现位置，始终通过 `statement_id` 连接作答和设计。

## 可选的 statement 特征

需要题目参数协变量时，`FCDesign` 可接收一张 `statement_features` 表，一行一个全局 statement ID。特征值只存一份，不重复到各 block 或被试作答行。Model 在所需 statement 上按 ID 对齐，选择明确的预测列，不根据行顺序拼接或自动使用所有数值列。

Likert 估计值、SE 与来源可以保存为特征或元数据，但存储它们本身不表示已经传播估计误差；固定特征、信息先验和联合标定的区别见 [参数协变量设计](parameter_models.md)。普通两表入口保持可用，不需要协变量的用户无需增加此表。

## 两个组合对象

`FCDesign` 内部将平面设计表拆成两个规范部分：statement 信息和题组成员信息。这样 statement 复用时只维护一份维度定义；对象也可在没有作答时用于设计检查与模拟。

默认 `allow_variable_block_size=False`，检查所有题组等大小；显式设置为 `True` 时接受不同大小。大小从设计成员表计算，不从作答记录推断。`from_frame` 与 `from_mapping` 使用相同规则，详见 [核心对象与先验配置](core_api.md)。

`FCData` 保存规范化作答和一个 `FCDesign`。同一设计可以用于不同样本；模型构建器负责将它们转换成模型所需数组。设计对象不保存先验、识别约束、估计参数或活动计算图。

拟议的基本入口：

```python
import pyfcirt as fc

design = fc.FCDesign.from_frame(design_table, response_format="rank")
data = fc.FCData.from_frames(responses=responses, design=design)
```

对象构造时复制输入并检查一致性。原始标签保留，模型适配时才生成连续整数索引；规范表不要求用户从 0 或 1 连续编号。后验中的 person、statement、block、trait 坐标使用这些标签。

## 更容易定义 statement、题组和维度

除表格入口外，提供一个适合小型测验的快捷入口，避免用户手写 Q 矩阵或重复填写每行：

```python
design = fc.FCDesign.from_mapping(
    traits={
        "extraversion": ["S1", "S4"],
        "conscientiousness": ["S2", "S5"],
        "agreeableness": ["S3", "S6"],
    },
    blocks={
        "B1": ["S1", "S2", "S3"],
        "B2": ["S4", "S5", "S6"],
    },
    keying={"S4": -1},
    response_format="rank",
)
```

快捷入口中，block 列表顺序确定参考 `position`，未指定的 keying 显式采用 `+1`。提供 `design.to_frame()` 让用户在拟合前查看或导出完整设计，包含全部默认值。这个入口与表格入口归一化为相同内容，不维护两套统计实现。

检查应指出具体位置，例如“B2 中 S7 没有维度定义”“S4 被分配给多个维度，当前入口仅支持单维 statement”。可提供设计摘要，列出每个维度的 statement 数、题组组成、重复使用及未使用 statement。结构检查不能据此保证模型可识别；识别检查仍属于模型模块。

## 多维加载与后续格式

首版使用单维 statement，避免提前实现通用配置系统。未来支持多维加载的模型时，增加显式 `statement_id, trait, keying` 映射长表，一行一个非零维度关系；题组成员表不变。该表表示加载结构和方向，实际载荷是待估计、固定还是受约束，由模型规格定义。不得把多个维度塞进逗号分隔的单元格，也不同时维护一份独立 Q 矩阵。

开启 `allow_variable_block_size=True` 后，同一设计可以混合不同 block size；大小直接由题组成员表计算，不设一个全测验共用的 blocksize。每个 person-block 的响应在模型内部编码为任务对应的事件；完整排名、PICK、MOLE 各有支持集。计算时按大小、任务和模型等兼容条件分组，用户不需要将不同题组补齐到相同长度。

测量族及其先验、识别配置由 `fc.Model` 的默认 family 与按 block 覆盖指定，不写入每一行作答或 statement 设计。这样同一份 `FCData` 可以交给不同模型规格，所有 block 仍在同一个图中共享 theta。Thurston 类使用完整排序事件概率，不把独立的成对 Bernoulli 当作其块级联合分布。详见 [block 组合设计](block_models.md)。

从旧数据导入时，允许显式列名映射；宽表、排列字符串及模型专属数字编码通过独立转换函数处理。`2PLM-RANK` 和 MGPPM 的导入适配须说明原始编码，并验证往返转换，不将某套模型的排列编号作为统一的数据协议。

保存时同时保留输入 encoding、设计的参考顺序、任务、可空值和 graded 量尺；读取后能重建默认选择顺序列表或显式的长表视图，不能把二者互相误认。

原实现的设计可作为参考：[MIRT4FC 的 BID 示例](https://github.com/SAL-Lab-ECNU/MIRT4FC)。它将作答 `Y` 与包含 Block、Item、Dim 的设计表分开；PyFCIRT 在此基础上明确全局 statement 身份和可读维度标签。
