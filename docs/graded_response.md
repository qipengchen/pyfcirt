# Graded response：仅限二元题组的输入与模型约定

日期：2026-10-25。本页为待实现设计，graded 属于基础输入的兼容范围；**二元题组指两条 statement，不是只有两个反应类别**。

首版模型改为 MGPPM：v0.1 同时完成完整排名和二元 graded 的拟合、诊断、预测、PPP、相对拟合、评分与保存，两分支分别验收。2PLM-RANK 不包含未定义的 graded，后续仍作为排名类模型接入。

## 1 一次题组作答对应一个类别

任务声明为 `response_format="graded"`，设计中 block size 必须为 2。用户对这两条 statement 的相对偏好程度给出一个类别，例如五级：

- 1：强烈偏好第 2 条。
- 2：较偏好第 2 条。
- 3：两者同样偏好。
- 4：较偏好第 1 条。
- 5：强烈偏好第 1 条。

这是一个显式定义的示例量尺。`position=1` 的 statement 是比较方向的参考，默认数值／类别顺序越高表示越偏好它。标签、类别顺序和中性类别若有，均由量尺规定；不从观测数据的最大值或是否为奇数类别推断。

默认 block 作答表仍只有 `person_id, block_id, response`，graded 的 response 是一个标量：

```python
responses = pd.DataFrame(
    {
        "person_id": ["P01", "P02"],
        "block_id": ["B1", "B1"],
        "response": [5, 2],
    }
)

design = fc.FCDesign.from_frame(
    design_table,
    response_format="graded",
    graded_scale={
        "categories": [1, 2, 3, 4, 5],
        "higher_prefers": "position_1",
        "labels": ["strongly_second", "second", "equal", "first", "strongly_first"],
    },
)
data = fc.FCData.from_frames(responses=responses, design=design)
```

API 名称仍是草案。`encoding="selection_order"` 决定排名任务的默认列表编码；graded 由显式任务和量尺走类别解析，不把标量 5 猜成“选择了第 5 条”。首版可共用一个明确 graded_scale；确有不同量尺需求时再增加按 block 的量尺覆盖。

内部映射到 `0..C-1`，保留原类别和标签。C 来自声明量尺，即使样本中某一类别无人使用也不能减少 C。不同人对两个 statement 的单独 Likert 打分不属于这种相对偏好任务。

## 2 与排名输入共用 FCData，保留各自语义

同一 FCDesign 可以通过 `block_response_formats` 分别声明 rank、pick、mole 和 graded；任何 graded block 都要求 K=2。基础阶段各测量族分别检验自身已支持组合；跨测量族混合仍按开发计划后续实现。

同一 FCData 保留 person/block 身份、原始输入与量尺；内部为排名类建立 statement-rank 视图，为 graded 保留一次 person-block 类别事件。graded 不能复制成两条独立 response，也不能塞进 rank 列当名次。

交换 statement 的参考顺序时，导入器须同步重编码类别，并保存映射。量尺不对称时使用显式映射；不能只交换 statement ID。keying 仍描述 trait 方向，不自动反转实际 graded 作答。

## 3 Graded 需要自己的响应概率

完整二元排名只有两种次序，不能仅靠对这两种次序边缘化产生五级的偏好强度。graded 由相应测量族提供类别概率及生成机制，例如累计阈值或 score-weighted 类别效用。

首轮模型范围依据已有教程：

- **TIRT graded，K=2**：审查累计 logit／probit 分支、阈值、误差尺度和二元组识别。
- **GTUM graded，K=2**：审查 unfolding 效用差与累计类别概率；triplet 示例中的 block 效应不直接复制为二元组默认结构。
- **MGPPM graded，K=2**：使用明确的类别 score 配置及 softmax；单独审查截距识别，不直接沿用完整排名的零和约束。
- **2PLM-RANK／GGUM-RANK**：首轮声明排名类格式；若增加 graded 变体，须另给统计定义与测试，不从同名 family 自动推断支持。

ordered 阈值、类别 score 等属于模型规格，不写入响应表；类别标签与方向属于任务量尺。均可在 PyMC 中形成多类别 observed RV，是否使用 OrderedLogistic、Categorical 或其他正确实现由族决定。

## 4 NA、评分、预测与验收

- graded 的类别 NA 是整次作答缺失，不是 PICK／MOLE 的未知内部次序；按 administration 协议处理，不当作“同样偏好”或额外类别。
- 新被试评分是基础能力，使用同一 graded 概率、量尺、阈值／score 和题目标定后验；不能先二分类别而丢掉偏好程度。
- 预测生成与原量尺相同的类别，逐观测 log-likelihood 对应一次 person-block。graded 与 rank 的 LOO 不直接混作同一响应空间。
- 检查 K=2 限制、类别映射往返、未出现类别、方向、阈值／score、NA 与标签；检查各类别概率归一化、固定参数 logp、梯度和生成。
- Formal tests check fixed-parameter probabilities, gradients and fixed-calibration scoring without posterior MCMC. Manual sampling is optional. Compare C=2 with a binary branch only when that model explicitly defines their equivalence.
