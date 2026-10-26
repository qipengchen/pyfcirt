# 以 block 为单位组合测量模型

一个被试在一个 block 上的响应是一次观测事件，可以是完整排名、PICK、MOLE 或二元题组的 graded。默认输入为顺序列表／类别标量。整体架构允许跨族共享特质；首版先用一个族打通 tutorial 完整分析，后续族复用该流程，再增加跨族联合。

以下为接口和统计约定草案，尚未实现模型或数值积分算法。

## 联合模型与固定的 block 分配

设被试为 n、题组为 b，已指定的测量族为 f_b，题组参数为 phi_b。在给定共同特质 theta_n 及模型参数后，先假定题组间条件独立：

\[
p(y\mid\theta,\phi)=\prod_{(n,b)\in\mathcal O}
p_{f_b}(y_{nb}\mid\theta_n,\phi_b).
\]

theta 的先验、维度顺序和相关结构在整个模型中只定义一次。题组间的额外依赖若确有需要，作为明确的模型扩展加入，不能从分组方式隐含产生。

共用维度标签不等于已解决识别。全局特质尺度和方向、每个族的效用位置与误差尺度等约束须相互兼容，并用整个混合设计验证。

这里是用户事先指定各 block 的测量族，不是推断未知族归属的混合分布模型。异质 block 的联合拟合依赖共同特质的尺度和识别约定，不把分别拟合的分数事后拼接成一个测量尺度。

以下示例假设 data 的设计中已包含 B1 至 B4。设计默认检查等大小；若显式开启 `allow_variable_block_size=True`，各 block 可以有不同大小。拟议使用方式：

```python
import pyfcirt as fc
from pyfcirt.models import TIRT, TwoPLRank, GGUMRank

model = fc.Model(
    data,
    family=TIRT(),
    block_families={
        "B3": TwoPLRank(),
        "B4": GGUMRank(),
    },
)

built = model.build()
fit = model.fit(chains=4, random_seed=2026, nuts_sampler="pymc")
```

`family` 提供默认族，`block_families` 按 block ID 明确覆盖。参数名和类名仍为草案。`Model.fit()` 委托同一个通用拟合函数；`fc.fit(built, ...)` 保留为显式入口，不维护两套采样实现。未知 block、未覆盖且没有默认族、重复冲突的配置均报错。

## 自定义先验的归属

全局结构先验与局部参数先验分开。局部直接先验按族默认值、族实例、block、statement 的明确规则解析；共享参数应用最终覆盖后仍有冲突时必须报错。普通先验用轻量 Prior 声明，特殊先验或层级结构可通过受控 PyMC 工厂表达；不在数据表中保存先验，也不默认自动缩放用户设置。

固定锚点、零和约束等 identification 与先验分别管理。题目参数可以使用独立参数回归组件；它替代该角色的直接先验生成路径，不叠加第二份参数。族仍负责 block 响应概率。作用域和工厂边界见 [核心对象](core_api.md)，协变量与扩展契约见 [参数与自定义组件](parameter_models.md)。

## 数据结构保持稳定

FCDesign 描述 statement、成员、参考位置、维度、方向与任务。FCData 默认接收 block 作答表；排名类转规范长表，graded 保留单次类别。block size 从设计计算，不要求全测验共用一个 blocksize。

测量族、先验和识别约束属于 `Model` 的统计规格，不重复存入作答行或每一行 statement 设计中。同一数据可以用于比较不同的模型规格。

基础格式包括 rank、pick、mole、graded，由默认任务与按 block 覆盖决定。graded 要求 K=2 及显式量尺，仅在声明支持的族上使用；模型须通过大小、任务概率、量尺和 NA 检查，不能因能读表就宣称支持。

同一 statement 在同一测量族中跨 block 使用时，其共享参数只定义一次，再供各计算组索引；若同一参数收到冲突的先验或约束，构建时明确报错。跨不同族复用同一 statement 时，首版拒绝隐含共享或拆分其参数；若以后支持，须显式定义参数关系、情境和识别方式。

## 借鉴 Bambi 的组件边界

Bambi 用 `Family` 关联 likelihood 与参数 link，并将统计规格交给 PyMC 后端构建。其后端分别构建常量、分布参数及响应组件。PyFCIRT 借鉴这种“规格到原生图”的边界，保留自身的 FC 数据与识别语义。[Bambi Family](https://github.com/bambinos/bambi/blob/main/bambi/families/family.py)、[Bambi PyMC 后端](https://github.com/bambinos/bambi/blob/main/bambi/backend/pymc.py)

Bambi 的公开 `Model` 使用一个 family；这里增加的是 PyFCIRT 自己的多 block-family 组合能力，不假定 Bambi 已提供相同功能，也不继承其内部类。[Bambi Model](https://bambinos.github.io/bambi/api/Model.html)

PyFCIRT 区分三件事：

- **测量族**：TIRT、2PLM-RANK、GGUM-RANK 等，决定特质与 statement 参数如何生成响应概率。
- **响应格式**：完整排名、best-only 等，决定观测到了什么。
- **观测分布**：完整排名可统一为排名模式上的离散分布；都使用 Categorical 并不代表测量族相同。

架构保留 `Model` 组合入口和清楚的族组件接口；按计划先让五个核心族达到基础建模与新被试评分，再增加混合族。始终不引入公式语言、多后端编译框架或自动插件发现。

## 构建与批处理

构建分成三步：

1. 校验并解析所有 block 分配，构建唯一的 person／trait 坐标及共享 theta、特质相关结构。
2. 各测量族为它管理的全部 statement 定义一次所需参数与约束，避免按大小分组后重复创建同一参数。
3. 将兼容的 block 按测量族配置、大小、响应格式及必要的计算形状分组，批量构建观测节点。

例如 TIRT 三联组、TIRT 四联组和 2PLM-RANK 三联组可形成三个计算组。同一被试的 theta 在这些组中由 person 索引引用，仍是同一个随机变量。

“按 block 建模”不要求“每个 block 创建一个 Python 对象和一组采样调用”。可以对少量异质组使用静态构建循环，组内批量计算被试与 block，避免按被试或观测展开计算图，也不为兼容大小而创建无效的全局巨大类别张量。

每个族的最小职责为：检查支持能力、准备数据、在给定共享潜变量的上下文中构建参数和观测节点、提供对应的生成机制。返回观测变量名称与观测映射供通用层使用；不启动 MCMC，也不创建另一份全局 theta。

## 排名模式与类别编码

含 K 个 statement 的无并列完整排名有 K! 个合法模式。用户默认输入从最偏好到最不偏好的位置列表，如 `[2,3,1]`；显式长表兼容入口仍可输入各 statement 的 rank。适配层将完整响应编码为内部从 0 开始的类别，两种入口含义一致。

类别映射使用 block 的参考 position 定义：先把作答转换为“从最偏好到最不偏好的局部位置排列”，再按固定的排列顺序编码。rank 向量和偏好顺序向量不能混用。类别编号只是标签，并无高低顺序，不使用有序类别回归来替代排名机制。

不同 block 的同一整数类别通过各自的 statement 映射解码；不同 K 的支持集不同，不能直接混入一个固定类别长度的观测张量。持久化需保存编码规则版本及各 block 的参考 statement 顺序。

类别数较小时，可用 `pm.Categorical(p=...)`；其概率必须对应全部合法响应并构成归一化分布。[PyMC Categorical](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Categorical.html)

统计上是多类别观测，不等于必须为所有被试、block 和 K! 模式永久保存完整概率张量。若某个机制支持直接计算已观测类别的概率及直接生成响应，可以使用具有 `logp` 和生成机制的 `CustomDist`。其支持集和统计含义须与类别表示一致，且接受相同的数值正确性与归一化检查。[PyMC CustomDist](https://www.pymc.io/projects/docs/en/stable/api/distributions/custom.html)

## PICK／MOLE 是完整排名的粗化观测

完整排名概率定义后，PICK 对兼容的所有“首位为 best”排名求和；MOLE 对所有“首位为 best、末位为 worst”排名求和。两者分别形成 K 和 K(K−1) 个事件类别，probability 与生成规则依照相应任务定义。

未选中／中间项的 NA 不减少 K，不是并列，也不自动插补。三项 MOLE 已经唯一确定完整次序；四项 MOLE 仍需汇总两个中间顺序。用稳定 logsumexp 汇总或使用严格等价的直接概率，具体输入见 [数据协议](data_format.md)。

逐观测 log-likelihood 使用实际观测事件的概率。预测生成后施加相同粗化规则，不能拿未观测的完整顺序当成实际答案。新被试评分也复用相同事件概率。真实整块未答／未施测在显式状态允许下不计入响应 likelihood。

## Thurston 类使用完整排序事件

以下讨论完整排名分支。二元 graded 属于偏好程度类别机制，须单独定义阈值／score 与识别，不能由两个排序事件的求和产生，详见 [graded 协议](graded_response.md)。

在给定 theta 的正态潜效用设定下，设一个 block 的效用为：

\[
U_{nb}\mid\theta_n\sim
\mathcal N(\mu_b(\theta_n),\Sigma_b).
\]

这里 Sigma_b 是给定特质后的效用误差协方差，不能把特质先验引入的总体变异重复算入。对于从最偏好到最不偏好的排列 pi：

\[
p_{nb,\pi}=P(U_{nb,\pi_1}>\cdots>U_{nb,\pi_K}\mid\theta_n).
\]

令 D_pi 取相邻效用差，则概率等于正交区域概率：

\[
P(D_\pi U_{nb}>0),\qquad
D_\pi U_{nb}\sim\mathcal N
(D_\pi\mu_b, D_\pi\Sigma_bD_\pi^\top).
\]

差分维度为 K-1。即使原始效用误差相互独立，共享 statement 的效用差也通常相关；将边际比较概率相乘、再对模式归一化，不等于以上完整排序模型。[Thurstonian 排名模型](https://www.ub.edu/gdne/documents/Pka_ranking.pdf)

参数也要有共同效用表示。例如成对截距须来自 statement 位置参数之差，不能将每对截距都自由估计后直接当成完整排序分布。相关要求与完整块级 likelihood 的讨论可见 [Frick 的块级信息研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC10656335/)。

因此，将教程中的成对 Bernoulli 改为本包的块级类别模型，需要重新确认联合概率和识别约束，而不只是更换 PyMC 分布名称。TIRT 可在线性效用均值上实现；其他 Thurston 类的均值与误差结构须分别按其公开定义实现和验证。

数值方案须检查区域概率精度、极端参数稳定性和对所有自由参数的梯度。若要使用 NUTS，不能直接把一个无可用梯度的外部 CDF 当作完整实现，也不能在每次 logp 调用中重新随机估计概率。确定性近似或固定积分节点也会引入近似误差，需要单独验证。

架构支持异质 K，不代表已经验证任意大小。先在小 K 上验证归一化、独立参考值、梯度和恢复，再决定支持范围；记录积分／近似方法、精度配置、运行时间与内存。类别数量随 K! 增长，但具体实现可避免无条件枚举全部模式。

## ArviZ 结果和预测

每个计算组可对应独立的 observed RV，并使用自己的观测维度。`ModelSpec` 保存组名称、测量族、类别编码、RV 名称，以及每条组内观测对应的 person_id、block_id 和稳定 obs_id。

原始后验仍保留标准 DataTree。按需将各组的逐观测 log likelihood 依照 obs_id 组合为统一的 person-block 视图，保留 person、block、family 坐标；不能把同一次响应的中间比较或不同存储副本重复计数。该重组属于 FC 的观测语义，LOO 等统计计算继续交给 ArviZ。

块级留出评估对应留出一个被试的一个 block 响应；预测新被试涉及不同的条件化或边际化目标，不能直接用同一结果命名替代。跨模型比较前须保证相同数据、响应空间和留出单位。

预测按任务解码为排名位置列表或 graded 原量尺类别，PICK／MOLE 列表保留结构性 NA；排名类另可导出长表。用户无需解释内部组名或记住类别整数；通用预测通过组件清单操作。

## 实施与检查顺序

详细任务见 [开发计划](roadmap.md) 的 V1-01–V1-16：首版 MGPPM 的排名和二元 graded 分别完成输入、拟合、诊断、预测、PPP、PSIS-LOO／DIC／WAIC、评分与保存。其余族复用同一流程；指标按固定参考抽样验收，详见 [数值规格](model_comparison.md)。

先实现设计与观测事件编码，再为首个族完成整个常规分析；不等全部族齐备才实现预测或 LOO。核心族逐步接齐后增加跨族联合，并验证共享 theta、混合预测与观测级结果重组。

必要检查包括：

- 对同一 K 的全部合法模式，概率归一化；对小 K，与独立数值参考一致。
- 固定参数 logp 与梯度、极端输入和标签重排不变性。
- 同一测量族的同一 statement 在不同大小组中参数不重复定义。
- 不同大小及不同族联合运行，小 MCMC、预测、保存读取和相关恢复均通过。
- 一个有效 person-block 响应恰好贡献一项完整 log likelihood，组内计算与单 block 参考一致。
- 对不支持的大小、格式、跨族参数共享或识别不足，给出带 block ID 的明确错误。
