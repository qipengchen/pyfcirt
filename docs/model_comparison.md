# 相对拟合：DIC、WAIC 与数值一致性

Implementation status (2026-10-25): conditional likelihood, native ArviZ PSIS-LOO, chainwise pV DIC and legacy WAIC are implemented. Independent references execute pinned unmodified R2jags post-processing source and actual ArviZ 0.22.0 in separate processes; see the [current guide](comparison.md). This page retains the statistical definitions and acceptance specification. Full workflow/recovery acceptance remains separate.

## 1 共同输入与统计目标

从原生结果提取 `log_likelihood[chain, draw, obs_id]`。每个 obs_id 对应一次有效 person-block：rank 为完整排名，PICK／MOLE 为兼容排名求和后的事件，graded 为实际观察到的一个类别。

首版计算条件于被试 theta 和题目参数的响应 likelihood。它不包含参数先验、超先验或变换 Jacobian，也不将 latent theta 积分掉。计算中保留概率的归一化常数；真实未答／未施测不进入观测集合。条件与边际指标是不同定义，结果必须标记 likelihood_target。

模型比较需要同一响应数据、任务、有效观测和预测目标。排名与 graded 可以各自计算指标，不能仅因属于同一模型而直接比较两套不同响应数据的数值。

## 2 DIC：按 R2jags 默认实现

按用户最新指定，以 R2jags 默认 `DIC=TRUE, pD=FALSE` 路径为准。已核对其 `jags` → `mcmc2bugs` → `as.bugs.array2` 调用链；最后一步对每条链的 deviance 使用样本方差，再平均链内结果。[R2jags jags](https://github.com/suyusung/R2jags/blob/master/R/jags.R)、[as.bugs.array2](https://github.com/suyusung/R2jags/blob/master/R/as.bugs.array2.R)

本次读取的 R2jags DESCRIPTION 标为 0.9-0；生成参考输出时锁定该版本、依赖版本和源码内容标识，不把持续变化的 master 链接当作已冻结的测试环境。

设 c 为链、s 为保留抽样、i 为观测：

```text
D[c,s]     = -2 * sum_i(log_likelihood[c,s,i])
Dbar[c]    = mean_s(D[c,s])
pV[c]      = var_s(D[c,s], ddof=1) / 2
DIC[c]     = Dbar[c] + pV[c]

Dbar       = mean_c(Dbar[c])
pV         = mean_c(pV[c])
DIC        = mean_c(DIC[c])
```

这里仍然直接使用响应似然形成 D，但复杂度修正采用 R2jags 的 pV。**不使用此前讨论的 `2*Dbar-D(mean_parameters)` 作为本包默认 DIC，也不在计算方差前混平所有链。** `pD=TRUE` 调用 rjags 另算的 DIC2 是另一条路径，不作为本次目标。

保留每条链的 Dbar、pV、DIC 以及最终平均，明确返回字段为 pV。方差使用 S−1 分母，每条链至少两次有效保留抽样；首版要求各链相同保留长度，不静默丢弃 NaN。改变链归属可能改变这个版本的 DIC，测试须保留链维度。

先对每个 draw 的**总 deviance**求方差，不能将逐观测 deviance 方差直接相加：后者丢失观测项随参数抽样变化时的协方差。不能以 log posterior 或缺少常数项的 Potential 数值代替响应 likelihood。

## 3 WAIC：按旧版 ArviZ 0.22.0 实现

用户指定旧版 ArviZ 为参考，固定到 0.22.0，使用其 `waic` 源码的数值约定。ArviZ 1 已移除 WAIC，因此运行包保留一个短的独立实现，当前 ArviZ 继续提供 PSIS-LOO 和通用诊断。[旧版源码](https://github.com/arviz-devs/arviz/blob/v0.22.0/arviz/stats/stats.py)、[ArviZ 迁移说明](https://python.arviz.org/en/stable/user_guide/migration_guide.html)

把 chain、draw 合并为 S 个同权抽样，只在 WAIC 的抽样汇总中这样处理：

```text
lppd[i]       = logsumexp(log_likelihood[:,i]) - log(S)
p_waic[i]     = var_s(log_likelihood[s,i], ddof=0)
elpd_waic[i]  = lppd[i] - p_waic[i]

elpd_waic     = sum_i(elpd_waic[i])
p_waic        = sum_i(p_waic[i])
WAIC          = -2 * elpd_waic
SE_elpd       = sqrt(n_obs * var_i(elpd_waic[i], ddof=0))
SE_WAIC       = 2 * SE_elpd
```

后验样本方差与跨观测 SE 方差都用 ddof=0，以匹配该版 ArviZ。不要直接使用 R `var()` 的默认分母，否则在有限样本下不一致。保留逐观测项；`p_waic[i]>0.4` 的可靠性警告、单观测情形和 pointwise 输出也按参考行为检查。

输出明确区分 log-scale 的 elpd_waic 和 deviance-scale 的 WAIC。旧版 ArviZ 的返回字段会随 scale 表示相应数值，对照测试先统一尺度和字段含义，不能只对比同名字段。WAIC 与 DIC 在本页采用 deviance scale，数值较小表示相对更优；PSIS-LOO 的 elpd 则明确标示方向。

## 4 数值一致性验收

数值测试固定输入，不分别重跑两套 MCMC。测试代码与运行包分开，运行包不依赖 R、JAGS 或旧版 ArviZ；参考测试经标准库 subprocess 直接调用已配置的独立 Rscript／旧 Python，现场比较输出；不需要 rpy2／pyjags，不在测试中安装软件或降级运行环境。

### A 固定 posterior log-likelihood 矩阵

- 用明确的 chain/draw/obs_id 小型 fixture，同时传给新实现和参考程序。覆盖不同链均值、不同链方差、观测间相关变化及常量抽样。
- **DIC**：实际调用 R2jags 的参考后处理路径产生 Dbar、pV、DIC；保留链内分量。特意使用链均值不同的 fixture，确保错误的 pooled-variance 实现不能通过。
- **WAIC**：在隔离环境实际调用 `arviz==0.22.0` 的 `waic(pointwise=True)`，对齐 log/deviance 两种尺度，比较逐观测项、有效参数数、总值、SE 与警告。
- 记录参考包版本、来源标识、固定输入、维度顺序、样本选择与实际执行结果。矩阵可直接在测试中定义，通过 CSV／标准库 JSON 交换；不强制增加期望文件和新的序列化依赖。若实际需要缓存，期望输出必须由真实运行保存，不手抄文档数值。

### B MGPPM 响应概率与 log-likelihood

- 对排名、PICK／MOLE 和二元 graded，固定 theta、signed loading、截距和 score 配置；用独立短参考计算每条观测概率，再与 PyMC 图中逐观测 logp 比较。
- 排名的零和截距和 graded 的独立截距分别处理，不将两个分支错误地统一为同一约束。graded 输入类别须先按量尺映射到正确 score 行。
- 对同一组固定参数抽样生成相同 log-likelihood，再计算 DIC／WAIC。这样分开定位模型概率错误与指标汇总错误。

### C 流程与边界

- 检查行顺序／分组／保存读取不改变 obs_id 及最终指标；WAIC 对样本顺序不变，R2jags 型 DIC 仅要求链内重排不变，不能任意重分配链。
- 覆盖多链、单观测、常量抽样、极小但有限的概率、NA、非有限 likelihood 和抽样数不足；不为获得有限输出而静默裁剪概率或删样本。
- 首版 float64 普通 fixture 的预设目标为 `rtol=1e-10, atol=1e-10`；近零量以绝对误差判断。极端用例另给明确参考和判据，不靠放宽容差掩盖口径差异。
- 小型 MGPPM MCMC 后在同一组后验抽样上跑参考比较，并验证保存读取后结果一致。后验是否充分探索另由链诊断与恢复判断，不能由指标计算一致来替代。

### D 依赖、触发和未配置环境

普通测试用已有 NumPy 和 pytest 比较独立公式、尺度、边界；LOO 在同环境直接对照当前 ArviZ。单独的 reference 测试实际调用 R2jags 默认后处理函数，以及隔离 Python 的 `arviz==0.22.0`。R2jags／rjags 的安装可能需要系统 JAGS，但 DIC 汇总对照不需要另外拟合一遍 JAGS 模型。

未配置参考环境时明确 skip，不影响普通测试；已经配置后版本不符、子进程失败、超时或数值差异必须报失败。首次实现、修改计算公式／聚合／观测单位、更新参考版本及发布前必须实际运行关键参考用例。不能以 skip、同一份公式的重复实现或两次独立 MCMC 的近似值代替对照证据。完整执行方式见 [测试计划](testing.md)。

## 5 模块与计划落点

`results.py` 负责带标签的共同 likelihood 视图，`model_comparison.py` 放短而直接的 DIC／WAIC 函数及方法元数据；PSIS-LOO 调用当前 ArviZ。不按测量族复制一套指标函数，也不建立通用指标插件框架。

首版 V1-11 拆为共同 likelihood、PSIS-LOO、DIC、WAIC、跨实现数值核对五个子任务。快速验收集中于 `tests/test_model_comparison.py`，直接参考对照放 `tests/reference/test_legacy_metrics.py` 及两个短参考脚本。按需保存少量固定资源，不建立独立验证库或强制 golden 文件生成框架。
