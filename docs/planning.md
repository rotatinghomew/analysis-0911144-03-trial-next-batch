# 下一批试制建议

`planner.py` 在指定时点可见的资料上给出下一批试制建议。它是一个**有边界的序贯试验建议**，不是配方最优解、不是安全认证，也不把强度与损耗折算成单一分数。

## 请求

```json
{
  "at": 41,
  "n_trials": 2,
  "cost_budget": 40,
  "strength_floor": 42,
  "loss_ceiling": 0.135,
  "confidence": 0.8,
  "intent": "auto",
  "component_bounds": {"a": [0.1, 0.7], "b": [0.05, 0.35]},
  "min_separation": 0.04,
  "exploration_distance": 0.06,
  "extrapolation_margin": 0.02,
  "repeat_tolerance_strength": null,
  "repeat_tolerance_loss": null,
  "seed": 0
}
```

| 字段 | 含义 |
| --- | --- |
| `at` | 只使用 `visible_at <= at` 的记录。缺省取资料中的最大可见时点。 |
| `n_trials` | 本周最多安排次数，1–20。 |
| `cost_budget` | 本周成本上限，按组分比例与 `costs` 的点积计算；不能高于资料里的 `max_cost`。 |
| `strength_floor` / `loss_ceiling` | 业务性能底线。可以只给一项，也可以都不给。**不给时不会用任何权重把两个目标压成一个分数。** |
| `confidence` | 要求新试验达到全部底线的后验概率，也是探索区的可行性门槛。 |
| `intent` | `auto`、`confirm`（只做确认）、`explore`（只做探索）。 |
| `component_bounds` | 制造可行规则，按组分给出 `[下限, 上限]`，默认 `[0, total_fraction]`。上限同时用于裁剪观测设计，避免不可达的观测定义证据边界。 |
| `min_separation` | 同一批建议之间的最小配方距离，避免把预算花在几乎相同的组合上。 |
| `exploration_distance` | 与最近已完成试验的距离达到该值才被视为边界/探索候选。 |
| `extrapolation_margin` | 允许超出已完成试验凸包的幅度；默认很小。 |
| `repeat_tolerance_strength/loss` | 判断“能否复现观测前沿”时允许的放宽量；缺省取该响应观测值的标准差。 |
| `seed` | 仅在高维组分数超过网格上限、改用拉丁超立方/ Dirichlet 采样时使用。 |

未知字段直接返回 400，避免调用者以为某个权重或目标被采纳而实际被忽略。

## 状态

| `status` | 含义 |
| --- | --- |
| `recommended` | 有建议，每条都带 `role` 和 `rationale`。 |
| `exploratory_only` | 没有任何组合能以声明置信度满足性能底线；建议只用于减少不确定性，**不表示已接近底线**。 |
| `preference_required` | 没有声明底线，且只给一个名额；强度与损耗冲突，不引入任意权重就无法唯一选点。返回前沿上的 `alternatives` 供人决定。 |
| `no_recommendation` | 证据不足、模型不稳定、没有可行候选或成本无法覆盖；`reasons` 说明原因。 |

## 模型与不确定性

- **两个响应分别建模。** 强度和损耗各有一个高斯过程，没有共享权重，也没有把两者合成一个分数。
- **同一炉次共享误差。** 每个炉次有随机截距，协方差按炉次成块；同炉次的多条记录不会被当成独立证据。新试验按新炉次计入炉次方差。
- **线性趋势均值 + 高斯过程残差。** 趋势系数用广义最小二乘解析消去，不需要手调正则权重。
- **超参数有下界。** 合成或稀疏数据不会让似然塌缩成无噪声插值，从而虚报确定性。
- **信息增益**按 `0.5·log(1 + 潜在方差/噪声方差)` 计算，两个响应各算一次。

## 选择规则（无加权分数）

1. **可制造与预算**：只保留满足 `component_bounds`、组分守恒和本周成本的候选。
2. **覆盖与安全间隔**：候选必须在已完成试验的局部覆盖半径内、在凸包（含 `extrapolation_margin`）内、远离失败配方与在途配方。覆盖半径由试验簿的实际间距推导。
3. **确认（`confirm`）**：在观测范围内、预测前沿上，且新炉次有至少 `0.5·confidence` 的概率复现观测前沿（允许 `repeat_tolerance_*` 的放宽）；若声明了底线，还要求达到底线的概率不低于 `confidence`。这一步比较的是**新试验的预测分布**，不是把收缩后的后验均值直接和带噪声的观测最大/最小值比较，避免系统性地偏向重复当前领先配方。
4. **探索（`explore`）**：在边界区域、底线的可行概率不低于 `confidence`，按强度信息增益与损耗信息增益**分别排序后交替**选出，两个排序不会相加。
5. **预算与分离**：按上述顺序装入本周名额，跳过间距过近或超预算的候选。

## 输出

```json
{
  "status": "recommended",
  "data_basis": {"visible_at": 41, "completed_trials": 42, "distinct_batches": 11,
                 "failed_trials": 0, "pending_trials": 0, "records_after_cutoff": 6,
                 "completed_ids": ["..."], "failed_ids": [], "pending_ids": []},
  "model": {"strength": {"batch_effect_sd": 0.004, "residual_sd": 1.13, "...": "..."},
            "loss": {"...": "..."}},
  "support_rule": {"support_radius": 0.074, "failure_clearance": 0.05,
                   "pending_clearance": 0.04, "extrapolation_margin": 0.02,
                   "hull": "convex_hull_margin", "status": "empirical"},
  "budget": {"requested_trials": 2, "planned_trials": 2, "total_cost": 2.72, "cost_budget": 40.0},
  "observed_frontier": [{"id": "t035", "strength": 45.17, "loss": 0.125, "...": "..."}],
  "suggestions": [{
    "fractions": {"a": 0.6, "b": 0.3, "c": 0.1},
    "cost": 2.8,
    "role": "confirm",
    "predicted": {"strength": {"mean": 44.07, "std": 1.2, "interval": [42.5, 45.6]},
                  "loss": {"mean": 0.128, "std": 0.0027, "interval": [0.124, 0.132]}},
    "floor_probabilities": {"strength": 0.96, "loss": 0.99, "joint": 0.96},
    "information_gain": {"strength": 0.03, "loss": 0.02},
    "support": {"inside_observed_ranges": true, "inside_feasibility_envelope": true,
                "distance_to_nearest_completed": 0.03, "distance_to_nearest_failed": null,
                "distance_to_nearest_pending": null, "failure_exposure": 0.0,
                "outside_strict_components": []},
    "rationale": "确认已有优势：……"
  }],
  "alternatives": [],
  "reasons": ["..."],
  "meaning": "基于指定时点可见资料的序贯试制建议，不是最优配方证明、不是安全认证，也不把强度与损耗换算为同一分数。",
  "engine": {"name": "grouped-gp-pareto", "version": "1.0.0"},
  "revision": 1,
  "data_digest": "……"
}
```

`reasons` 会显式记录：使用了哪个时点、排除了多少条后续记录、多少条在途试验未参与、失败试验如何被使用、炉次数量是否偏少、是否没有失败证据、安全规则只是经验规则等。

## 历史建议

每次 `POST /plans` 都会把结果连同 `revision`、`data_digest`、规范化后的请求和引擎版本写入 `plans` 表。之后导入新资料只会产生新版本，旧建议按原样取回，不会在读取时被重新计算。`GET /plans` 列出建议，`GET /plans/{id}` 取回当时可见的完整结果。

## 明确不做的事

- 不在强度与损耗之间使用任意权重或固定换算率。
- 不把在途试验当作结果，不把失败试验当作强度为零。
- 不把同炉次记录当作独立重复，也不把预测可行性当作安全认证。
- 观测太少时不伪造“最优配方”，而是给出 `no_recommendation` 和原因。
