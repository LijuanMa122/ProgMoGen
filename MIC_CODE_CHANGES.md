# MIC 代码改动说明（结合实现）

本文档说明在 **ProgMoGen** 仓库上为实现论文 *Motion-Inference-as-Control (MIC)* 所做的全部代码改动：新增了哪些文件、改了什么调用关系、关键函数如何对应论文公式。  
方法公式与复现清单见 [`MIC_REPRODUCTION_README.md`](./MIC_REPRODUCTION_README.md)；本文聚焦 **“代码里具体做了什么”**。

---

## 1. 改动总览：不动原链路，旁路接入 MIC

### 1.1 设计原则

| 原则 | 做法 |
|------|------|
| 不破坏 ProgMoGen | 原 `ddim_sample_loop_opt_fn`、原 `task_configs_eval/`、原 `script_eval/eval_task_*.sh` **保持不动** |
| 旁路扩展 | 新建 `mic/`、`task_configs_mic/`、`*_mic.py`、`*_mic.sh` |
| 复用资产 | 继续用同一 MDM checkpoint、DDIM100、HumanML3D、`atomic_lib`、`f_loss`/`f_eval` 语义 |
| 可切换入口 | 评测脚本把 `sample_fn` 从 `ddim_sample_loop_opt_fn` 换成 `ddim_sample_loop_mic` |

### 1.2 ProgMoGen vs MIC（代码路径对比）

```
【ProgMoGen 原路径】
script_eval/eval_task_hsi2.sh
  → tasks/eval_task.py
    → diffusion/ddim.py::ddim_sample_loop_opt_fn
      → 全程可微 DDIM × iterations
      → f_loss(x0) → Adam(noise_init)

【MIC 新路径】
script_eval/eval_task_hsi2_mic.sh
  → tasks/eval_task_mic.py
    → diffusion/ddim_mic.py::ddim_sample_loop_mic
      → mic/warm_start.py          # 可选：短程 ProgMoGen 优化
      → mic/sample_loop_mic.py     # 每步控制
          → Eq.9 / Eq.10 → Eq.11 → Eq.13 → inject
```

核心差异一句话：

- **ProgMoGen**：约束在「整段 denoising 之后」反传到初始噪声。  
- **MIC**：约束在「每一步 denoising」变成控制量 `u_t` 注入动力学。

---

## 2. 新增文件树

```
project/ProgMoGen/
├── MIC_REPRODUCTION_README.md      # 论文公式 ↔ 复现清单
├── MIC_CODE_CHANGES.md             # 本文：代码改动说明
│
└── progmogen/
    ├── mic/                        # ★ MIC 算法核心（新建包）
    │   ├── __init__.py
    │   ├── config.py               # 超参 dataclass
    │   ├── constraints.py          # Constraint 接口 + mask
    │   ├── common_criteria.py      # 共享 skating criterion
    │   ├── cem.py                  # CEM proposal q（Eq.9）
    │   ├── control_laws.py         # Eq.9 / Eq.10
    │   ├── feedback_regulator.py   # Eq.11
    │   ├── control_allocator.py    # Eq.12–13
    │   ├── tweedie.py              # pred_xstart / σ(t)
    │   ├── inject.py               # u_t → DDIM 一步
    │   ├── warm_start.py           # ProgMoGen 短程预热
    │   ├── sample_loop_mic.py      # Sec 3.4 主循环
    │   └── eval_utils.py           # 评测入口公共绑定
    │
    ├── diffusion/
    │   ├── ddim_mic.py             # ★ 继承 ddim，暴露 ddim_sample_loop_mic
    │   └── ddim_relax_mic.py       # ★ 继承 ddim_relax（GEO/HOI）
    │
    ├── task_configs_mic/           # ★ 各任务异构约束声明
    │   ├── eval_task_hsi1_mic_config.py
    │   ├── eval_task_hsi2_mic_config.py
    │   ├── eval_task_hsi3_mic_config.py
    │   ├── eval_task_geo1_relax_mic_config.py
    │   └── eval_task_hoi1_relax_mic_config.py
    │
    ├── tasks/
    │   ├── eval_task_mic.py              # HSI-2/3 入口
    │   ├── eval_task_hsi1_mic.py         # HSI-1 入口（head_gt）
    │   └── eval_task_goal_relaxed_mic.py # GEO-1 / HOI-1 入口
    │
    ├── eval/
    │   ├── main_eval_hsi1_mic.py
    │   ├── main_eval_hsi2_mic.py         # + unsuccess_rate
    │   ├── main_eval_hsi3_mic.py
    │   ├── main_eval_geo1_relax_mic.py
    │   └── main_eval_hoi1_relax_mic.py
    │
    └── script_eval/
        ├── eval_task_hsi1_mic.sh
        ├── eval_task_hsi2_mic.sh
        ├── eval_task_hsi3_mic.sh
        ├── eval_task_geo1_relax_mic.sh
        ├── eval_task_hoi1_relax_mic.sh
        └── eval_all_mic.sh
```

**未修改的原文件（重要）**：`diffusion/ddim.py`、`diffusion/ddim_relax.py`、`task_configs_eval/*`、原 `eval_task*.py` 主逻辑均未改写；MIC 只通过继承与新入口接入。

---

## 3. 调用链（从脚本到单步控制）

以 HSI-2 为例：

```
sh script_eval/eval_task_hsi2_mic.sh
        │
        ▼
tasks/eval_task_mic.py::main()
  · DiffusionClass = InpaintingGaussianDiffusionMIC
  · get_gen_motion_mic(...)
        │
        ▼
对每个 sample：
  bind f_loss / f_eval（供 warm-start + 最终指标）
  sample_fn = diffusion.ddim_sample_loop_mic
        │
        ▼
diffusion/ddim_mic.py::ddim_sample_loop_mic
  · build_mic_constraints(task_module, ...)
  · run_mic_sample_loop(...)
        │
        ▼
mic/sample_loop_mic.py
  1) warm_start → noise_init
  2) for t = T-1 … 0:
       pred_xstart = Tweedie(x_t)
       for each constraint k:
         objective → Eq.10 | criterion → Eq.9+CEM
       W ← Eq.11
       u ← Eq.13
       x ← inject(u)   # condition_score
  3) f_eval → loss_ret_val
        │
        ▼
保存 gen.npy → eval/main_eval_hsi2_mic.py
```

---

## 4. 核心模块详解（结合代码）

### 4.1 `mic/config.py` — 超参

`MICConfig` 集中存放论文 Implementation Details 与可扫超参：

| 字段 | 默认 | 对应论文 |
|------|------|----------|
| `M` | 16 | Eq.9 采样数 |
| `elite_ratio` | 0.2 | CEM elite 20% |
| `eta` | 0.0 | DDIM 确定性 |
| `warm_start` | True | 先 ProgMoGen/DNO 预热 |
| `gamma / W_max / W_init / ema_alpha` | … | Eq.11 |
| `lambda_` | 1.0 | Eq.12 的 λ |
| `ablation` | `"none"` | 消融开关 |

任务 config 里可用 `MIC_GAMMA` 等覆盖；CLI 可用 `--mic_ablation`、`--mic_no_warm_start`。

---

### 4.2 `mic/constraints.py` — 统一约束接口

每个约束是一个 `Constraint`：

```python
@dataclass
class Constraint:
    name: str
    type: ConstraintType   # OBJECTIVE | CRITERION
    energy_fn: Callable    # v_d / E，输入 x0_hat
    mask / mask_fn         # 空间-时间 scope M_k
```

- **objective**：可微，走 Eq.10（`autograd`）。  
- **criterion**：可不可微，只前向评估，走 Eq.9。  
- `violation()` → `max(0, energy)`，供 regulator 使用。  
- `latent_mask_from_joint_frames`：骨架级 mask（当前按**帧**激活 latent；关节→feature 精细映射可后续加强）。

任务侧通过：

```python
def build_mic_constraints(shape, length, device) -> List[Constraint]:
    ...
```

统一装配。

---

### 4.3 `mic/control_laws.py` — Eq.9 / Eq.10

#### Objective（Eq.10）

```python
def objective_control_eq10(...):
    x0 = x0_hat.detach().requires_grad_(True)
    loss = constraint.energy(x0, diffusion=diffusion)   # v_d
    grad = autograd.grad(loss, x0)[0]
    sigma = sigma_of_t(diffusion, t, ...)               # √(1-α̅_t)
    u = -sigma * grad                                   # ∇ log p ∝ -∇v
```

含义：在 Tweedie 干净估计上算 ProgMoGen 同类能量，得到 DPS 风格梯度控制。

#### Criterion（Eq.9 + CEM）

```python
def criterion_control_eq9(...):
    d_eps = cem.sample(M)                    # q = N(μ, Σ)
    for m in range(M):
        x0_m = perturb_x0_with_noise(...)    # 候选 ẑ_T^{(m)}
        E_m = constraint.energy(x0_m, ...)
        log_w = -E_m + log p0 - log q
    π = softmax(log_w)
    u = Σ π_m · dε_m
    cem.update_elite(d_eps, π, elite_ratio=0.2)
```

注意：PyTorch 1.7 无 `torch.pi`，`cem.py` 已改用 `math.pi`。

---

### 4.4 `mic/cem.py` — Cross-Entropy Method

`CEMState` 维护对角高斯 proposal：

- `sample(M)`：从 `q` 采噪声增量  
- `log_prob`：算 `log q`  
- `update_elite`：按权重取 top 20%，EMA 更新 `μ, log_std`

对应论文：每步用 elite 更新 `q`，使采样更偏向低终端代价区域。

---

### 4.5 `mic/feedback_regulator.py` — Eq.11

```python
c̃ = max(0, v_d(ẑ_T))
scale ← EMA(c̃)          # 跨约束归一化
c = c̃ / scale
W ← clip(W + γ·c, 0, W_max)
```

性质：持续违反则 `W` 上升；满足（`c̃=0`）则停止累积。

---

### 4.6 `mic/control_allocator.py` — Eq.13

对角 mask 下的闭式逐元素解：

```python
u_i = (Σ_k W_k² M_{k,i} u_{k,i}) / (λ + Σ_k W_k² M_{k,i})
```

消融模式：

| `ablation` | 行为 |
|------------|------|
| `no_regulation` | `W=1`，仍做 allocate |
| `no_allocation` | `u = Σ W_k u_k` |
| `no_coordination` | `u = mean(u_k)` |
| `objective_only` / `criterion_only` | 只启用一类约束 |

---

### 4.7 `mic/inject.py` — 控制注入 DDIM（Eq.3）

采用方案 A：复用 ProgMoGen 已有 `condition_score`：

```python
h = u / σ(t)                    # u = σ · h
out = condition_score(cond_fn→h, out_orig, x, t)
# 再标准 DDIM mean 更新（eta=0）
```

即把 MIC 的 `u_t` 转成 score guidance `h_t`，接到 MDM 原有引导钩子上。

---

### 4.8 `mic/warm_start.py` — 论文要求的稳定初始化

短程复用 ProgMoGen 噪声优化：

```python
noise_init.requires_grad = True
for it in range(warm_start_iterations):
    pred_res = f_forward_return_middle_list(...)  # 可微整段 DDIM
    loss = f_loss(...)
    loss.backward(); optimizer.step()
return noise_init.detach()
```

注意：必须设置 `diffusion.n_noise = 0`，否则 `ddim_sample_known_noise` 会报错（已修）。

`use_goal=True` 时（GEO/HOI），`f_loss` 带 `target_gt`。

---

### 4.9 `mic/sample_loop_mic.py` — Sec 3.4 主循环

伪代码即实现结构：

```python
x = warm_start(...) or randn
regulator = FeedbackRegulator(...)
cem_states = {k: CEMState(...) for criterion k}

for t in reversed(range(T)):
    x0_hat, out = get_pred_xstart(model, x, t)
    for k, c in constraints:
        u_k = Eq9 or Eq10
    W = regulator.update(x0_hat, constraints)
    u = allocate(u_list, W, masks, λ)
    x = ddim_step_with_control(..., u)
```

这是相对 ProgMoGen **最大的机制变化**：优化变量从「整段反传的 `noise_init`」变成「逐步合成的 `u_t`」。

---

## 5. Diffusion 类扩展

### 5.1 `diffusion/ddim_mic.py`

```python
class InpaintingGaussianDiffusionMIC(InpaintingGaussianDiffusion):
    def ddim_sample_loop_mic(...):
        constraints = build_constraints_from_task(...)
        return run_mic_sample_loop(self, model, ...)
```

- 继承 `diffusion/ddim.py`：保留 `sample_to_joints`、`f_forward_return_middle_list`、归一化等。  
- 签名对齐 `ddim_sample_loop_opt_fn`，方便入口替换。

### 5.2 `diffusion/ddim_relax_mic.py`

```python
class InpaintingGaussianDiffusionRelaxMIC(InpaintingGaussianDiffusion):  # from ddim_relax
    def ddim_sample_loop_mic(...): ...
```

用于 GEO/HOI：需要 `sample_to_joints_with_XZ_offset` / `joints_to_sample_with_XZ_offset` 等 relax 工具（骨架里 MIC 主路径主要用 `sample_to_joints` + hard target；relax 钩子已绑定便于后续接完整 multi-epoch relax）。

---

## 6. 任务配置（`task_configs_mic/`）

每个任务统一模式：

1. 保留 ProgMoGen 的 `lr / iterations / f_loss / f_eval`（给 warm-start 与最终 C.Err）  
2. 新增 `build_mic_constraints(...)`  
3. 默认约束三元组：**任务 objective + skating criterion + success criterion**

| 任务 | Objective | Criterion |
|------|-----------|-----------|
| HSI-1 | 头高 equal 三关键帧 | skating；误差&lt;0.05 success |
| HSI-2 | overhead barrier | skating；barrier 规则 success |
| HSI-3 | X/Z∈[-1,1] | skating；不越界 success |
| GEO-1 | 腕到平面距离 | skating；mean dist&lt;thresh |
| HOI-1 | 腕起终点 equal | skating；RMSE&lt;thresh |

以 HSI-2 为例（`eval_task_hsi2_mic_config.py`）：

```python
return [
  Constraint("overhead_barrier", OBJECTIVE, _objective_barrier, mask_fn=mask_barrier),
  Constraint("foot_skating",     CRITERION, _criterion_skating, mask_fn=mask_feet),
  Constraint("success_check",    CRITERION, _criterion_success, mask_fn=mask_barrier),
]
```

共享 skating 实现在 `mic/common_criteria.py`，内部调用原有 `eval.metrics.calculate_skating_ratio`。

---

## 7. 评测入口改动

### 7.1 三个入口脚本

| 文件 | 覆盖任务 | 相对原版关键差异 |
|------|----------|------------------|
| `tasks/eval_task_mic.py` | HSI-2, HSI-3 | `DiffusionClass=InpaintingGaussianDiffusionMIC`；`sample_fn=ddim_sample_loop_mic` |
| `tasks/eval_task_hsi1_mic.py` | HSI-1 | 加载 `EVAL_HSI1_FILE_NAME` 的 `head_gt`；多 batch（512） |
| `tasks/eval_task_goal_relaxed_mic.py` | GEO-1, HOI-1 | `ddim_relax_mic`；设置 `target_gt`（平面随机 / HOI 目标）；`use_goal` warm-start |

公共逻辑：`mic/eval_utils.py` 的 `load_mic_cfg`、`bind_task_hooks`。

### 7.2 与原 `eval_task.py` 的最小差异点

原代码（概念上）：

```python
DiffusionClass = InpaintingGaussianDiffusion  # ddim.py
sample_fn = diffusion.ddim_sample_loop_opt_fn
```

MIC：

```python
DiffusionClass = InpaintingGaussianDiffusionMIC  # ddim_mic.py
sample_fn = diffusion.ddim_sample_loop_mic
# + task_module.build_mic_constraints / mic_cfg
```

数据加载、CFG、mask、`gen.npy` 保存格式与 ProgMoGen 一致，便于直接对比。

---

## 8. 评测脚本改动

原 ProgMoGen 对 HSI-2/3 等只打：

- skate ratio  
- jittor_max（Max Acc）  
- constraint_error  

论文 Table 还需要 **Unsucc.Rate** 与 **Pass**。新增：

| 文件 | 新增指标 |
|------|----------|
| `main_eval_hsi1_mic.py` | 复用 HSI-1 mae + unsuccess；Pass stub |
| `main_eval_hsi2_mic.py` | barrier 规则 unsuccess |
| `main_eval_hsi3_mic.py` | 有界区域 unsuccess |
| `main_eval_geo1_relax_mic.py` | 按 loss 阈值 unsuccess |
| `main_eval_hoi1_relax_mic.py` | 按 loss 阈值 unsuccess |

`pass_rate` 目前打印 `nan` 并标注 TODO（MuJoCo 未接入）。

---

## 9. Shell 脚本

| 脚本 | 作用 |
|------|------|
| `eval_task_hsi1_mic.sh` | HSI-1 MIC（512） |
| `eval_task_hsi2_mic.sh` | HSI-2 MIC（32） |
| `eval_task_hsi3_mic.sh` | HSI-3 MIC（32） |
| `eval_task_geo1_relax_mic.sh` | GEO-1 MIC |
| `eval_task_hoi1_relax_mic.sh` | HOI-1 MIC |
| `eval_all_mic.sh` | 顺序跑全部 |

脚本使用 `set -eu`（兼容 `sh`/dash；已去掉不支持的 `pipefail`）。

运行示例：

```bash
cd progmogen
conda activate mdm
sh script_eval/eval_task_hsi2_mic.sh
```

---

## 10. 运行时数据流（单样本）

```
text + length (+ head_gt / target_gt)
        │
        ▼
┌─────────────────── Warm-start（可选）───────────────────┐
│ noise_init ~ N(0,I)                                      │
│ for it = 1..N_ws:                                        │
│   x0 = DDIM(noise_init)   # 可微，模型冻结               │
│   loss = f_loss(x0)       # ProgMoGen 任务损失           │
│   Adam 更新 noise_init                                   │
└───────────────────────────┬─────────────────────────────┘
                            ▼
┌─────────────────── MIC 逐步控制 ────────────────────────┐
│ x ← noise_init                                           │
│ for t = T-1 … 0:                                         │
│   ẑ ← MDM.pred_xstart(x,t)                               │
│   u_obj  ← -σ ∇ v_task(ẑ)          # Eq.10               │
│   u_sk   ← IS+CEM(E=skating)       # Eq.9                │
│   u_suc  ← IS+CEM(E=success)       # Eq.9                │
│   W ← integral(violations)         # Eq.11               │
│   u ← WLS allocate(u_*, M_*)       # Eq.13               │
│   x ← DDIM_step(x, t; h=u/σ)       # Eq.3                │
└───────────────────────────┬─────────────────────────────┘
                            ▼
              x0 → joints → 写入 gen.npy
                            ▼
              Skating / MaxAcc / C.Err / Unsucc
```

---

## 11. 兼容性修复（踩坑记录）

| 问题 | 原因 | 修复位置 |
|------|------|----------|
| `set: Illegal option -o pipefail` | `sh`→dash 不支持 pipefail | 各 `*_mic.sh` 改为 `set -eu` |
| `no attribute 'n_noise'` | warm-start 直调 `f_forward_*` 未初始化 | `warm_start.py` / `sample_loop_mic.py` 设 `n_noise=0` |
| `torch has no attribute 'pi'` | PyTorch 1.7 | `cem.py` 改用 `math.pi` |

---

## 12. 仍是骨架、待加强的部分

以下按论文完整复现仍需继续打磨（骨架已留接口）：

1. **Criterion 的 Tweedie 扰动离散化**（`tweedie.perturb_x0_with_noise`）需与 supplementary 完全对齐。  
2. **Scope mask** 目前按帧粗粒度；可做成关节→HumanML3D feature 精确映射。  
3. **GEO/HOI** 完整 multi-epoch `goal_relaxed` warm-start（现为 hard-target 短程噪声优化）。  
4. **MuJoCo Pass** 物理仿真检查未实现。  
5. **`γ / λ / W_max / guidance_scale`** 需按任务扫参对齐论文表格数值。  
6. Objective/criterion 控制量纲统一（Eq.9 的 `dε` vs Eq.10 的 `σ∇`）可再加归一化层。

---

## 13. 如何对照论文公式读代码

| 论文 | 代码入口 |
|------|----------|
| Eq.9 | `mic/control_laws.py::criterion_control_eq9` + `mic/cem.py` |
| Eq.10 | `mic/control_laws.py::objective_control_eq10` |
| Eq.11 | `mic/feedback_regulator.py::FeedbackRegulator.update` |
| Eq.12–13 | `mic/control_allocator.py::allocate` |
| Eq.3 注入 | `mic/inject.py::ddim_step_with_control` |
| Sec 3.4 主循环 | `mic/sample_loop_mic.py::run_mic_sample_loop` |
| Warm-start | `mic/warm_start.py::prog_mogen_warm_start` |
| 任务约束列表 | `task_configs_mic/*_mic_config.py::build_mic_constraints` |

---

## 14. 小结

本次改动 = **在 ProgMoGen 旁路新增一整套 MIC 推理栈**：

1. **算法包 `mic/`**：实现论文 Eq.9–13 + warm-start + DDIM 注入。  
2. **扩散子类 `ddim_mic` / `ddim_relax_mic`**：挂上 `ddim_sample_loop_mic`。  
3. **任务配置 `task_configs_mic/`**：把每个 benchmark 任务拆成异构约束列表。  
4. **评测入口与脚本 `*_mic.*`**：与原脚本平行，便于 A/B。  
5. **评测补 Unsucc.Rate**：对齐论文表格列（Pass 待接 MuJoCo）。

原 ProgMoGen 训练/推理代码路径可继续使用；MIC 通过新脚本独立运行与对比。
