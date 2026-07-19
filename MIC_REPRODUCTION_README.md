# MIC 完整复现指南（基于 ProgMoGen）

本文档面向在现有 **ProgMoGen** 代码库上完整复现论文：

> *Training-free Controllable Human Motion Generation under Heterogeneous Constraints*  
> Hui et al., arXiv:2607.01990（Motion-Inference-as-Control, **MIC**）

**原则**：所有方法细节以论文 PDF 正文公式与文字为准；实现落点对齐 ProgMoGen 仓库中的 MDM + DDIM + 任务约束接口。

---

## 0. 一句话结论：MIC 相对 ProgMoGen 改了什么

| | ProgMoGen | MIC |
|---|---|---|
| 约束介入时机 | **整段 DDIM 跑完**后，对最终 `x₀` 算 `f_loss`，反传到 **初始噪声 `noise_init`**，Adam 迭代优化 | 在 **每一步去噪** 注入控制量 `u_t`，边去噪边控 |
| 约束类型 | 仅 **可微 objective**（`equal` / `less_than` 等 surrogate） | **objective（梯度）** + **criterion（仅前向评估，无梯度）** 统一控制接口 |
| 多约束合成 | 任务里手写把多个 loss 相加 | **Feedback Regulator（权重积分）** + **Control Allocator（按 scope 加权最小二乘）** |
| 初始化 | 直接从随机噪声优化 | **先用 ProgMoGen/DNO 式优化做 warm-start**，再用 MIC 逐步控制 |

核心心智模型：

```
ProgMoGen:  noise_init ──(100步可微DDIM)──► x0 ──f_loss──► ∂/∂noise_init ──Adam──► 更新噪声
MIC:        x_t ──(每步)──► 各约束算 u_{k,t} ──协调──► u_t ──注入动力学──► x_{t-1}
            （objective 用 ∇；criterion 用 path-integral / CEM；最后 warm-start 可接 ProgMoGen）
```

---

## 1. ProgMoGen 去噪过程（必须先吃透，再改）

### 1.1 关键链路

```
script_eval/eval_task_*.sh
  → tasks/eval_task.py | eval_task_goal_relaxed.py | eval_task_hsi1.py
    → 绑定 task_configs_eval/*.py 的 f_loss / f_eval
    → diffusion/ddim.py::InpaintingGaussianDiffusion.ddim_sample_loop_opt_fn*
    → 内部 f_forward_return_middle_list → ddim_sample_loop_progressive_opt_known_noise
    → 每步 ddim_sample_known_noise（eta=0，确定性）
```

关键文件：

- `progmogen/diffusion/ddim.py`：标准约束优化（优化 `noise_init`）
- `progmogen/diffusion/ddim_relax.py`：GEO/HOI 松弛约束 + XZ offset
- `progmogen/diffusion/gaussian_diffusion_v2.py`：DDIM 数学、`p_mean_variance`、`condition_score`
- `progmogen/atomic_lib/math_utils.py`：关节索引与可微约束原子
- `progmogen/task_configs_eval/*.py`：各任务 `f_loss` / `f_eval` 与 `lr/iterations`

### 1.2 ProgMoGen 优化伪代码（与源码一致）

```python
# ddim.py :: ddim_sample_loop_opt_fn
rng = np.random.default_rng(np_seed)
noise_list = [randn(shape) for _ in range(T)]   # 各步噪声固定；eta=0 时实际不参与
noise_init = randn(shape); noise_init.requires_grad_(True)
opt = Adam([noise_init], lr=task_config.lr)

for it in range(task_config.iterations):          # 如 HSI-2: lr=0.005, it=100
    pred_res = full_ddim_reverse(noise_init, noise_list)  # T=100 步，梯度打通
    loss = f_loss(pred_res, ...)                  # 内部 sample_to_joints → 关节空间
    loss.backward()
    opt.step()

return pred_res.detach()
```

要点：

1. **约束不在逐步去噪里**，而在整段 denoising 之后。
2. 模型权重冻结；可优化变量只有 `noise_init`。
3. `sample_to_joints`：`do_inv_norm` → `recover_from_ric` → `[1,22,3,T]`。
4. `--use_ddim_tag 1` → respacing `ddim100`（100 步）。

### 1.3 DDIM 单步（ProgMoGen / MDM）

`ddim_sample_known_noise` / `ddim_sample`（`eta=0`）：

```
out = p_mean_variance(model, x_t, t)          # 得 pred_xstart ≈ x̂0
eps = predict_eps_from_xstart(x_t, t, x̂0)
mean = sqrt(α̅_{t-1}) * x̂0 + sqrt(1-α̅_{t-1}) * eps
x_{t-1} = mean                                 # eta=0 时无噪声项
```

已有 score-guidance 钩子（**MIC 注入控制时应复用/改造**）：

```python
# gaussian_diffusion_v2.py :: condition_score
eps = eps - sqrt(1 - α̅_t) * cond_fn(x, t, ...)
# 再由新 eps 重算 pred_xstart / mean
```

论文中控制 `h_t` 加在 score 上（Eq.3）；离散实现应对齐这种 **改 eps / 改 mean** 的注入方式，而不是再去 Adam 优化 `noise_init`（那一步仅作 warm-start）。

### 1.4 各评测任务在 ProgMoGen 中的约束（客观函数）

| 任务 | 配置 | 约束内容（ProgMoGen 可微 surrogate） |
|------|------|--------------------------------------|
| HSI-1 | `eval_task_hsi1_config.py` | 首/中/末帧头高 = GT（`equal`） |
| HSI-2 | `eval_task_hsi2_config.py` | 首尾头高>1.5；中间头高<0.5；中间脚高<0（`greater_than`/`less_than`） |
| HSI-3 | `eval_task_hsi3_config.py` | 全身 X,Z ∈ [-1,1] |
| GEO-1 | `eval_task_geo1_relax_config.py` | 左手腕贴平面 + relax + RT 变换回硬约束 |
| HOI-1 | `eval_task_hoi1_relax_config.py` | 左手腕起终点目标 + relax |

论文强调：ProgMoGen **只能**用这类可微 surrogate；对 **criterion**（如 skating 是否超阈、是否成功、MuJoCo 是否通过）只能事后评测，不能在生成中直接用非可微反馈——这正是 MIC 要解决的。

---

## 2. 论文方法全景（公式级，不可遗漏）

### 2.1 问题设定（Sec 3.1）

- 文本条件运动先验：`p(x | c^p)`（预训练 MDM）
- 约束能量似然：`p(d|x) ∝ exp(-v_d(x))`
- 目标后验（Eq.2）：

```
p(x | c^p, d) ∝ p(x | c^p) · exp(-v_d(x))
```

希望采样既贴近先验（自然），又满足约束。

### 2.2 受控扩散动力学（Eq.1–4）

无控 VP-SDE 反向（Eq.1）：

```
dx_t = [ -β(t)/2 · x_t - β(t) ∇_{x_t} log p_t(x_t|c^p) ] dt + √β(t) dε̄_t
```

注入 guidance `h_t`（Eq.3）：

```
dx_t = [ -β(t)/2 · x_t - β(t)(∇ log p_t + h_t) ] dt + √β(t) dε̄_t
```

换元 `z_t = x_{T-t}` 写成控制形式（Eq.4）：

```
dz_t = b(z_t,t) dt + σ(t) ( u_t(z_t,t) dt + dε_t )
```

其中：

- `b(z_t,t) = β(T-t)/2 · z_t + β(T-t) ∇ log p_{T-t}(z_t|c^p)`
- `σ(t) = √β(T-t)`
- `u_t = √β(T-t) · h_{T-t}`  ← **实际要算并注入的控制量**

### 2.3 最优控制目标（Eq.5–6）

```
u*_t = argmin_u E[ E(z_T) + ∫_t^T (1/2)||u_s||² ds ]
```

- 终端代价 `E(z_T)`：由约束评估器 `v_d` 在最终运动上给出
- 运行代价：惩罚控制能量，避免偏离先验过远

解析解（Eq.6，desirability `ψ`）：

```
u*_t dt = σ(t) ∇_{z_t} log ψ(z_t,t) dt
        = σ(t) ∇_{z_t} log E_{P⁰}[ exp(-E(z_T)) | z_t ] dt
```

`P⁰` = 无控扩散轨迹分布（`u≡0`）。

### 2.4 Criterion 形式（无梯度，Eq.7 → 实例化 Eq.9）

等价 path-integral 形式（Eq.7）：

```
u*_t dt = E_{P⁰}[ exp(-E(z_T)) dε_t | z_t ] / E_{P⁰}[ exp(-E(z_T)) | z_t ]
```

**实用近似（Eq.9，重要性采样 + CEM）**：

在每步：

1. 从 proposal `q = N(μ, Σ)` 采 `M` 个噪声增量 `{dε_t^{(m)}}`（论文：**M=16**）
2. 用 **Tweedie** 从当前状态估干净运动 `ẑ_T^{(m)}`
3. 算终端代价 `E(ẑ_T^{(m)})`
4. 权重：

```
π̃_t^{(m)} = exp(-E(ẑ_T^{(m)})) · p₀(dε_t^{(m)}) / q(dε_t^{(m)})
π_t^{(m)}  = π̃_t^{(m)} / Σ_j π̃_t^{(j)}
u_t dt     = Σ_m π_t^{(m)} dε_t^{(m)}
```

5. **Cross-Entropy Method**：按权重取 elite（论文：**elite ratio = 20%**），用 elite 更新 `q` 的 `μ,Σ`，使 proposal 更靠近低代价区域  
6. `q` 初值：标准高斯 `N(0,I)`，**每个去噪步更新**

### 2.5 Objective 形式（有梯度，Eq.8 → 实例化 Eq.10）

```
u*_t dt = σ(t) ∇_z log p_t(d | z_t) dt
```

实用近似（Eq.10，Tweedie + DPS 风格）：

```
ẑ_T = Tweedie(z_t)                    # 当前步干净估计 = MDM 的 pred_xstart
u_t dt = σ(t) ∇_z log p(d | ẑ_T) dt
       ≈ -σ(t) ∇_{ẑ_T} v_d(ẑ_T) · ∂ẑ_T/∂z   # 能量形式
```

实现上：对 `pred_xstart`（或解码后的 joints）算可微 `v_d`（即 ProgMoGen 的 `f_loss`/`f_eval` 同类函数），`autograd.grad` 得到控制方向。论文写明这与 **Diffusion Posterior Sampling / gradient-based guidance** 对齐。

### 2.6 约束协调（Sec 3.3，Eq.11–13）——不可省略

给定 `K` 个约束，各自得到 `{u_{k,t}}`。

#### (A) Feedback Regulator（Eq.11）

违反度：

```
c̃_{k,t} = max(0, v_{d_k}(ẑ_T))     # 在当前步 Tweedie 干净估计上评估
```

用 **EMA running scale** 把 `c̃` 归一化成可比较的 `c_{k,t}`（论文正文：指数滑动平均尺度；具体 EMA 系数在正文未给数值，实现时需可配置，建议默认 `α_ema∈[0.9,0.99]`，并做消融）。

权重积分更新：

```
W_{k,t+1} = Π_{[0, W_max]} ( W_{k,t} + γ · c_{k,t} )
```

性质（论文强调）：

- 持续违反 → `W` 累积增大
- 一旦满足（`c̃=0`）→ 停止累积，避免霸占控制预算

#### (B) Control Allocator（Eq.12–13）

每个约束有空间-时间 scope 掩码 `M_k`（对角，作用在展平后的 `u ∈ R^{N·J·D}`）：

```
u_t = argmin_u  Σ_k || W_{k,t} (M_k u - M_k u_{k,t}) ||²  + λ ||u||²
```

闭式解（Eq.13）：

```
u_t = ( Σ_k W_{k,t}² M_kᵀ M_k + λ I )^{-1} ( Σ_k W_{k,t}² M_kᵀ M_k u_{k,t} )
```

因 `M_k` 对角，可 **逐元素** 实现，不必真建 `NJD×NJD` 大矩阵：

```
# 对每个坐标 i：
denom_i = λ + Σ_k W_k² · M_k[i,i]
numer_i = Σ_k W_k² · M_k[i,i] · u_{k,t}[i]
u_t[i]  = numer_i / denom_i
```

### 2.7 Overall Inference（Sec 3.4）——主循环

论文文字版算法（补充材料中有完整伪代码；正文流程如下，复现必须严格按此顺序）：

```
输入: 文本 c^p，K 个约束 {类型, v_d, M_k}，预训练 MDM
可选: ProgMoGen warm-start 得到较好的 x_T 或中间状态

初始化:
  x_T ~ N(0,I) 或 warm-start 后的噪声/状态
  W_k ← W_init（如小正数或 0）
  q ← N(0,I)（criterion 用）
  EMA scale 初始化

for t = T, T-1, ..., 1:          # DDIM 反向
  1. 模型一步: pred_xstart = Tweedie(x_t)   # MDM p_mean_variance
  2. for k = 1..K:
       if criterion-based:
           用 Eq.9 + CEM 算 u_{k,t}（需 M 次采样/评估）
       else:  # objective-based
           用 Eq.10 对 pred_xstart 反传算 u_{k,t}
  3. 用 ẑ_T=pred_xstart 算各 c̃_{k,t}，更新 W_{k,t+1}（Eq.11）
  4. Allocator 合成 u_t（Eq.13）
  5. 将 u_t 注入扩散一步（对应 Eq.3/4 中的 h_t / u_t）得到 x_{t-1}
  6. （criterion）用 elite 更新 q

输出 x_0
```

### 2.8 Implementation Details（论文实验节，必须对齐）

| 项目 | 论文取值 |
|------|----------|
| Backbone | 官方 MDM，HumanML3D 预训练 |
| 采样 | **DDIM**（与 ProgMoGen benchmark 一致，即 `ddim100`） |
| Criterion 采样数 M | **16** |
| Proposal q | 初始标准高斯；每步 CEM 更新；**elite ratio 20%** |
| Objective 约束 | 跟随 ProgMoGen benchmark 构造；每步 Tweedie 得 `ẑ_T` |
| Criterion 例子 | foot-skating、success checks、物理仿真可行性等 |
| Warm-start | **采用 ProgMoGen / DNO 的优化策略作稳定初始化**，再跑 MIC |
| Benchmark | 同 ProgMoGen：HSI-1/2/3，GEO-1，HOI-1 |
| 指标 | Skating, Max Acc, C.Err, Unsucc. Rate, Pass(MuJoCo)；HSI-1 另报 FID/Diversity/R-Prec |

正文未给出数值但公式中出现、实现必须做成 **可配置超参** 的量：

| 符号 | 含义 | 建议实现 |
|------|------|----------|
| `γ` | 积分增益 | 配置文件；按任务扫 |
| `W_max` | 权重上界 | 配置文件 |
| `λ` | allocator 正则 | 配置文件 |
| `W_init` | 初始权重 | 小正数或 0 |
| EMA `α` | 违反度归一化 | 配置文件 |
| control scale | `σ(t)` 与离散步长乘子 | 与 DDIM `β/α̅` 对齐，可再乘全局 `guidance_scale` |
| warm-start iterations | ProgMoGen 预热步数 | 可少于完整评测 iterations |

---

## 3. 代码改动清单（文件级，怎么改）

建议在 ProgMoGen 旁开分支/目录，**不破坏原 `ddim_sample_loop_opt_fn`**，新增 MIC 采样路径。

### 3.1 新建模块（推荐结构）

```
progmogen/
  mic/
    __init__.py
    control_laws.py          # Eq.9 / Eq.10
    cem.py                   # Cross-Entropy Method
    feedback_regulator.py    # Eq.11
    control_allocator.py     # Eq.12–13
    constraints.py           # 每任务 objective + criterion + mask M_k
    tweedie.py               # 封装 pred_xstart
    inject.py                # 把 u_t 注入 DDIM 一步
    sample_loop_mic.py       # Sec 3.4 主循环
    warm_start.py            # 调用现有 ProgMoGen opt 若干步
  diffusion/
    ddim_mic.py              # 继承 InpaintingGaussianDiffusion，挂 MIC loop
  task_configs_mic/          # 每任务声明 constraint 列表
  script_eval/
    eval_task_*_mic.sh
  eval/
    main_eval_*_mic.py       # 补 Unsucc.Rate / Pass
```

### 3.2 `diffusion/ddim_mic.py`（核心改动）

**不要**再走「全程可微 + Adam(`noise_init`)」作为主方法；改为：

```python
def ddim_sample_loop_mic(self, model, shape, ..., constraints, mic_cfg):
    # 1) warm-start（可选）
    if mic_cfg.warm_start:
        noise_init = self.prog_mogen_warm_start(...)  # 少量 iterations 的现有 opt
        x = noise_init.detach()
    else:
        x = th.randn(shape, device=...)

    regulator = FeedbackRegulator(K, gamma=..., W_max=..., ema_alpha=...)
    cem_states = [CEMState() for _ in constraints if c.is_criterion]

    for i in reversed(range(self.num_timesteps)):
        t = tensor([i]*bs)
        out = self.p_mean_variance(model, x, t, ...)
        z_hat = out["pred_xstart"]                    # Tweedie

        u_list = []
        for k, c in enumerate(constraints):
            if c.type == "criterion":
                u_k = criterion_control_eq9(self, model, x, t, z_hat, c, cem_states[k], M=16)
            else:
                u_k = objective_control_eq10(self, x, t, z_hat, c)  # autograd
            u_list.append(u_k)

        W = regulator.update(z_hat, constraints)      # Eq.11
        u = allocate(u_list, W, [c.mask for c in constraints], lam=mic_cfg.lambda_)  # Eq.13

        x = self.ddim_step_with_control(model, x, t, out, u, eta=0.0)  # 注入

    return x
```

### 3.3 控制注入 `ddim_step_with_control`（对应 Eq.3）

论文：在 score 上加 `h_t`，且 `u_t = √β · h`。

两种等价落地（选一并统一全项目）：

**方案 A（推荐，复用 `condition_score`）**

```python
def cond_fn(x, t, **kwargs):
    # 返回 h_t；需与 u_t 换算: h = u / σ(t)  （σ=√β(T-t) 或离散等价尺度）
    return h_from_u(u, t)

out = condition_score(cond_fn, out_orig, x, t)
# 再标准 DDIM mean 更新
```

**方案 B（直接改 mean / 加在 sample 上）**

```python
# 在算完 DDIM mean_pred 后：
x_{t-1} = mean_pred + scale(t) * u_t
```

注意：criterion 的 Eq.9 给出的是 **噪声增量加权**，量纲接近 `dε`；objective 的 Eq.10 含 `σ(t)∇`。实现时必须 **按约束类型归一化到同一 `u` 空间**，再进 allocator，否则协调会失效。

### 3.4 `control_laws.py`：Eq.9 / Eq.10 实现细节

#### Objective（Eq.10）

```python
def objective_control_eq10(diffusion, x_t, t, x0_hat, constraint):
    x0 = x0_hat.detach().requires_grad_(True)
    # 解码到关节（与 ProgMoGen 一致）
    joints = diffusion.sample_to_joints(x0)   # 或对 latent 直接算，需与 mask 一致
    loss = constraint.v_d(joints)             # 标量能量 = 原 f_eval / f_loss
    g = autograd.grad(loss, x0)[0]            # ∇_{x0} v_d
    # log p(d|x0) ∝ -v_d → ∇ log p = -∇v
    sigma = diffusion.sigma_of_t(t)           # √β 或离散调度对应量
    u = - sigma * g                           # 再乘步长/guidance_scale
    return u.detach()
```

Tweedie：MDM 已是 **x0-prediction**，`pred_xstart` 即论文 `ẑ_T`。

#### Criterion（Eq.9 + CEM）

```python
def criterion_control_eq9(diffusion, model, x_t, t, x0_base, constraint, cem, M=16):
    # 1. 从 q=N(μ,Σ) 采 M 个 dε
    dε = cem.sample(M, shape=x_t.shape)      # 可对角 Σ 以省内存

    # 2. 对每个样本构造候选下一步/候选 x0
    #    实用做法（与 DPS/path-integral 文献一致）：
    #    在当前 DDIM 更新方向上扰动，或对 x0_hat 加噪再评估
    #    论文：对 dε^{(m)} 用 Tweedie 得 ẑ_T^{(m)}
    costs = []
    for m in range(M):
        x0_m = tweedie_with_noise_perturb(x_t, t, dε[m], model)  # 需与论文一致的采样定义
        E_m = constraint.E(x0_m)              # 可非可微：skating、success、sim
        log_w = -E_m + log_p0(dε[m]) - log_q(dε[m], cem.mu, cem.Sigma)
        costs.append((E_m, log_w, dε[m]))

    # 3. 稳定 softmax 得 π
    log_w = stack([...]); π = softmax(log_w)
    u = sum(π[m] * dε[m] for m in range(M))

    # 4. CEM: elite = top 20% by π（或 by -E）
    cem.update_elite(dε, π, elite_ratio=0.2)
    return u
```

**CEM 更新（正文描述）**：

- 选高权重 elite 子集（20%）
- 用 elite 样本更新 `μ, Σ`（对角协方差更稳：`Σ = diag(var(elite)) + εI`）
- 可对 `μ,Σ` 做动量平滑防塌缩

### 3.5 `feedback_regulator.py`（Eq.11）

```python
class FeedbackRegulator:
    def __init__(self, K, gamma, W_max, ema_alpha=0.95, W_init=0.0):
        self.W = ones(K) * W_init
        self.scale = ones(K)          # EMA of |c̃|
        ...

    def update(self, x0_hat, constraints):
        for k, c in enumerate(constraints):
            c_tilde = max(0.0, float(c.v_d(x0_hat)))   # 或 criterion 的违反度
            self.scale[k] = ema_alpha*self.scale[k] + (1-ema_alpha)*max(c_tilde, eps)
            c_norm = c_tilde / self.scale[k]
            self.W[k] = clip(self.W[k] + gamma * c_norm, 0, W_max)
        return self.W
```

### 3.6 `control_allocator.py`（Eq.13）

```python
def allocate(u_list, W, masks, lam):
    # u_list[k], masks[k]: 与 motion 同形 [1,C,1,T] 或展平
    numer = 0
    denom = lam
    for uk, Wk, Mk in zip(u_list, W, masks):
        numer = numer + (Wk**2) * Mk * uk
        denom = denom + (Wk**2) * Mk
    return numer / denom
```

### 3.7 `constraints.py`：每个任务的异构约束拆分

论文：同一运动上同时有 **objective** 与 **criterion**（如任务约束 + skating + success + 仿真）。

对每个任务定义列表，例如 **HSI-2**：

| k | 类型 | `v_d` / `E` | Scope `M_k` |
|---|------|-------------|-------------|
| 0 | objective | overhead barrier 能量（现有 `loss_overhead_barrier`） | 头关节关键帧 + 脚关键帧 |
| 1 | criterion | foot skating ratio / 是否超阈 | 双脚全序列（或接触帧） |
| 2 | criterion | success check（头/脚阈值是否全满足） | 与任务关节帧相同 |
| 3 | criterion（可选） | MuJoCo / 物理稳定性（论文 Fig.4） | 全身或根轨迹 |

**HSI-1**：头高 `equal` 为 objective；skating / success(0.05) / Pass 为 criterion。  
**HSI-3**：有界行走 objective；越界 success、skating 等为 criterion。  
**GEO-1 / HOI-1**：平面/手腕目标为 objective（可保留 relax 预处理）；criterion 同理。

`M_k` 构造规则（按论文 “joints and frames”）：

- 只在约束相关的 `(joint, frame, xyz)` 上为 1，其余 0  
- 全序列约束（skating）：脚关节、全部有效长度帧为 1  
- 关键帧约束：对应 `t_0, t_mid, t_end` 为 1  

若控制定义在 **263-d latent** 而非 joints：需把关节 mask 映射到 HumanML3D feature 索引（根轨迹、RIC 等），或 **始终在 joints 空间算 u 再映射回 latent**（更清晰，推荐）。

### 3.8 Warm-start（论文明确要求）

```python
def prog_mogen_warm_start(diffusion, model, ..., short_iterations):
    # 直接调用现有 ddim_sample_loop_opt_fn，但 iterations 取较小值
    # 或跑完完整 ProgMoGen 后，用其 noise_init / x0 作为 MIC 起点
    return noise_init_or_x
```

推荐流水线（与论文 “stable initialization” 一致）：

1. **Phase A**：现有 ProgMoGen `f_loss` 优化 `noise_init`（可用原 `lr/iterations` 或减半）  
2. **Phase B**：从优化后的 `noise_init` 出发，**不再反传整段**，改为 MIC 逐步控制（此时可同时打开 criterion）

### 3.9 入口与脚本

复制并修改：

- `script_eval/eval_task_hsi2.sh` → `eval_task_hsi2_mic.sh`  
  - `task_config` → `task_configs_mic/eval_task_hsi2_mic_config.py`  
  - 调用 `tasks/eval_task_mic.py`（新），内部 `ddim_sample_loop_mic`

`eval_task_mic.py`：大体克隆 `eval_task.py` 的数据加载 / 保存 `gen.npy`，仅替换 sample 函数绑定。

### 3.10 评测代码补齐（论文表格需要，ProgMoGen 部分缺失）

论文指标：Skating ↓, Max Acc ↓, C.Err ↓, **Unsucc. Rate ↓**, **Pass ↑**。

| 任务 | ProgMoGen 现有 | MIC 复现需补 |
|------|----------------|--------------|
| HSI-1 | skate, jittor_max, mae, **unsuccess_rate** | + MuJoCo Pass；FID 脚本已有 |
| HSI-2 | skate, jittor_max, C.Err | **+ unsuccess_rate + Pass** |
| HSI-3 | 同 HSI-2 | **+ unsuccess_rate + Pass** |
| GEO-1 / HOI-1 | skate, jittor, C.Err | **+ unsuccess_rate + Pass** |

**Unsuccess 判定（对齐 ProgMoGen HSI-1 精神 + 各任务语义）**：

- HSI-1：三帧头高误差均 `<0.05` 才算成功（已有）  
- HSI-2：建议定义（与 barrier 规则一致）：  
  - `head_y(0)>1.5`, `head_y(T-1)>1.5`, `head_y(mid)<0.5`, 中间脚不浮空等；可对不等式留小阈值 ε  
- HSI-3：全程关节 X,Z 在 `[-1,1]` 内  
- GEO-1：腕到平面距离均值/最大 < 阈值  
- HOI-1：起终点腕位置误差 < 阈值  

**Max Acc**：即现有 `get_jittor_stat(..., order=2, stat_type="max")`。  
**Skating**：`calculate_skating_ratio`。  
**Pass**：论文用 MuJoCo 物理仿真检查；仓库内 **无现成脚本**，需按 PhysDiff / Embodied 相关设定接入（论文引用 MuJoCo + Luo et al.）。若短期无法复现仿真，先实现其余指标，Pass 单独模块化。

---

## 4. 与 ProgMoGen 逐步对照的“必改点”清单

1. **新增逐步控制采样环**，替代（主路径上）整段反传优化。  
2. **实现 Eq.9 CEM criterion 控制**（M=16, elite 20%）。  
3. **实现 Eq.10 objective 控制**（Tweedie + ∇v_d）。  
4. **实现 Eq.11 regulator + Eq.13 allocator**。  
5. **为每个任务声明多约束 + mask M_k**。  
6. **Warm-start 接现有 `ddim_sample_loop_opt_fn`**。  
7. **评测补 Unsucc.Rate / Pass**，才能对齐论文 Table 1–2。  
8. **保持 MDM 权重、DDIM100、HumanML3D、文本 split 与 ProgMoGen 脚本一致**，否则数字不可比。

**不要改**（除非做消融）：

- MDM 网络结构与 checkpoint  
- HumanML3D 归一化 Mean/Std  
- `recover_from_ric` 运动表示  
- 原 atomic 可微约束的几何定义（objective 应复用）

---

## 5. 推荐实现顺序（工程节奏）

### Phase 1 — 单约束 objective MIC（验证注入正确）

1. 只在 HSI-1 上实现 Eq.10 + 注入 DDIM  
2. 关闭 CEM / allocator（K=1）  
3. 对比：纯 ProgMoGen vs warm-start+MIC vs 仅 MIC  
4. 盯 C.Err / Unsucc.Rate 是否下降且 Skating 不爆

### Phase 2 — Criterion Eq.9

1. 加 skating 为 criterion（用现有 `calculate_skating_ratio` 作 `E`）  
2. M=16, CEM 20%  
3. 与 Baseline A（surrogate 梯度）对比，对应论文 Table 3 精神

### Phase 3 — 协调机制

1. K≥2：任务 objective + skating（+ success）  
2. 实现 regulator + allocator  
3. 消融：`w/o regulation` / `w/o allocation` / `w/o coordination`（对齐 Table 5）

### Phase 4 — 全任务 + 评测

1. HSI-2/3, GEO-1, HOI-1（relax 任务：可 MIC 前保留 ProgMoGen relax，或把硬约束直接当 objective）  
2. 补齐 Unsucc / Pass  
3. 对齐 Table 1–2 协议（样本数：open-set 脚本多为 32；HSI-1 为 512）

---

## 6. 超参数配置模板（建议 `mic_config.yaml`）

```yaml
ddim_steps: 100
eta: 0.0

warm_start:
  enable: true
  lr: 0.005          # 跟原 task_config
  iterations: 50     # 可少于原 100

criterion:
  M: 16
  elite_ratio: 0.2
  cem_eps: 1.0e-4    # Σ 抖动
  cem_momentum: 0.5  # 可选

regulator:
  gamma: 0.1         # 需扫
  W_max: 10.0
  W_init: 0.0
  ema_alpha: 0.95

allocator:
  lambda: 1.0        # 需扫

guidance:
  # 把 u 映射进 DDIM 的全局尺度
  scale: 1.0

seed_policy: same_as_progmogen  # np_seed 与脚本一致
```

---

## 7. 论文消融应对应的实现开关

| 论文设定 | 实现开关 |
|----------|----------|
| Baseline A：criterion→surrogate+梯度 | 强制所有约束走 Eq.10 |
| Separate handling | objective 用 ProgMoGen 反传；criterion 用 Eq.9；无统一 allocator |
| w/o regulation | 固定 `W_k`，只跑 Eq.13 |
| w/o allocation | `u = Σ_k W_k u_k`（或平均），不用 mask 最小二乘 |
| w/o coordination | `u = mean_k(u_k)` |
| MIC full | Eq.9/10 + Eq.11 + Eq.13 + warm-start |

---

## 8. 验证正确性的检查清单

- [ ] 单步：无约束时 MIC 退化为标准 DDIM，结果与 `ddim_sample_loop` 一致（`u=0`）  
- [ ] Objective-only：loss 随步下降；与 DPS 行为类似  
- [ ] Criterion-only：`E` 不可微也能改运动；M↑ 应更稳  
- [ ] CEM：elite 平均代价应下降；`q` 不数值爆炸  
- [ ] Regulator：持续违反时 `W` 升；满足后停止升  
- [ ] Allocator：两约束冲突时，mask 外分量接近 0  
- [ ] Warm-start off/on：on 应更稳、C.Err 更好  
- [ ] 指标：HSI-2 能打出 Skating / MaxAcc / C.Err / Unsucc / Pass 五列  

---

## 9. 关键公式速查

| 编号 | 内容 |
|------|------|
| Eq.1 | 无控反向 VP-SDE |
| Eq.2 | 约束后验 |
| Eq.3 | 含 `h_t` 的受控扩散 |
| Eq.4 | 控制形式 `u_t` |
| Eq.5 | 最优控制目标 |
| Eq.6 | `u* = σ ∇ log ψ` |
| Eq.7 | criterion 无梯度形式 |
| Eq.8 | objective 梯度形式 |
| Eq.9 | criterion 重要性采样实例化 + CEM |
| Eq.10 | objective Tweedie 梯度实例化 |
| Eq.11 | Feedback regulator |
| Eq.12–13 | Control allocator 及闭式解 |

---

## 10. 与仓库文件的快速索引

| 用途 | 路径 |
|------|------|
| ProgMoGen 主优化环 | `progmogen/diffusion/ddim.py` → `ddim_sample_loop_opt_fn` |
| Relax 优化环 | `progmogen/diffusion/ddim_relax.py` |
| DDIM / condition_score | `progmogen/diffusion/gaussian_diffusion_v2.py` |
| 约束原子 | `progmogen/atomic_lib/math_utils.py` |
| HSI-2 任务 | `progmogen/task_configs_eval/eval_task_hsi2_config.py` |
| HSI-2 脚本 | `progmogen/script_eval/eval_task_hsi2.sh` |
| Skating / jittor | `progmogen/eval/metrics.py`, `metrics2.py` |
| HSI-1 unsuccess | `progmogen/eval/main_eval_hsi1.py` |
| 评测入口 | `progmogen/tasks/eval_task.py` |

---

## 11. 仓库骨架（已落地）

已在仓库中搭好可运行骨架（逻辑按本文第 3 节），目录：

```
progmogen/mic/                          # Eq.9–13 核心
progmogen/diffusion/ddim_mic.py         # 标准任务 MIC
progmogen/diffusion/ddim_relax_mic.py   # GEO/HOI MIC
progmogen/task_configs_mic/             # HSI-1/2/3, GEO-1, HOI-1
progmogen/tasks/eval_task_mic.py
progmogen/tasks/eval_task_hsi1_mic.py
progmogen/tasks/eval_task_goal_relaxed_mic.py
progmogen/script_eval/eval_task_*_mic.sh
progmogen/script_eval/eval_all_mic.sh
progmogen/eval/main_eval_*_mic.py       # + unsuccess_rate
```

各任务脚本（在 `progmogen/` 下，conda env `mdm`）：

```bash
cd progmogen
sh script_eval/eval_task_hsi1_mic.sh          # known, 512 samples
sh script_eval/eval_task_hsi2_mic.sh
sh script_eval/eval_task_hsi3_mic.sh
sh script_eval/eval_task_geo1_relax_mic.sh
sh script_eval/eval_task_hoi1_relax_mic.sh
# 或一次跑全部：
sh script_eval/eval_all_mic.sh
```

骨架已接通：warm-start → 逐步 Eq.9/10 → Eq.11 → Eq.13 → `condition_score` 注入。后续需按实验细调 `γ/λ/W_max`、criterion 的 Tweedie 扰动定义、以及 MuJoCo Pass。

---

## 12. 总结

复现 MIC = 在 ProgMoGen 的 **同一 MDM+DDIM+任务定义** 上，把约束执行机制从：

> 「整段 denoising → 可微 loss → 优化初始噪声」

换成：

> 「每步 Tweedie →（Eq.9 或 Eq.10）得各约束 `u_k` → Eq.11 调权重 → Eq.13 分配 → 注入 DDIM」，并以 ProgMoGen 优化作 warm-start；评测补齐 Unsucc.Rate 与 Pass。

按本文第 3–5 节文件与阶段推进，即可在现有仓库上完整落地论文正文中的 MIC。
