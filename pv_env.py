"""
PV-BESS Environment (constraint-aware, CVaR tail-risk in reward).

- Violation / p_applied / SoC 动力学同前。
- 风险：滚动 loss_buffer（跨 episode 保留，maxlen 可配），每步
      loss = alpha*|P_grid-P_target| + beta*max(0,-economic_reward)
      cvar = mean( buffer 中 worst ceil(alpha_tail*n) 项 )，alpha_tail 默认 0.1
      risk_penalty = lambda_cvar * cvar
  观测中风险特征为当前 cvar（策略可见尾部压力）。
- Reward：
      tracking_reward = -tracking_scale * |p_grid - target|  （默认 tracking_scale=5）
      constraint_penalty：**意图越界连续惩罚**（若按意图 SoC 会出界、爬坡超额），系数 50/15，
      使惩罚能与 tracking 抗衡，避免「靠环境裁剪白嫖 tracking reward」
      smooth_penalty = smooth_coef * |p_applied - p_prev|  （默认 0.1，w_smooth=1）
      soc_center：w_soc_center * (-0.5*(SoC_next-0.5)^2)，软引导电量维持在中间区
  观测含归一化约束裕度（SoC 上下界距、功率归一化裕度）+ CVaR 标量。
  可调 set_lambda_cvar() 做分阶段训练。
"""

from __future__ import annotations

import gzip
import pickle
from collections import deque
from typing import Dict, Tuple, Union

import gym
import numpy as np
from gym import spaces


ObsType = Union[np.ndarray, Dict[str, np.ndarray]]


class PVBESSEnv(gym.Env):
    """
    PV + Battery dispatch environment.

    Action (continuous):
        a ∈ [-1, 1] → p_battery_desired = a * P_rated  (MW)

    Hard constraints (environment-enforced):
        p_battery = clip(p_battery, -P_rated, P_rated)
        p_battery = clip(p_battery, prev_power - ramp_limit, prev_power + ramp_limit)
        soc_next  = clip(soc_next, SOC_MIN, SOC_MAX)

    Observation:
        - mode="simple": flat vector
        - mode="multiscale": Dict with short_term (CNN), long_term (LSTM), current (MLP)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        data_path: str,
        mode: str = "simple",
        episode_length: int = 288,
        # backward-compat args (kept to avoid breaking old runners)
        cvar_alpha: float = 0.1,
        # physical parameters
        capacity_mwh: float = 1.0,
        power_rated_mw: float = 0.5,
        soc_min: float = 0.1,
        soc_max: float = 0.9,
        eta_charge: float = 0.95,
        eta_discharge: float = 0.95,
        dt_hours: float = 5.0 / 60.0,
        ramp_rate_max: float = 0.25,  # fraction of P_rated per step
        # targets & prices
        target_power_mw: float = 0.0,
        price_buy: float = 100.0,
        price_sell: float = 80.0,
        price_peak_multiplier: float = 1.5,
        # risk: CVaR over rolling loss_buffer（alpha_tail 与 cvar_alpha 一致，默认 0.1）
        alpha_track_loss: float = 1.0,
        beta_econ_loss: float = 1.0,
        loss_buffer_maxlen: int = 400,
        loss_ema_decay: float = 0.95,
        loss_var_decay: float = 0.95,
        k_tail: float = 1.0,
        tail_scale: float = 1.0,
        lambda_cvar: float = 1.0,
        lambda_smooth: float = 0.1,
        w_const: float = 1.0,
        const_intent_flat: float = 0.0,  # 新增：意图违规定额惩罚（每步触发即扣）
        w_track: float = 1.0,
        w_econ: float = 1.0,
        w_risk: float = 1.0,
        w_smooth: float = 1.0,
        w_soc_center: float = 1.0,
        tracking_scale: float = 5.0,
        smooth_coef: float = 0.1,
        smooth_scale: float = 10.0,
        economic_scale: float = 0.3,
        # Safe RL / debug: violation 基于未裁剪的 p_desired；debug_violation 放大意图以验证检测链路
        debug_violation: bool = False,
        wandb_log_interval: int = 0,
        debug_print_interval: int = 0,
        debug_p_desired_buf_max: int = 1000,
        # ── 消融实验开关 ──────────────────────────────────────────────────────
        # Ablation 1: 动态 EMA 风险代理 vs 静态重罚
        #   True  (默认/DR3L_full): 用 EMA 滚动代理驱动 risk_penalty
        #   False (Static Heavy):  旁路 EMA，直接用 static_risk_weight × 当步 loss
        use_dynamic_ema: bool = True,
        static_risk_weight: float = 10.0,
        # Ablation 2: EMA 时间视野（衰减系数 beta）
        #   0.9  → 默认最优；0.0 → 无记忆（即时 loss）；0.99 → 极长记忆
        #   公式: loss_ema = ema_beta * loss_ema + (1 - ema_beta) * current_loss
        ema_beta: float = 0.9,
        # Ablation 3: 意图惩罚 vs 状态惩罚
        #   True  (默认/DR3L_full): 约束惩罚基于原始 Actor 输出 (p_desired)
        #   False (State Penalty):  约束惩罚基于环境裁剪后实际执行的动作 (p_applied)
        use_intent_penalty: bool = True,
        # Ablation 2 extra: 移动平均（MA）风险代理，与 EMA 对照
        #   use_moving_average=True 时，risk_penalty 改用 MA(last ma_window steps)
        use_moving_average: bool = False,
        ma_window: int = 10,
        # Ablation 3 extra: 测试期移除安全投影（暴露策略是否已内化约束边界）
        #   train 阶段保持 False；test 阶段设 True 可验证「意图惩罚 → 内化安全」假设
        disable_projection: bool = False,
        # ── Lagrangian 约束优化开关 ───────────────────────────────────────────
        # use_lagrangian_constraint=True 时：
        #   - reward 中移除 constraint_penalty（w_const 项）
        #   - info["violation_magnitude"] 返回原始约束违规幅度供算法层 λ 更新
        #   - constraint 由算法层 Lagrangian 乘子 λ 显式控制（而非 reward shaping）
        # use_lagrangian_constraint=False（默认）：兼容原 DR3L 行为，完全不变
        use_lagrangian_constraint: bool = False,
        **kwargs,
    ):
        super().__init__()

        if mode not in {"simple", "multiscale"}:
            raise ValueError(f"mode must be 'simple' or 'multiscale', got {mode!r}")

        self.mode = mode
        self.episode_length = int(episode_length)

        # Backward compatibility for runners that pass cvar_alpha.
        self.cvar_alpha = float(cvar_alpha)

        # Data
        with gzip.open(data_path, "rb") as f:
            self.samples = pickle.load(f)

        # Physical params
        self.capacity_mwh = float(capacity_mwh)
        self.power_rated_mw = float(power_rated_mw)
        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)
        self.eta_charge = float(eta_charge)
        self.eta_discharge = float(eta_discharge)
        self.dt_hours = float(dt_hours)
        self.ramp_rate_max = float(ramp_rate_max)

        # Targets & prices
        self.target_power_mw = float(target_power_mw)
        self.price_buy = float(price_buy)
        self.price_sell = float(price_sell)
        self.price_peak_multiplier = float(price_peak_multiplier)

        # Risk / loss parameters (Markov state)
        self.alpha_track_loss = float(alpha_track_loss)
        self.beta_econ_loss = float(beta_econ_loss)
        self.loss_ema_decay = float(loss_ema_decay)
        self.loss_var_decay = float(loss_var_decay)
        self.k_tail = float(k_tail)
        self.tail_scale = float(tail_scale)
        self.lambda_cvar = float(lambda_cvar)
        self.lambda_smooth = float(lambda_smooth)

        # Smoothness & reward weights
        self.w_const = float(w_const)
        self.const_intent_flat = float(const_intent_flat)
        self.w_track = float(w_track)
        self.w_econ = float(w_econ)
        self.w_risk = float(w_risk)
        self.w_smooth = float(w_smooth)
        self.w_soc_center = float(w_soc_center)

        # Scaling knobs to stabilize reward magnitudes
        self.tracking_scale = float(tracking_scale)
        self.smooth_coef = float(smooth_coef)
        self.smooth_scale = float(smooth_scale)
        self.economic_scale = float(economic_scale)

        self.debug_violation = bool(debug_violation)
        self.wandb_log_interval = int(wandb_log_interval)
        self.debug_print_interval = int(debug_print_interval)
        self.debug_p_desired_buf_max = max(1, int(debug_p_desired_buf_max))

        # 消融实验开关（Ablation 1-3）
        self.use_dynamic_ema = bool(use_dynamic_ema)
        self.static_risk_weight = float(static_risk_weight)
        self.ema_beta = float(np.clip(ema_beta, 0.0, 1.0))
        self.use_intent_penalty = bool(use_intent_penalty)
        self.use_moving_average = bool(use_moving_average)
        self.ma_window = max(1, int(ma_window))
        self.disable_projection = bool(disable_projection)
        self.use_lagrangian_constraint = bool(use_lagrangian_constraint)
        # EMA & MA 风险代理状态（跨 episode 保留，与 _loss_buffer 一致）
        self.loss_ema: float = 0.0
        self._ma_loss_buffer: deque = deque(maxlen=self.ma_window)
        self.loss_ma: float = 0.0

        # Action space: normalized in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.loss_buffer_maxlen = max(50, int(loss_buffer_maxlen))
        self._loss_buffer: deque[float] = deque(maxlen=self.loss_buffer_maxlen)

        # Observation: rolling CVaR + 约束裕度（SoC 上下、功率裕度）
        self._risk_feature_dim = 1
        self._constraint_feature_dim = 3

        if self.mode == "simple":
            # 6 + soc + pv + forecast + prev_a_norm + cvar + 3 constraint margins
            obs_dim = (
                6
                + 1
                + 1
                + 1
                + 1
                + self._risk_feature_dim
                + self._constraint_feature_dim
            )
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
        else:
            # current: 6 + soc + cvar + 3 margins => 11
            cur_dim = 6 + 1 + self._risk_feature_dim + self._constraint_feature_dim
            self.observation_space = spaces.Dict(
                {
                    "short_term": spaces.Box(low=-np.inf, high=np.inf, shape=(6, 12), dtype=np.float32),
                    "long_term": spaces.Box(low=-np.inf, high=np.inf, shape=(288, 2), dtype=np.float32),
                    "current": spaces.Box(low=-np.inf, high=np.inf, shape=(cur_dim,), dtype=np.float32),
                }
            )

        # Episode state (initialized in reset)
        self.current_idx = 0
        self.step_count = 0
        self.soc = 0.5
        self.p_battery_prev = 0.0

        self.cvar = 0.0

        # Logging accumulators (episode-only; NOT used in reward)
        self.episode_tracking_rewards = []
        self.episode_economic_rewards = []
        self.episode_risk_penalties = []
        self.episode_smooth_penalties = []
        self.episode_constraint_penalties = []
        self.episode_soc_center_terms = []
        self.episode_total_rewards = []
        self.episode_losses = []
        self.episode_soc_violation_flags = []
        self.episode_ramp_violation_flags = []
        self.episode_power_violation_flags = []
        self.episode_intent_soc_violation_flags = []
        self.episode_intent_ramp_violation_flags = []
        self.episode_soc_clips = 0
        self.episode_ramp_clips = 0

        # 跨 episode 计数（用于 debug 打印 / WandB step）；reset 不清零
        self.global_env_step = 0
        self._p_desired_buf: deque[float] = deque(maxlen=self.debug_p_desired_buf_max)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _hour_from_step(self) -> float:
        return (self.step_count * self.dt_hours) % 24.0

    def _economic_reward(self, p_grid_mw: float, _hour: float) -> float:
        """economic_reward = revenue - cost (per spec); × dt_hours for $/interval scaling."""
        if p_grid_mw >= 0.0:
            cost = p_grid_mw * self.price_buy
            revenue = 0.0
        else:
            cost = 0.0
            revenue = (-p_grid_mw) * self.price_sell

        econ = (revenue - cost) * self.dt_hours
        return float(self.economic_scale * econ)

    def _delta_soc(self, p_battery_mw: float) -> float:
        """Δsoc from battery power p (MW), same sign convention as before."""
        if p_battery_mw > 0.0:
            energy_change_mwh = -p_battery_mw / self.eta_discharge * self.dt_hours
        else:
            energy_change_mwh = -p_battery_mw * self.eta_charge * self.dt_hours
        return float(energy_change_mwh / self.capacity_mwh)

    def set_lambda_cvar(self, value: float) -> None:
        """分阶段训练时更新环境风险权重（不改变 loss_buffer）。"""
        self.lambda_cvar = float(value)

    def _tail_cvar_from_buffer(self) -> float:
        """Worst ceil(cvar_alpha * n) 步损失的样本均值（CVaR 代理）。"""
        n = len(self._loss_buffer)
        if n == 0:
            return 0.0
        alpha = float(self.cvar_alpha)
        alpha = min(max(alpha, 1e-6), 1.0)
        k = max(1, int(np.ceil(alpha * n)))
        arr = np.asarray(self._loss_buffer, dtype=np.float64)
        if k >= n:
            return float(np.mean(arr))
        part = np.partition(arr, -k)[-k:]
        return float(np.mean(part))

    def _update_risk_state(self, p_grid_mw: float, economic_reward: float) -> Tuple[float, float]:
        tracking_term = abs(p_grid_mw - self.target_power_mw)
        loss = float(
            self.alpha_track_loss * tracking_term
            + self.beta_econ_loss * max(0.0, -economic_reward)
        )
        self._loss_buffer.append(loss)
        self.cvar = self._tail_cvar_from_buffer()
        # EMA 风险代理更新（Ablation 2: ema_beta 控制时间视野）
        # loss_ema = ema_beta * loss_ema + (1 - ema_beta) * current_loss
        self.loss_ema = float(self.ema_beta * self.loss_ema + (1.0 - self.ema_beta) * loss)
        # MA 风险代理更新（Ablation 2 extra: MA 对照）
        self._ma_loss_buffer.append(loss)
        self.loss_ma = float(np.mean(self._ma_loss_buffer))
        return loss, self.cvar

    def _risk_penalty(self, current_loss: float = 0.0) -> float:
        """计算风险惩罚。

        Ablation 1 (use_dynamic_ema):
          True  → 动态代理（EMA 或 MA）: lambda_cvar * max(0, surrogate)
          False → 静态重罚: static_risk_weight * max(0, current_loss)

        Ablation 2 (use_moving_average, 仅当 use_dynamic_ema=True 有效):
          False → EMA 代理: surrogate = loss_ema
          True  → MA 代理:  surrogate = loss_ma
        """
        if self.use_dynamic_ema:
            surrogate = self.loss_ma if self.use_moving_average else self.loss_ema
            return float(self.lambda_cvar * max(0.0, surrogate))
        else:
            return float(self.static_risk_weight * max(0.0, current_loss))

    def _constraint_margin_features(self) -> np.ndarray:
        """归一化 SoC 距上下界裕度 + 功率归一化裕度（供策略感知约束松紧）。"""
        span = max(self.soc_max - self.soc_min, 1e-6)
        soc_margin_low = float((self.soc - self.soc_min) / span)
        soc_margin_high = float((self.soc_max - self.soc) / span)
        pr = max(self.power_rated_mw, 1e-6)
        ramp_margin = float(1.0 - abs(self.p_battery_prev) / pr)
        return np.array([soc_margin_low, soc_margin_high, ramp_margin], dtype=np.float32)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self) -> ObsType:
        # long_term 需 288 步历史：起始索引至少为 288，避免 LSTM 输入越界/缺历史
        min_start = 288
        high = len(self.samples) - self.episode_length
        if high <= min_start:
            raise ValueError(
                f"样本太短：需要 len(samples) > episode_length + 288，"
                f"当前 len={len(self.samples)}, episode_length={self.episode_length}"
            )
        self.current_idx = int(np.random.randint(min_start, high))
        self.step_count = 0
        self.soc = 0.5
        self.p_battery_prev = 0.0

        # 不重置 _loss_buffer / cvar：跨 episode 滚动 CVaR（分阶段/长训一致）

        # reset episode logs
        self.episode_tracking_rewards = []
        self.episode_economic_rewards = []
        self.episode_risk_penalties = []
        self.episode_smooth_penalties = []
        self.episode_constraint_penalties = []
        self.episode_soc_center_terms = []
        self.episode_total_rewards = []
        self.episode_losses = []
        self.episode_soc_violation_flags = []
        self.episode_ramp_violation_flags = []
        self.episode_power_violation_flags = []
        self.episode_intent_soc_violation_flags = []
        self.episode_intent_ramp_violation_flags = []
        self.episode_soc_clips = 0
        self.episode_ramp_clips = 0

        return self._get_observation()

    def _get_observation(self) -> ObsType:
        sample = self.samples[self.current_idx + self.step_count]

        risk_features = np.array([self.cvar], dtype=np.float32)
        constraint_features = self._constraint_margin_features()

        if self.mode == "simple":
            pv_actual = float(sample["pv_actual"])
            pv_forecast = float(sample.get("pv_forecast", pv_actual))
            obs = np.concatenate(
                [
                    np.asarray(sample["current_state"], dtype=np.float32),
                    np.asarray([self.soc], dtype=np.float32),
                    np.asarray([pv_actual], dtype=np.float32),
                    np.asarray([pv_forecast], dtype=np.float32),
                    np.asarray([self.p_battery_prev / self.power_rated_mw], dtype=np.float32),
                    risk_features,
                    constraint_features,
                ]
            ).astype(np.float32)
            return obs

        # multiscale
        current = np.concatenate(
            [
                np.asarray(sample["current_state"], dtype=np.float32),
                np.asarray([self.soc], dtype=np.float32),
                risk_features,
                constraint_features,
            ]
        ).astype(np.float32)

        return {
            "short_term": np.asarray(sample["short_term"], dtype=np.float32).T,  # (6, 12)
            "long_term": np.asarray(sample["long_term"], dtype=np.float32),     # (288, 2)
            "current": current,
        }

    def step(self, action: np.ndarray) -> Tuple[ObsType, float, bool, Dict]:
        self.global_env_step += 1

        # 不对 action 做 clip：违规检测与策略输出解耦（Gym 仍声明 Box[-1,1]）
        a = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        p_desired = a * self.power_rated_mw
        if self.debug_violation:
            p_desired *= 2.0

        p_prev = float(self.p_battery_prev)
        ramp_limit = self.ramp_rate_max * self.power_rated_mw

        soc_raw_desired = float(self.soc + self._delta_soc(p_desired))
        soc_violation = bool(soc_raw_desired < self.soc_min or soc_raw_desired > self.soc_max)
        ramp_violation = bool(abs(p_desired - p_prev) > ramp_limit)
        power_violation = bool(abs(p_desired) > self.power_rated_mw)

        p_power_clipped = float(np.clip(p_desired, -self.power_rated_mw, self.power_rated_mw))
        power_clipped = bool(abs(p_desired - p_power_clipped) > 1e-9)
        p_before_ramp = p_power_clipped

        if self.disable_projection:
            # Ablation 3 extra（测试期）：移除爬坡/SoC 运行约束投影
            # 仅保留额定功率硬限（物理不可超越），暴露策略是否内化安全边界
            p_applied = p_power_clipped
            ramp_clipped = False
        else:
            p_applied = float(np.clip(p_power_clipped, p_prev - ramp_limit, p_prev + ramp_limit))
            ramp_clipped = bool(p_applied != p_before_ramp)
            if ramp_clipped:
                self.episode_ramp_clips += 1

        soc_raw = float(self.soc + self._delta_soc(p_applied))
        if self.disable_projection:
            # 仅保留 [0, 1] 物理极限（电量不能为负/超额）
            soc_next = float(np.clip(soc_raw, 0.0, 1.0))
        else:
            soc_next = float(np.clip(soc_raw, self.soc_min, self.soc_max))
        soc_clipped = bool(soc_next != soc_raw)
        if soc_clipped and not self.disable_projection:
            self.episode_soc_clips += 1

        pr = max(self.power_rated_mw, 1e-6)
        soc_span = max(self.soc_max - self.soc_min, 1e-6)

        # Ablation 3: use_intent_penalty
        #   True  (DR3L_full/默认) → 路径 A：按 Actor 原始输出 p_desired 计算越界惩罚
        #   False (State Penalty)  → 路径 B：按实际执行动作 p_applied 的状态越界计算惩罚
        if self.use_intent_penalty:
            # 路径 A：意图惩罚——幅度压过 tracking，强迫网络学习物理边界
            soc_excess_normalized = float(
                max(0.0, soc_raw_desired - self.soc_max) + max(0.0, self.soc_min - soc_raw_desired)
            ) / soc_span
            ramp_excess = float(max(0.0, abs(p_desired - p_prev) - ramp_limit) / pr)
            # 定额惩罚触发基于意图违规标志
            _flat_trigger_soc = int(soc_violation)
            _flat_trigger_ramp = int(ramp_violation)
        else:
            # 路径 B：状态惩罚——仅惩罚环境物理截断后仍发生的实际越界
            # SoC：p_applied 执行后的原始 SoC（截断前）可能超界
            soc_excess_normalized = float(
                max(0.0, soc_raw - self.soc_max) + max(0.0, self.soc_min - soc_raw)
            ) / soc_span
            # 爬坡：p_applied 已由环境保证满足爬坡限制，故此项恒为 0
            ramp_excess = 0.0
            # 定额惩罚触发基于实际裁剪标志
            _flat_trigger_soc = int(soc_clipped)
            _flat_trigger_ramp = int(ramp_clipped)

        constraint_penalty = float(50.0 * soc_excess_normalized + 15.0 * ramp_excess)

        soc_clip_depth_norm = float(
            max(0.0, self.soc_min - soc_raw) + max(0.0, soc_raw - self.soc_max)
        ) / soc_span
        power_clip_magnitude = float(abs(p_desired - p_power_clipped) / pr)
        ramp_clip_magnitude = float(abs(p_power_clipped - p_applied) / pr)

        sample = self.samples[self.current_idx + self.step_count]
        pv_actual = float(sample["pv_actual"]) * 2.0  # project-specific scaling
        pv_forecast = float(sample.get("pv_forecast", sample["pv_actual"])) * 2.0

        hour = self._hour_from_step()
        load = float(0.5 + 0.5 * np.sin(2 * np.pi * (hour - 6.0) / 24.0))

        p_grid = load - pv_actual - p_applied

        tracking_term = abs(p_grid - self.target_power_mw)
        tracking_reward = -self.tracking_scale * tracking_term

        economic_reward = self._economic_reward(p_grid, hour)

        loss, cvar = self._update_risk_state(p_grid, economic_reward)
        # Ablation 1: 传入当步 loss 供静态重罚模式使用
        risk_penalty = self._risk_penalty(current_loss=loss)

        smooth_penalty = float(self.smooth_coef * abs(p_applied - p_prev))

        rw_track = self.w_track * tracking_reward
        rw_econ = self.w_econ * economic_reward
        rw_risk = -self.w_risk * risk_penalty
        rw_smooth = -self.w_smooth * smooth_penalty
        # 定额惩罚：触发标志由 use_intent_penalty 决定（意图/实际）
        intent_flat_penalty = float(
            self.const_intent_flat * (_flat_trigger_soc + _flat_trigger_ramp)
        )
        rw_const = -self.w_const * constraint_penalty - intent_flat_penalty

        # SoC 回归中值软引导（路径 B）：恒 ≤0，远离 0.5 时更负
        soc_center_shaping = float(-0.5 * (soc_next - 0.5) ** 2)
        rw_soc_center = float(self.w_soc_center * soc_center_shaping)

        # Lagrangian 模式：constraint 从 reward 中完全移除，由算法层 λ 控制
        # 原始 DR3L 模式（兼容）：constraint_penalty 保留在 reward 中
        if self.use_lagrangian_constraint:
            total_reward = rw_track + rw_econ + rw_risk + rw_smooth + rw_soc_center
        else:
            total_reward = rw_track + rw_econ + rw_risk + rw_smooth + rw_const + rw_soc_center

        self.soc = soc_next
        self.p_battery_prev = p_applied
        self.step_count += 1

        done = self.step_count >= self.episode_length

        info: Dict = {
            "soc": self.soc,
            "p_battery": p_applied,
            "p_desired": p_desired,
            "p_grid": p_grid,
            "pv_actual": pv_actual,
            "pv_forecast": pv_forecast,
            "load": load,
            "hour": hour,
            "soc_clipped": soc_clipped,
            "ramp_clipped": ramp_clipped,
            # 主指标：实际发生的裁剪（与 constraint_penalty 一致）
            "soc_violation": int(soc_clipped),
            "ramp_violation": int(ramp_clipped),
            "power_violation": int(power_clipped),
            "intent_soc_violation": int(soc_violation),
            "intent_ramp_violation": int(ramp_violation),
            "intent_power_violation": int(power_violation),
            # ── Lagrangian 约束接口 ──────────────────────────────────────────
            # violation: 当步是否发生意图越界（SoC 或 ramp），供 λ 判断约束是否激活
            "violation": int(soc_violation or ramp_violation),
            # violation_magnitude: 原始约束违规幅度（未加权），供算法层作为 cost 信号
            # = constraint_penalty（与 reward 中被减去的量相同）
            # 当 use_lagrangian_constraint=True 时，此值用于 λ 的 dual ascent 更新
            "violation_magnitude": float(constraint_penalty),
            "soc_excess_normalized": float(soc_excess_normalized),
            "ramp_excess_intent": float(ramp_excess),
            "soc_clip_depth_norm": float(soc_clip_depth_norm),
            "power_clip_magnitude": float(power_clip_magnitude),
            "ramp_clip_magnitude": float(ramp_clip_magnitude),
            "loss": loss,
            "cvar": float(cvar),
            "loss_ema": float(cvar),           # backward-compat alias
            # ── 消融实验专用日志字段 ──────────────────────────────────────────
            "loss_ema_surrogate": float(self.loss_ema),   # 实际 EMA 风险代理值
            "loss_ma_surrogate": float(self.loss_ma),     # MA 风险代理值
            "active_risk_weight": float(                  # 当前有效风险权重
                self.lambda_cvar if self.use_dynamic_ema else self.static_risk_weight
            ),
            "loss_buffer_len": len(self._loss_buffer),
            "tracking_reward": tracking_reward,
            "economic_reward": economic_reward,
            "risk_penalty": risk_penalty,
            "smooth_penalty": smooth_penalty,
            "constraint_penalty": constraint_penalty,
            "total_reward": float(total_reward),
            "reward_term_track": float(rw_track),
            "reward_term_econ": float(rw_econ),
            "reward_term_risk": float(rw_risk),
            "reward_term_smooth": float(rw_smooth),
            "reward_term_const": float(rw_const),
            "reward_term_intent_flat": float(-intent_flat_penalty),
            "intent_flat_penalty": float(intent_flat_penalty),
            "soc_center_shaping": float(soc_center_shaping),
            "reward_term_soc_center": float(rw_soc_center),
        }

        self.episode_tracking_rewards.append(tracking_reward)
        self.episode_economic_rewards.append(economic_reward)
        self.episode_risk_penalties.append(risk_penalty)
        self.episode_smooth_penalties.append(smooth_penalty)
        self.episode_constraint_penalties.append(constraint_penalty)
        self.episode_soc_center_terms.append(rw_soc_center)
        self.episode_total_rewards.append(float(total_reward))
        self.episode_losses.append(loss)
        self.episode_soc_violation_flags.append(int(soc_clipped))
        self.episode_ramp_violation_flags.append(int(ramp_clipped))
        self.episode_power_violation_flags.append(int(power_clipped))
        self.episode_intent_soc_violation_flags.append(int(soc_violation))
        self.episode_intent_ramp_violation_flags.append(int(ramp_violation))

        self._p_desired_buf.append(float(p_desired))
        if self.debug_print_interval > 0 and self.global_env_step % self.debug_print_interval == 0:
            buf = np.asarray(self._p_desired_buf, dtype=np.float64)
            print(
                f"[pv_env debug] global_env_step={self.global_env_step} "
                f"p_desired stats (buf min/max): {float(buf.min()):.5f} {float(buf.max()):.5f} | "
                f"soc={self.soc:.5f} p_applied={p_applied:.5f} "
                f"clip(soc,ramp,pwr)=({int(soc_clipped)},{int(ramp_clipped)},{int(power_clipped)}) "
                f"intent=({int(soc_violation)},{int(ramp_violation)},{int(power_violation)})"
            )

        if self.wandb_log_interval > 0 and self.global_env_step % self.wandb_log_interval == 0:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(
                        {
                            "violation/soc": float(int(soc_clipped)),
                            "violation/ramp": float(int(ramp_clipped)),
                            "violation/power": float(int(power_clipped)),
                            "debug/p_desired": float(p_desired),
                            "debug/p_applied": float(p_applied),
                        },
                        step=int(self.global_env_step),
                    )
            except ImportError:
                pass

        if done:
            info.update(self._episode_metrics())

        obs = self._get_observation() if not done else self._get_observation()
        return obs, float(total_reward), bool(done), info

    def _episode_metrics(self) -> Dict:
        losses = np.asarray(self.episode_losses, dtype=np.float32)
        n = len(self.episode_total_rewards)
        out: Dict = {
            "episode_return": float(np.sum(self.episode_total_rewards)),
            "tracking_reward_mean": float(np.mean(self.episode_tracking_rewards)) if self.episode_tracking_rewards else 0.0,
            "economic_reward_mean": float(np.mean(self.episode_economic_rewards)) if self.episode_economic_rewards else 0.0,
            "risk_penalty_mean": float(np.mean(self.episode_risk_penalties)) if self.episode_risk_penalties else 0.0,
            "smooth_penalty_mean": float(np.mean(self.episode_smooth_penalties)) if self.episode_smooth_penalties else 0.0,
            "constraint_penalty_mean": float(np.mean(self.episode_constraint_penalties)) if self.episode_constraint_penalties else 0.0,
            "soc_center_reward_mean": float(np.mean(self.episode_soc_center_terms)) if self.episode_soc_center_terms else 0.0,
            "violation_rate_soc": float(np.mean(self.episode_soc_violation_flags)) if n else 0.0,
            "violation_rate_ramp": float(np.mean(self.episode_ramp_violation_flags)) if n else 0.0,
            "violation_rate_power": float(np.mean(self.episode_power_violation_flags)) if n else 0.0,
            "violation_rate_soc_intent": float(np.mean(self.episode_intent_soc_violation_flags)) if n else 0.0,
            "violation_rate_ramp_intent": float(np.mean(self.episode_intent_ramp_violation_flags)) if n else 0.0,
            "soc_clip_count": int(self.episode_soc_clips),
            "ramp_clip_count": int(self.episode_ramp_clips),
        }
        if losses.size > 0:
            out["mean_loss"] = float(losses.mean())
            out["max_loss"] = float(losses.max())
            # For evaluation compatibility with existing scripts (NOT used in reward):
            # CVaR_0.1 = mean of worst 10% losses within the episode.
            sorted_losses = np.sort(losses)[::-1]
            n_tail = max(1, int(0.1 * len(sorted_losses)))
            out["cvar_0.1"] = float(np.mean(sorted_losses[:n_tail]))
        else:
            out["mean_loss"] = 0.0
            out["max_loss"] = 0.0
            out["cvar_0.1"] = 0.0
        return out

    def render(self, mode: str = "human"):
        return None


def make_env(
    data_path: str,
    mode: str = "simple",
    lambda_cvar: float = 1.0,
    lambda_smooth: float = 0.1,
    cvar_alpha: float = 0.1,
    loss_buffer_maxlen: int = 400,
    debug_violation: bool = False,
    wandb_log_interval: int = 0,
    debug_print_interval: int = 0,
    const_intent_flat: float = 0.0,
    # 消融实验开关（透传至 PVBESSEnv）
    use_dynamic_ema: bool = True,
    ema_beta: float = 0.9,
    static_risk_weight: float = 10.0,
    use_intent_penalty: bool = True,
    use_moving_average: bool = False,
    ma_window: int = 10,
    disable_projection: bool = False,
    # Lagrangian 约束开关（透传至 PVBESSEnv）
    use_lagrangian_constraint: bool = False,
    **kwargs,
) -> PVBESSEnv:
    """
    Factory helper (kept for backward compatibility).

    消融实验参数说明：
      use_dynamic_ema     : True=EMA/MA动态代理（DR3L_full）; False=静态重罚
      ema_beta            : EMA衰减系数（0.9=默认; 0.0=无记忆; 0.99=长记忆）
      static_risk_weight  : use_dynamic_ema=False 时的固定惩罚倍率
      use_intent_penalty  : True=基于Actor原始输出; False=基于执行后状态
      use_moving_average  : True=用 MA 代替 EMA 作为风险代理
      ma_window           : MA 窗口步数
      disable_projection  : True=测试期移除安全投影（验证约束内化）

    注：DR3L 分阶段实验（exp5b / multiseed / stability）中，`const_intent_flat` 等
    由 `run_experiments.py` 的 `env_kw_phases` 在 episode 边界经 `_apply_env_kw` 切换；
    与 `results/strict/FIXED_exp5b_dr3l_phased_results.md` 中 Plan B 一致（阶段1 固定 1.0，
    阶段2 为 0.3）。
    """
    return PVBESSEnv(
        data_path=data_path,
        mode=mode,
        lambda_cvar=lambda_cvar,
        lambda_smooth=lambda_smooth,
        cvar_alpha=cvar_alpha,
        loss_buffer_maxlen=loss_buffer_maxlen,
        debug_violation=debug_violation,
        wandb_log_interval=wandb_log_interval,
        debug_print_interval=debug_print_interval,
        const_intent_flat=const_intent_flat,
        use_dynamic_ema=use_dynamic_ema,
        ema_beta=ema_beta,
        static_risk_weight=static_risk_weight,
        use_intent_penalty=use_intent_penalty,
        use_moving_average=use_moving_average,
        ma_window=ma_window,
        disable_projection=disable_projection,
        use_lagrangian_constraint=use_lagrangian_constraint,
        **kwargs,
    )


class MarkovianReward:
    """
    独立 reward 计算器（旧版 EMA 风险）；与 PVBESSEnv 的滚动 CVaR buffer 不同。
    论文实验请以 PVBESSEnv 为准。
    """

    def __init__(
        self,
        target_power_mw: float = 0.0,
        price_buy: float = 100.0,
        price_sell: float = 80.0,
        dt_hours: float = 5.0 / 60.0,
        alpha_track_loss: float = 1.0,
        beta_econ_loss: float = 1.0,
        loss_ema_decay: float = 0.95,
        lambda_cvar: float = 1.0,
        w_track: float = 1.0,
        w_econ: float = 1.0,
        w_risk: float = 1.0,
        w_smooth: float = 1.0,
        w_const: float = 1.0,
        tracking_scale: float = 5.0,
        economic_scale: float = 0.3,
    ):
        self.target_power_mw = float(target_power_mw)
        self.price_buy = float(price_buy)
        self.price_sell = float(price_sell)
        self.dt_hours = float(dt_hours)

        self.alpha_track_loss = float(alpha_track_loss)
        self.beta_econ_loss = float(beta_econ_loss)
        self.loss_ema_decay = float(loss_ema_decay)
        self.lambda_cvar = float(lambda_cvar)

        self.w_track = float(w_track)
        self.w_econ = float(w_econ)
        self.w_risk = float(w_risk)
        self.w_smooth = float(w_smooth)
        self.w_const = float(w_const)

        self.tracking_scale = float(tracking_scale)
        self.economic_scale = float(economic_scale)

        self.loss_ema = 0.0

    def reset(self):
        self.loss_ema = 0.0

    def _economic_reward(self, p_grid_mw: float) -> float:
        if p_grid_mw >= 0.0:
            cost = p_grid_mw * self.price_buy
            revenue = 0.0
        else:
            cost = 0.0
            revenue = (-p_grid_mw) * self.price_sell
        econ = (revenue - cost) * self.dt_hours
        return float(self.economic_scale * econ)

    def _update_risk(self, p_grid_mw: float, economic_reward: float) -> float:
        tracking_term = abs(p_grid_mw - self.target_power_mw)
        loss = float(
            self.alpha_track_loss * tracking_term
            + self.beta_econ_loss * max(0.0, -economic_reward)
        )
        d = self.loss_ema_decay
        self.loss_ema = float(d * self.loss_ema + (1.0 - d) * loss)
        return loss

    def _risk_penalty(self) -> float:
        return float(self.lambda_cvar * max(0.0, self.loss_ema))

    def compute(
        self,
        p_grid_mw: float,
        p_battery_mw: float,
        p_battery_prev_mw: float,
        hour: float = 0.0,
        constraint_penalty: float = 0.0,
    ) -> Tuple[float, Dict]:
        tracking_term = abs(p_grid_mw - self.target_power_mw)
        tracking_reward = -self.tracking_scale * tracking_term
        economic_reward = self._economic_reward(p_grid_mw)

        loss = self._update_risk(p_grid_mw, economic_reward)
        risk_penalty = self._risk_penalty()
        smooth_penalty = float(abs(p_battery_mw - p_battery_prev_mw))

        total_reward = (
            self.w_track * tracking_reward
            + self.w_econ * economic_reward
            - self.w_risk * risk_penalty
            - self.w_smooth * smooth_penalty
            - self.w_const * constraint_penalty
        )

        info = {
            "tracking_reward": tracking_reward,
            "economic_reward": economic_reward,
            "risk_penalty": risk_penalty,
            "smooth_penalty": smooth_penalty,
            "constraint_penalty": float(constraint_penalty),
            "total_reward": float(total_reward),
            "loss": loss,
            "loss_ema": self.loss_ema,
        }
        return float(total_reward), info

