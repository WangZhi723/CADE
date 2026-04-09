"""
Risk-aware + constraint-penalty reward (Markov), aligned with pv_env.PVBESSEnv.

Risk uses only loss_ema (no tail/variance history in the penalty):
    loss = alpha * |P_grid - P_target| + beta * max(0, -economic_reward)
    loss_ema = decay * loss_ema + (1 - decay) * loss
    risk_penalty = lambda_cvar * loss_ema

Economic (per step, × dt_hours for interval scaling):
    if P_grid >= 0: cost = P_grid * price_buy; revenue = 0
    else: cost = 0; revenue = -P_grid * price_sell
    economic_reward = (revenue - cost) * dt_hours  (then × economic_scale)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    from config import Config
except Exception:  # pragma: no cover
    Config = None


@dataclass
class RewardParams:
    dt_hours: float = 5.0 / 60.0
    price_buy: float = 100.0
    price_sell: float = 80.0

    target_power_mw: float = 0.0

    alpha_track_loss: float = 1.0
    beta_econ_loss: float = 1.0

    loss_ema_decay: float = 0.95
    lambda_cvar: float = 1.0

    w_track: float = 1.0
    w_econ: float = 1.0
    w_risk: float = 1.0
    w_smooth: float = 1.0
    w_const: float = 1.0

    tracking_scale: float = 10.0
    economic_scale: float = 1.0


class RiskAwareReward:
    """
    Markov risk (loss_ema) + optional constraint penalty; mirrors pv_env reward logic.
    """

    def __init__(self, params: RewardParams | None = None):
        if params is None:
            params = self._from_config_or_defaults()
        self.p = params
        self.reset()

    @staticmethod
    def _from_config_or_defaults() -> RewardParams:
        if Config is None:
            return RewardParams()

        return RewardParams(
            dt_hours=5.0 / 60.0,
            price_buy=getattr(Config, "PRICE_BUY", 100.0),
            price_sell=getattr(Config, "PRICE_SELL", 80.0),
            target_power_mw=getattr(Config, "TARGET_POWER", 0.0),
            alpha_track_loss=getattr(Config, "ALPHA_TRACK_LOSS", 1.0),
            beta_econ_loss=getattr(Config, "BETA_ECON_LOSS", 1.0),
            lambda_cvar=getattr(Config, "LAMBDA_CVAR", 1.0),
            w_const=getattr(Config, "W_CONST", 1.0),
        )

    def reset(self):
        self.t = 0
        self.loss_ema = 0.0

        self.stats = {
            "total_reward": 0.0,
            "tracking_reward": 0.0,
            "economic_reward": 0.0,
            "risk_penalty": 0.0,
            "smooth_penalty": 0.0,
            "constraint_penalty": 0.0,
        }

    def compute_economic_reward(self, p_grid_mw: float, hour: float = 0.0) -> float:
        if p_grid_mw >= 0.0:
            cost = p_grid_mw * self.p.price_buy
            revenue = 0.0
        else:
            cost = 0.0
            revenue = (-p_grid_mw) * self.p.price_sell

        econ = (revenue - cost) * self.p.dt_hours
        return float(self.p.economic_scale * econ)

    def compute_tracking_reward(self, p_grid_mw: float) -> Tuple[float, float]:
        tracking_term = abs(p_grid_mw - self.p.target_power_mw)
        tracking_reward = -self.p.tracking_scale * tracking_term
        return float(tracking_reward), float(tracking_term)

    def _update_risk_state(self, p_grid_mw: float, economic_reward: float) -> float:
        tracking_term = abs(p_grid_mw - self.p.target_power_mw)
        loss = float(
            self.p.alpha_track_loss * tracking_term
            + self.p.beta_econ_loss * max(0.0, -economic_reward)
        )
        d = self.p.loss_ema_decay
        self.loss_ema = float(d * self.loss_ema + (1.0 - d) * loss)
        self.t += 1
        return loss

    def compute_risk_penalty(self) -> float:
        return float(self.p.lambda_cvar * max(0.0, self.loss_ema))

    def compute_total_reward(
        self,
        p_grid_mw: float,
        p_battery_mw: float,
        p_battery_prev_mw: float,
        hour: float,
        constraint_penalty: float = 0.0,
    ) -> Tuple[float, Dict]:
        tracking_reward, _ = self.compute_tracking_reward(p_grid_mw)
        economic_reward = self.compute_economic_reward(p_grid_mw, hour)

        loss = self._update_risk_state(p_grid_mw, economic_reward)
        risk_penalty = self.compute_risk_penalty()

        smooth_penalty = float(abs(p_battery_mw - p_battery_prev_mw))

        rw_track = self.p.w_track * tracking_reward
        rw_econ = self.p.w_econ * economic_reward
        rw_risk = -self.p.w_risk * risk_penalty
        rw_smooth = -self.p.w_smooth * smooth_penalty
        rw_const = -self.p.w_const * float(constraint_penalty)

        total_reward = rw_track + rw_econ + rw_risk + rw_smooth + rw_const

        self.stats["total_reward"] += total_reward
        self.stats["tracking_reward"] += tracking_reward
        self.stats["economic_reward"] += economic_reward
        self.stats["risk_penalty"] += risk_penalty
        self.stats["smooth_penalty"] += smooth_penalty
        self.stats["constraint_penalty"] += float(constraint_penalty)

        info = {
            "tracking_reward": tracking_reward,
            "economic_reward": economic_reward,
            "risk_penalty": risk_penalty,
            "smooth_penalty": smooth_penalty,
            "constraint_penalty": float(constraint_penalty),
            "total_reward": float(total_reward),
            "loss": loss,
            "loss_ema": self.loss_ema,
            "reward_term_track": float(rw_track),
            "reward_term_econ": float(rw_econ),
            "reward_term_risk": float(rw_risk),
            "reward_term_smooth": float(rw_smooth),
            "reward_term_const": float(rw_const),
        }

        return float(total_reward), info

    def get_statistics(self) -> Dict:
        return dict(self.stats)

    def get_risk_metrics(self) -> Dict[str, float]:
        """兼容 modules/environment 观测中的滚动 CVaR 占位：用 loss_ema 代替。"""
        return {"cvar_005": float(self.loss_ema)}
