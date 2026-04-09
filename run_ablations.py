#!/usr/bin/env python3
"""
run_ablations.py — DR3L 消融实验系统（论文级）
===============================================

运行环境：conda activate solar

用法示例
--------
# 顺序跑全部组，3 seeds，WandB 开启，500 eps
conda run -n solar python run_ablations.py \\
    --seeds 0 1 2 --n_episodes 500 --use_wandb

# 断点续训（已有 checkpoint 则从断点继续）
conda run -n solar python run_ablations.py \\
    --seeds 0 1 2 --n_episodes 500 --use_wandb --resume

# 只跑 A 和 E 组，4 进程并行
conda run -n solar python run_ablations.py \\
    --configs A_DR3L_full E_State_Penalty \\
    --seeds 0 1 2 --n_workers 4 --use_wandb --resume

# 跳过 Pareto sweep 加速调试
conda run -n solar python run_ablations.py \\
    --seeds 0 --n_episodes 50 --skip_pareto

实验组
------
  A  DR3L_full        EMA β=0.9, intent penalty
  B  Static_Heavy     static risk weight=10, no EMA
  C  EMA β sweep      β ∈ {0.0, 0.5, 0.9, 0.99}
  D  MA vs EMA        moving-average window ∈ {5, 10}
  E  State_Penalty    penalty on projected (executed) action

输出结构
--------
  results/ablations/
    {config}/seed_{seed}.json      # 完整 episode 日志
    {config}/seed_{seed}.csv       # CSV 方便画图
    pareto/pareto_dynamic_vs_static.json
    ema/ema_analysis.json
    intent/intent_generalization.json
    summary.json

  checkpoints/ablations/
    {config}/seed_{seed}/          # agent.pt + training_meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Optional WandB
# ──────────────────────────────────────────────────────────────────────────────
try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False
    print("⚠️  wandb 未安装。运行 pip install wandb 后可启用 WandB 日志。")

from algorithms import DR3L
from pv_env import PVBESSEnv, make_env
from training_checkpoint import (
    checkpoint_exists,
    load_training_checkpoint,
    save_training_checkpoint,
)

# ──────────────────────────────────────────────────────────────────────────────
# 路径
# ──────────────────────────────────────────────────────────────────────────────
TRAIN_PATH = "processed_data/strict/alice/train_rl.pkl.gz"
TEST_PATH  = "processed_data/strict/alice/test_rl.pkl.gz"

# ──────────────────────────────────────────────────────────────────────────────
# 训练阶段调度（与 experiment_ablation 保持一致）
# ──────────────────────────────────────────────────────────────────────────────
PHASE1_EPS = 300

PHASE1_ENV_KW: Dict[str, Any] = dict(
    w_const=2.0, const_intent_flat=1.0, w_soc_center=1.5, lambda_cvar=0.1,
)
PHASE2_ENV_KW: Dict[str, Any] = dict(
    w_const=3.0, const_intent_flat=0.3, w_soc_center=1.0, lambda_cvar=0.5,
)

BASE_ENV_KW: Dict[str, Any] = dict(cvar_alpha=0.1, loss_buffer_maxlen=400)
BASE_AGENT_KW: Dict[str, Any] = dict(cvar_alpha=0.1, lambda_scale=0.5, rho_scale=2.0)

PARETO_STATIC_WEIGHTS = [1.0, 5.0, 10.0, 50.0, 100.0]
PARETO_LAMBDA_SCALES  = [0.1, 0.3, 0.5, 1.0]

# ──────────────────────────────────────────────────────────────────────────────
# 实验配置表
# ──────────────────────────────────────────────────────────────────────────────
def _make_configs() -> Dict[str, Dict]:
    def _cfg(abl, agent_kw=None, label=""):
        return {"ablation": abl,
                "agent_kw": {**BASE_AGENT_KW, **(agent_kw or {})},
                "label": label}

    cfgs: Dict[str, Dict] = {}

    cfgs["A_DR3L_full"] = _cfg(
        dict(use_dynamic_ema=True, ema_beta=0.9, use_intent_penalty=True,
             use_moving_average=False),
        label="DR3L_full (EMA β=0.9, intent)")

    cfgs["B_Static_Heavy"] = _cfg(
        dict(use_dynamic_ema=False, static_risk_weight=10.0,
             use_intent_penalty=True, use_moving_average=False),
        label="Static Heavy (w=10)")

    for beta in [0.0, 0.5, 0.9, 0.99]:
        tag = str(beta).replace(".", "p")
        cfgs[f"C_EMA_beta{tag}"] = _cfg(
            dict(use_dynamic_ema=True, ema_beta=beta, use_intent_penalty=True,
                 use_moving_average=False),
            label=f"EMA β={beta}")

    for win in [5, 10]:
        cfgs[f"D_MA_w{win}"] = _cfg(
            dict(use_dynamic_ema=True, ema_beta=0.9, use_intent_penalty=True,
                 use_moving_average=True, ma_window=win),
            label=f"MA window={win}")

    cfgs["E_State_Penalty"] = _cfg(
        dict(use_dynamic_ema=True, ema_beta=0.9, use_intent_penalty=False,
             use_moving_average=False),
        label="State Penalty (projected)")

    return cfgs


CONFIGS = _make_configs()

# ──────────────────────────────────────────────────────────────────────────────
# AblationLogger（JSON + CSV，每 seed 独立文件）
# ──────────────────────────────────────────────────────────────────────────────
EPISODE_FIELDS = [
    "episode", "seed",
    "return", "violation_rate", "cvar", "max_loss", "mean_loss",
    "violation_soc", "violation_ramp", "violation_intent_soc",
    "ema_risk", "ma_risk", "instant_loss", "risk_weight",
    "raw_action_mean", "raw_action_std",
    "proj_action_mean", "proj_action_std", "action_delta_mean",
    "rw_track", "rw_econ", "rw_risk", "rw_smooth", "rw_const",
]


class AblationLogger:
    """
    每个 (config, seed) 对应独立 logger。
    支持追加写入（断点续训后继续记录到同一文件末尾）。
    """
    def __init__(self, results_dir: str, config_name: str, seed: int,
                 append: bool = False):
        self.config_name = config_name
        self.seed = seed

        out_dir = Path(results_dir) / config_name
        out_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = out_dir / f"seed_{seed}.json"
        self.csv_path  = out_dir / f"seed_{seed}.csv"

        # 若 append=True 且文件存在，从已有数据接续
        self.episodes: List[Dict] = []
        if append and self.json_path.exists():
            try:
                with open(self.json_path) as f:
                    old = json.load(f)
                self.episodes = old.get("episodes", [])
                print(f"  📂 日志续写：已加载 {len(self.episodes)} 条历史记录")
            except Exception:
                self.episodes = []

        mode = "a" if (append and self.csv_path.exists()) else "w"
        self._csv_file = open(self.csv_path, mode, newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=EPISODE_FIELDS, extrasaction="ignore"
        )
        if mode == "w":
            self._csv_writer.writeheader()

    def log_episode(self, episode: int, data: Dict[str, Any]) -> None:
        record = {"episode": episode, "seed": self.seed, **data}
        self.episodes.append(record)
        self._csv_writer.writerow(record)
        self._csv_file.flush()

    def save_json(self, metadata: Optional[Dict] = None) -> None:
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump({"config": self.config_name, "seed": self.seed,
                       "metadata": metadata or {}, "episodes": self.episodes},
                      f, indent=2, default=float)

    def close(self) -> None:
        self._csv_file.close()

    def summary_stats(self, last_n: int = 50) -> Dict[str, float]:
        eps = self.episodes[-last_n:] if len(self.episodes) >= last_n else self.episodes
        if not eps:
            return {}
        out = {}
        for k in ["return", "violation_rate", "cvar", "max_loss", "ema_risk"]:
            vals = [e[k] for e in eps if k in e]
            if vals:
                out[f"{k}_mean"] = float(np.mean(vals))
                out[f"{k}_std"]  = float(np.std(vals))
        return out


# ──────────────────────────────────────────────────────────────────────────────
# 环境构建辅助
# ──────────────────────────────────────────────────────────────────────────────
def _build_env(data_path: str, ablation_kw: Dict,
               extra_env_kw: Dict, mode: str = "multiscale") -> PVBESSEnv:
    return make_env(data_path, mode=mode,
                    **BASE_ENV_KW, **extra_env_kw, **ablation_kw)


def _apply_phase_kw(env: PVBESSEnv, phase_kw: Dict) -> None:
    settable = {"w_const", "const_intent_flat", "w_soc_center",
                "lambda_cvar", "w_track", "w_econ", "w_risk", "w_smooth"}
    for k, v in phase_kw.items():
        if k in settable and hasattr(env, k):
            if k == "lambda_cvar" and hasattr(env, "set_lambda_cvar"):
                env.set_lambda_cvar(float(v))
            else:
                setattr(env, k, float(v))


# ──────────────────────────────────────────────────────────────────────────────
# 单 episode 执行（返回步级统计）
# ──────────────────────────────────────────────────────────────────────────────
def _run_episode(agent: DR3L, env: PVBESSEnv) -> Dict[str, Any]:
    obs  = env.reset()
    done = False
    raw_a, proj_a = [], []
    soc_v, ramp_v, isoc_v, losses = [], [], [], []
    rt, re, rr, rs, rc = [], [], [], [], []

    while not done:
        st, lt = obs["short_term"], obs["long_term"]
        action, lp = agent.select_action(st, lt)
        obs2, rew, done, info = env.step(action)
        nst, nlt = obs2["short_term"], obs2["long_term"]
        agent.store_transition(st, lt, action, rew, lp, nst, nlt, done)

        if (getattr(agent, "off_policy", False) and len(agent.replay) >= 256
                and agent._global_learn_step
                   % max(1, agent.replay_learn_every) == 0):
            agent.learn_from_replay(batch_size=256)

        obs = obs2
        raw_a.append(float(action.reshape(-1)[0]))
        proj_a.append(float(info["p_battery"]))
        soc_v.append(float(info.get("soc_violation", 0)))
        ramp_v.append(float(info.get("ramp_violation", 0)))
        isoc_v.append(float(info.get("intent_soc_violation", 0)))
        losses.append(float(info.get("loss", 0.0)))
        rt.append(float(info.get("reward_term_track",  0.0)))
        re.append(float(info.get("reward_term_econ",   0.0)))
        rr.append(float(info.get("reward_term_risk",   0.0)))
        rs.append(float(info.get("reward_term_smooth", 0.0)))
        rc.append(float(info.get("reward_term_const",  0.0)))

    ra = np.asarray(raw_a,  dtype=np.float32)
    pa = np.asarray(proj_a, dtype=np.float32)
    return {
        "info":       info,
        "env":        env,
        "return":     float(info.get("episode_return", 0.0)),
        "cvar":       float(info.get("cvar_0.1", 0.0)),
        "max_loss":   float(info.get("max_loss", 0.0)),
        "mean_loss":  float(info.get("mean_loss", float(np.mean(losses)))),
        "viol_soc":   float(np.mean(soc_v)),
        "viol_ramp":  float(np.mean(ramp_v)),
        "viol_isoc":  float(np.mean(isoc_v)),
        "raw_mean":   float(ra.mean()), "raw_std":  float(ra.std()),
        "proj_mean":  float(pa.mean()), "proj_std": float(pa.std()),
        "delta_mean": float(np.abs(ra - pa).mean()),
        "rw_track":   float(np.mean(rt)), "rw_econ":  float(np.mean(re)),
        "rw_risk":    float(np.mean(rr)), "rw_smooth":float(np.mean(rs)),
        "rw_const":   float(np.mean(rc)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 核心 worker（训练 + 测试评估）
# ──────────────────────────────────────────────────────────────────────────────
def _worker_full(args: tuple) -> Dict:
    """
    完整 worker：训练 → 测试评估，在同一进程内完成。
    支持 WandB 日志、断点续训（通过 options['resume'] 和 checkpoint_dir）。
    """
    (config_name, config, seed, n_episodes,
     results_dir, checkpoint_root, device, options) = args

    use_wandb       = options.get("use_wandb", False) and _WANDB_OK
    wandb_project   = options.get("wandb_project", "dr3l-ablations")
    wandb_entity    = options.get("wandb_entity", None)
    do_resume       = options.get("resume", False)
    ckpt_every      = options.get("checkpoint_every", 50)
    eval_eps        = options.get("eval_episodes", 30)
    eval_warmup     = options.get("eval_warmup", 5)

    try:
        # ── 随机状态（每个 worker 独立隔离）────────────────────────────────
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        ablation_kw = config["ablation"]
        agent_kw    = config["agent_kw"]
        run_slug    = f"{config_name}_s{seed}"

        # ── checkpoint 目录 ─────────────────────────────────────────────────
        ckpt_dir = Path(checkpoint_root) / config_name / f"seed_{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # ── Logger（续写模式）──────────────────────────────────────────────
        logger = AblationLogger(results_dir, config_name, seed,
                                append=do_resume)

        # ── 环境 ───────────────────────────────────────────────────────────
        env = _build_env(TRAIN_PATH, ablation_kw, PHASE1_ENV_KW)

        # ── Agent（独立初始化）────────────────────────────────────────────
        agent = DR3L(state_dim=10, device=device, **agent_kw)

        # ── 指标容器 ───────────────────────────────────────────────────────
        metrics = {
            "episode_returns":   [],
            "episode_cvars":     [],
            "episode_max_losses":[],
            "episode_rewards":   [],
        }
        start_episode    = 0
        wandb_resume_id  = None
        in_phase2        = False

        # ── 断点续训 ────────────────────────────────────────────────────────
        if do_resume and checkpoint_exists(ckpt_dir):
            try:
                start_episode, metrics, wandb_resume_id, extra = \
                    load_training_checkpoint(agent, ckpt_dir, device)
                # 恢复 phase 状态
                if start_episode >= PHASE1_EPS:
                    in_phase2 = True
                    _apply_phase_kw(env, PHASE2_ENV_KW)
                    # replay 在 checkpoint 里没有保存，phase2 无需清空
                print(f"  📂 [{run_slug}] 断点恢复：从 episode {start_episode} 继续")
            except Exception as e:
                print(f"  ⚠️  [{run_slug}] 加载 checkpoint 失败，从头训练: {e}")
                start_episode = 0

        if start_episode >= n_episodes:
            print(f"  ⏭️  [{run_slug}] 已完成 {start_episode}/{n_episodes}，跳过训练")
            # 仍需做测试评估
            skip_train = True
        else:
            skip_train = False

        # ── WandB 初始化 ────────────────────────────────────────────────────
        wb_run = None
        if use_wandb:
            try:
                init_kw: Dict[str, Any] = {
                    "project": wandb_project,
                    "name":    run_slug,
                    "group":   config_name,
                    "tags":    [config_name, f"seed_{seed}",
                                "ablation",
                                "dynamic" if ablation_kw.get("use_dynamic_ema") else "static",
                                "intent"  if ablation_kw.get("use_intent_penalty") else "state"],
                    "config": {
                        "config_name":  config_name,
                        "label":        config.get("label", ""),
                        "seed":         seed,
                        "n_episodes":   n_episodes,
                        "device":       device,
                        **ablation_kw,
                        **{f"agent_{k}": v for k, v in agent_kw.items()},
                    },
                }
                if wandb_entity:
                    init_kw["entity"] = wandb_entity
                if wandb_resume_id and do_resume:
                    init_kw["id"]     = wandb_resume_id
                    init_kw["resume"] = "allow"
                wb_run = wandb.init(**init_kw)
                print(f"  📊 [{run_slug}] WandB run: {wb_run.name}  url: {wb_run.url}")
            except Exception as e:
                print(f"  ⚠️  [{run_slug}] WandB 初始化失败: {e}")
                wb_run = None

        # ── 训练循环 ─────────────────────────────────────────────────────────
        if not skip_train:
            t0 = time.time()
            for episode in tqdm(range(start_episode, n_episodes),
                                initial=start_episode, total=n_episodes,
                                desc=f"{config_name[:20]}|s{seed}", leave=False):

                # 阶段切换
                if not in_phase2 and episode >= PHASE1_EPS:
                    in_phase2 = True
                    _apply_phase_kw(env, PHASE2_ENV_KW)
                    agent.replay.clear()
                    agent._per_learn_count = 0
                    print(f"\n  🔄 [{run_slug}] Phase 2 切换 (episode {episode})")

                ep = _run_episode(agent, env)

                viol_rate = (ep["viol_soc"] + ep["viol_ramp"]) / 2.0
                metrics["episode_returns"].append(ep["return"])
                metrics["episode_cvars"].append(ep["cvar"])
                metrics["episode_max_losses"].append(ep["max_loss"])
                metrics["episode_rewards"].append(ep["return"])

                log_data: Dict[str, Any] = {
                    "return":          ep["return"],
                    "violation_rate":  viol_rate,
                    "cvar":            ep["cvar"],
                    "max_loss":        ep["max_loss"],
                    "mean_loss":       ep["mean_loss"],
                    "violation_soc":   ep["viol_soc"],
                    "violation_ramp":  ep["viol_ramp"],
                    "violation_intent_soc": ep["viol_isoc"],
                    "ema_risk":        float(ep["env"].loss_ema),
                    "ma_risk":         float(ep["env"].loss_ma),
                    "instant_loss":    ep["mean_loss"],
                    "risk_weight":     float(
                        ep["env"].lambda_cvar if ep["env"].use_dynamic_ema
                        else ep["env"].static_risk_weight),
                    "raw_action_mean": ep["raw_mean"], "raw_action_std":   ep["raw_std"],
                    "proj_action_mean":ep["proj_mean"],"proj_action_std":  ep["proj_std"],
                    "action_delta_mean":ep["delta_mean"],
                    "rw_track":  ep["rw_track"], "rw_econ":   ep["rw_econ"],
                    "rw_risk":   ep["rw_risk"],  "rw_smooth": ep["rw_smooth"],
                    "rw_const":  ep["rw_const"],
                }
                logger.log_episode(episode, log_data)

                # ── WandB per-episode log ──────────────────────────────────
                if wb_run is not None:
                    wb_log: Dict[str, Any] = {
                        "train/return":          ep["return"],
                        "train/violation_rate":  viol_rate,
                        "train/cvar":            ep["cvar"],
                        "train/max_loss":        ep["max_loss"],
                        "train/ema_risk":        float(ep["env"].loss_ema),
                        "train/ma_risk":         float(ep["env"].loss_ma),
                        "train/instant_loss":    ep["mean_loss"],
                        "train/risk_weight":     log_data["risk_weight"],
                        "train/action_delta":    ep["delta_mean"],
                        "train/rw_track":        ep["rw_track"],
                        "train/rw_risk":         ep["rw_risk"],
                        "train/rw_const":        ep["rw_const"],
                        "train/phase":           2 if in_phase2 else 1,
                        "train/episode":         episode,
                    }
                    # 100-episode 移动均值
                    last100 = metrics["episode_returns"][-100:]
                    wb_log["train/return_100ep_avg"] = float(np.mean(last100))
                    wandb.log(wb_log, step=episode)

                # ── Checkpoint 保存 ────────────────────────────────────────
                if (episode + 1) % ckpt_every == 0 or (episode + 1) == n_episodes:
                    wid = wb_run.id if wb_run else None
                    save_training_checkpoint(
                        agent, ckpt_dir,
                        last_completed_episode=episode,
                        metrics=metrics,
                        wandb_run_id=wid,
                        device=device,
                        extra={"ema_risk_history":
                               [e.get("ema_risk", 0) for e in logger.episodes[-ckpt_every:]]},
                    )
                    tqdm.write(f"  💾 [{run_slug}] checkpoint 保存 ep {episode+1}")

            elapsed = time.time() - t0
            print(f"\n  ✅ [{run_slug}] 训练完成 {n_episodes} eps，耗时 {elapsed/60:.1f} min")
        # end if not skip_train

        train_summary = logger.summary_stats(last_n=50)
        logger.save_json({
            "config_name": config_name,
            "seed": seed, "n_episodes": n_episodes,
            "ablation": ablation_kw,
        })
        logger.close()

        # ── 测试评估 ──────────────────────────────────────────────────────────
        test_base_kw = dict(lambda_cvar=0.5, w_const=3.0, const_intent_flat=0.0)
        test_env   = _build_env(TEST_PATH, ablation_kw, test_base_kw)
        eval_normal = _evaluate(agent, test_env, eval_eps, eval_warmup,
                                disable_projection=False)

        # E 组 & A 组：额外做「去投影」测试，验证约束内化
        eval_no_proj = None
        if config_name in ("A_DR3L_full", "E_State_Penalty"):
            eval_no_proj = _evaluate(agent, test_env, eval_eps, eval_warmup,
                                     disable_projection=True)

        # ── WandB test log ───────────────────────────────────────────────────
        if wb_run is not None:
            wb_test: Dict[str, Any] = {
                "test/return_mean":    eval_normal["episode_returns_mean"],
                "test/cvar_mean":      eval_normal["episode_cvars_mean"],
                "test/violation_mean": eval_normal["violation_any_rate_mean"],
                "test/action_delta":   eval_normal["action_delta_mean"],
            }
            if eval_no_proj:
                wb_test["test/viol_no_projection"] = \
                    eval_no_proj["violation_any_rate_mean"]
            wandb.log(wb_test)
            wandb.finish()

        return {
            "status":           "ok",
            "config":           config_name,
            "seed":             seed,
            "train_summary":    train_summary,
            "test_normal":      eval_normal,
            "test_no_projection": eval_no_proj,
            "json_path":        str(logger.json_path),
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] {config_name} seed={seed}:\n{tb}")
        if use_wandb and _WANDB_OK:
            try:
                wandb.finish(exit_code=1)
            except Exception:
                pass
        return {"status": "error", "config": config_name,
                "seed": seed, "error": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# 测试评估
# ──────────────────────────────────────────────────────────────────────────────
def _evaluate(agent: DR3L, env: PVBESSEnv,
              n_eps: int = 30, warmup: int = 5,
              disable_projection: bool = False) -> Dict[str, float]:
    _orig = env.disable_projection
    env.disable_projection = disable_projection

    for _ in range(warmup):
        obs = env.reset()
        done = False
        while not done:
            act, _ = agent.select_action(obs["short_term"], obs["long_term"])
            obs, _, done, _ = env.step(act)

    rets, cvars, max_l, viol_any, deltas = [], [], [], [], []
    for _ in range(n_eps):
        obs = env.reset()
        done = False
        ep_soc, ep_ramp, ep_d = [], [], []
        while not done:
            act, _ = agent.select_action(obs["short_term"], obs["long_term"])
            obs, _, done, info = env.step(act)
            ep_soc.append(float(info.get("soc_violation", 0)))
            ep_ramp.append(float(info.get("ramp_violation", 0)))
            ep_d.append(abs(float(act.reshape(-1)[0]) * env.power_rated_mw
                            - float(info["p_battery"])))
        rets.append(float(info["episode_return"]))
        cvars.append(float(info.get("cvar_0.1", 0.0)))
        max_l.append(float(info.get("max_loss", 0.0)))
        any_v = float(np.mean([max(s, r) for s, r in zip(ep_soc, ep_ramp)]))
        viol_any.append(any_v)
        deltas.append(float(np.mean(ep_d)))

    env.disable_projection = _orig
    return {
        "episode_returns_mean":    float(np.mean(rets)),
        "episode_returns_std":     float(np.std(rets)),
        "episode_cvars_mean":      float(np.mean(cvars)),
        "episode_cvars_std":       float(np.std(cvars)),
        "episode_max_losses_mean": float(np.mean(max_l)),
        "violation_any_rate_mean": float(np.mean(viol_any)),
        "action_delta_mean":       float(np.mean(deltas)),
        "disable_projection":      disable_projection,
        "n_episodes":              n_eps,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pareto sweep
# ──────────────────────────────────────────────────────────────────────────────
def run_pareto_sweep(seeds: List[int], n_eps: int,
                     results_dir: str, checkpoint_root: str,
                     device: str, options: Dict) -> List[Dict]:
    pareto_dir = Path(results_dir) / "pareto"
    pareto_dir.mkdir(parents=True, exist_ok=True)
    out_path = pareto_dir / "pareto_dynamic_vs_static.json"

    points: List[Dict] = []
    print("\n" + "=" * 68)
    print("PARETO SWEEP  static_risk_weight ∈ {} | lambda_scale ∈ {}".format(
        PARETO_STATIC_WEIGHTS, PARETO_LAMBDA_SCALES))
    print("=" * 68)

    rows: List[tuple] = []
    for w in PARETO_STATIC_WEIGHTS:
        rows.append(("static", w, None,
                     dict(use_dynamic_ema=False, static_risk_weight=w,
                          use_intent_penalty=True, use_moving_average=False)))
    for ls in PARETO_LAMBDA_SCALES:
        rows.append(("dr3l", None, ls,
                     dict(use_dynamic_ema=True, ema_beta=0.9,
                          use_intent_penalty=True, use_moving_average=False)))

    for method, w, ls, abl in rows:
        label = f"Static w={w}" if method == "static" else f"DR3L λ={ls}"
        cfg_name = (f"Pareto_Static_w{w:g}" if method == "static"
                    else f"Pareto_DR3L_ls{ls:g}")
        agent_kw = {**BASE_AGENT_KW, **({"lambda_scale": ls} if ls else {})}
        cfg = {"ablation": abl, "agent_kw": agent_kw, "label": label}

        rets, cvars, viols = [], [], []
        for seed in seeds:
            r = _worker_full((cfg_name, cfg, seed, n_eps,
                              str(pareto_dir), checkpoint_root,
                              device, options))
            if r["status"] == "ok":
                t = r["test_normal"]
                rets.append(t["episode_returns_mean"])
                cvars.append(t["episode_cvars_mean"])
                viols.append(t["violation_any_rate_mean"])

        pt: Dict[str, Any] = {
            "method": method,
            "risk_weight":  float(w)  if w  is not None else None,
            "lambda_scale": float(ls) if ls is not None else None,
            "label": label,
            "return_mean":    float(np.mean(rets))  if rets  else float("nan"),
            "return_std":     float(np.std(rets))   if rets  else float("nan"),
            "cvar_mean":      float(np.mean(cvars)) if cvars else float("nan"),
            "cvar_std":       float(np.std(cvars))  if cvars else float("nan"),
            "violation_mean": float(np.mean(viols)) if viols else float("nan"),
            "violation_std":  float(np.std(viols))  if viols else float("nan"),
            "n_seeds":        len(rets),
        }
        points.append(pt)
        print(f"  {label:22s}  ret={pt['return_mean']:7.1f}±{pt['return_std']:.1f}"
              f"  cvar={pt['cvar_mean']:.4f}  viol={pt['violation_mean']:.3f}")

    with open(out_path, "w") as f:
        json.dump(points, f, indent=2, default=float)
    print(f"\n✅ Pareto 数据: {out_path}")
    return points


# ──────────────────────────────────────────────────────────────────────────────
# 分析文件提取
# ──────────────────────────────────────────────────────────────────────────────
def _extract_ema_analysis(all_results: Dict, results_dir: str) -> None:
    out_dir = Path(results_dir) / "ema"
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis: Dict[str, Any] = {}
    for cfg_name, seeds_data in all_results.items():
        if not (cfg_name.startswith("C_") or cfg_name.startswith("D_")):
            continue
        ema_traj, loss_traj = [], []
        for r in seeds_data.values():
            if r.get("status") != "ok":
                continue
            jp = Path(r.get("json_path", ""))
            if jp.exists():
                with open(jp) as f:
                    d = json.load(f)
                eps = d.get("episodes", [])
                ema_traj.append([e.get("ema_risk", 0.0) for e in eps])
                loss_traj.append([e.get("instant_loss", 0.0) for e in eps])
        if ema_traj:
            analysis[cfg_name] = {"ema_risk": ema_traj, "instant_loss": loss_traj}
    with open(out_dir / "ema_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=float)
    print(f"✅ EMA 分析: {out_dir / 'ema_analysis.json'}")


def _extract_intent_analysis(all_results: Dict, results_dir: str) -> None:
    out_dir = Path(results_dir) / "intent"
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis: Dict[str, Any] = {}
    for cfg_name in ("A_DR3L_full", "E_State_Penalty"):
        if cfg_name not in all_results:
            continue
        nv, npv, deltas = [], [], []
        for r in all_results[cfg_name].values():
            if r.get("status") != "ok":
                continue
            nv.append(r["test_normal"].get("violation_any_rate_mean", float("nan")))
            tp = r.get("test_no_projection")
            if tp:
                npv.append(tp.get("violation_any_rate_mean", float("nan")))
            jp = Path(r.get("json_path", ""))
            if jp.exists():
                with open(jp) as f:
                    d = json.load(f)
                eps = d.get("episodes", [])
                if eps:
                    deltas.append(float(np.mean([e.get("action_delta_mean", 0)
                                                 for e in eps])))
        analysis[cfg_name] = {
            "violation_with_projection":    nv,
            "violation_without_projection": npv,
            "action_delta_train_mean":      deltas,
        }
    with open(out_dir / "intent_generalization.json", "w") as f:
        json.dump(analysis, f, indent=2, default=float)
    print(f"✅ Intent 分析: {out_dir / 'intent_generalization.json'}")


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="DR3L 消融实验（conda activate solar）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # ── 实验参数 ───────────────────────────────────────────────────────────────
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n_episodes", type=int, default=500,
                        help="每 config×seed 的训练 episode 数")
    parser.add_argument("--configs", type=str, nargs="*", default=None,
                        help="只运行指定 config（不填=全部）\n可选: "
                             + " ".join(CONFIGS.keys()))
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda / cpu（默认 cuda）")
    # ── WandB ─────────────────────────────────────────────────────────────────
    parser.add_argument("--use_wandb", action="store_true",
                        help="启用 WandB 日志")
    parser.add_argument("--wandb_project", type=str, default="dr3l-ablations")
    parser.add_argument("--wandb_entity",  type=str, default=None)
    # ── 断点续训 ───────────────────────────────────────────────────────────────
    parser.add_argument("--resume", action="store_true",
                        help="从 checkpoint 断点续训")
    parser.add_argument("--checkpoint_root", type=str,
                        default="checkpoints/ablations",
                        help="checkpoint 保存根目录")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                        help="每多少 episode 保存一次 checkpoint")
    # ── 输出 ───────────────────────────────────────────────────────────────────
    parser.add_argument("--results_dir", type=str, default="results/ablations")
    # ── 并行 ───────────────────────────────────────────────────────────────────
    parser.add_argument("--n_workers", type=int, default=1,
                        help="并行进程数（1=顺序；>1 用 spawn multiprocessing）")
    # ── Pareto ─────────────────────────────────────────────────────────────────
    parser.add_argument("--skip_pareto", action="store_true")
    parser.add_argument("--pareto_episodes", type=int, default=300)
    parser.add_argument("--pareto_seeds",    type=int, nargs="*", default=None)
    # ── 评估 ───────────────────────────────────────────────────────────────────
    parser.add_argument("--eval_episodes", type=int, default=30)
    parser.add_argument("--eval_warmup",   type=int, default=5)
    args = parser.parse_args()

    # ── 设备检查 ──────────────────────────────────────────────────────────────
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，回退到 CPU")
        device = "cpu"
    else:
        device = args.device

    # ── 筛选 configs ─────────────────────────────────────────────────────────
    run_cfgs = {k: v for k, v in CONFIGS.items()
                if args.configs is None or k in args.configs}
    if not run_cfgs:
        print(f"⚠️  无匹配 config。可选: {list(CONFIGS.keys())}")
        return

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_root).mkdir(parents=True, exist_ok=True)

    options = {
        "use_wandb":        args.use_wandb and _WANDB_OK,
        "wandb_project":    args.wandb_project,
        "wandb_entity":     args.wandb_entity,
        "resume":           args.resume,
        "checkpoint_every": args.checkpoint_every,
        "eval_episodes":    args.eval_episodes,
        "eval_warmup":      args.eval_warmup,
    }

    # ── 打印计划 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"DR3L 消融实验  |  device={device}  "
          f"seeds={args.seeds}  n_eps={args.n_episodes}")
    print(f"  WandB={options['use_wandb']}  "
          f"resume={args.resume}  "
          f"ckpt_every={args.checkpoint_every}")
    print(f"  results_dir   : {args.results_dir}")
    print(f"  checkpoint_root: {args.checkpoint_root}")
    print("=" * 68)
    for k, v in run_cfgs.items():
        status = ""
        for s in args.seeds:
            cd = Path(args.checkpoint_root) / k / f"seed_{s}"
            if checkpoint_exists(cd):
                import json as _j
                try:
                    meta = _j.load(open(cd / "training_meta.json"))
                    ep = meta.get("last_completed_episode", -1)
                    status += f" [s{s}:ep{ep+1}]"
                except Exception:
                    status += f" [s{s}:ckpt?]"
        print(f"  {k:30s}  {v['label']}{status}")

    tasks = [
        (cfg_name, cfg, seed, args.n_episodes,
         args.results_dir, args.checkpoint_root, device, options)
        for cfg_name, cfg in run_cfgs.items()
        for seed in args.seeds
    ]
    print(f"\n共 {len(tasks)} 个任务 "
          f"({len(run_cfgs)} configs × {len(args.seeds)} seeds)\n")

    # ── 执行 ────────────────────────────────────────────────────────────────
    all_results: Dict[str, Dict] = defaultdict(dict)

    if args.n_workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        print(f"🚀 并行模式：{args.n_workers} 进程（注：WandB 多进程下各 run 独立）")
        with ctx.Pool(processes=args.n_workers) as pool:
            task_results = pool.map(_worker_full, tasks)
    else:
        print("🔄 顺序执行")
        task_results = []
        for t in tasks:
            task_results.append(_worker_full(t))

    for r in task_results:
        all_results[r.get("config", "unknown")][f"seed_{r.get('seed', -1)}"] = r

    # ── 控制台 summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("ABLATION SUMMARY（test set）")
    print("=" * 68)
    print(f"{'Config':35s}  {'Return':>12}  {'CVaR':>10}  {'Viol%':>8}")
    print("-" * 68)

    summary_table: Dict[str, Dict] = {}
    for cfg_name, seeds_data in sorted(all_results.items()):
        rets, cvars, viols = [], [], []
        for r in seeds_data.values():
            if r.get("status") == "ok":
                t = r.get("test_normal", {})
                rets.append(t.get("episode_returns_mean", float("nan")))
                cvars.append(t.get("episode_cvars_mean",  float("nan")))
                viols.append(t.get("violation_any_rate_mean", float("nan")))
        if rets:
            rm, rs = float(np.nanmean(rets)), float(np.nanstd(rets))
            cm, cs = float(np.nanmean(cvars)), float(np.nanstd(cvars))
            vm, vs = float(np.nanmean(viols)), float(np.nanstd(viols))
            lbl = run_cfgs.get(cfg_name, {}).get("label", cfg_name)
            print(f"  {lbl:33s}  {rm:7.1f}±{rs:.1f}  "
                  f"{cm:.4f}±{cs:.4f}  {100*vm:.1f}%±{100*vs:.1f}%")
            summary_table[cfg_name] = {
                "label": lbl, "return_mean": rm, "return_std": rs,
                "cvar_mean": cm, "cvar_std": cs,
                "viol_mean": vm, "viol_std": vs, "n_seeds": len(rets),
            }
        else:
            print(f"  {cfg_name:33s}  (无有效结果)")

    # ── 保存总 summary ───────────────────────────────────────────────────────
    sp = Path(args.results_dir) / "summary.json"
    with open(sp, "w") as f:
        json.dump(
            {"summary": summary_table,
             "raw_results": {
                 cfg: {sk: {kk: vv for kk, vv in rv.items() if kk != "traceback"}
                       for sk, rv in sd.items()}
                 for cfg, sd in all_results.items()
             }},
            f, indent=2, default=float,
        )
    print(f"\n✅ 总 summary: {sp}")

    _extract_ema_analysis(dict(all_results), args.results_dir)
    _extract_intent_analysis(dict(all_results), args.results_dir)

    # ── Pareto sweep ─────────────────────────────────────────────────────────
    if not args.skip_pareto:
        p_seeds = args.pareto_seeds or args.seeds
        p_opts  = {**options, "use_wandb": False}   # Pareto 不上传 WandB
        run_pareto_sweep(
            seeds=p_seeds, n_eps=args.pareto_episodes,
            results_dir=args.results_dir,
            checkpoint_root=args.checkpoint_root,
            device=device, options=p_opts,
        )

    print("\n🎉 全部实验完成！")
    print(f"   results/ablations/              ← JSON/CSV episode 日志")
    print(f"   results/ablations/summary.json  ← 汇总表格")
    print(f"   results/ablations/pareto/       ← Pareto 曲线数据")
    print(f"   results/ablations/ema/          ← EMA 时间结构分析")
    print(f"   results/ablations/intent/       ← 约束内化验证")
    print(f"   checkpoints/ablations/          ← 断点 checkpoint")


if __name__ == "__main__":
    main()
