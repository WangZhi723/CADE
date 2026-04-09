"""
IEEE TSG Experiments Runner
Implements all experiments for DR3L paper
"""

import numpy as np
import torch
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from training_checkpoint import (
    checkpoint_exists,
    load_training_checkpoint,
    save_training_checkpoint,
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
import shutil
import time

# Optional wandb support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Install with: pip install wandb")
    print("   Training will continue without wandb logging.")

from pv_env import make_env
from algorithms import DDPG, PPOAgent, DR3L, DR3LQuantile


def _env_lambda_at_episode(episode: int, phases: List[Tuple[int, float]]) -> float:
    """按分段表返回该 episode 应使用的 env.lambda_cvar（episode 为 0-based）。"""
    if not phases:
        return 0.0
    acc = 0
    for length, lam in phases:
        if episode < acc + int(length):
            return float(lam)
        acc += int(length)
    return float(phases[-1][1])


def _warm_restart_agent_lr(agent, lr_actor: float = 3e-4, lr_critic: float = 3e-4) -> None:
    """λ_cvar 升档时将优化器学习率重置为初值，减轻奖励分布突变导致的崩溃。"""
    if isinstance(agent, DDPG):
        for pg in agent.actor_optimizer.param_groups:
            pg["lr"] = lr_actor
        for pg in agent.critic_optimizer.param_groups:
            pg["lr"] = lr_critic
    elif isinstance(agent, PPOAgent):
        for pg in agent.optimizer.param_groups:
            pg["lr"] = lr_actor
    elif isinstance(agent, DR3L):
        for pg in agent.optimizer_actor.param_groups:
            pg["lr"] = lr_actor
        for pg in agent.optimizer_feature_critic.param_groups:
            pg["lr"] = lr_critic


class ExperimentRunner:
    """Run all experiments for IEEE TSG paper (With WandB Support)"""
    
    def __init__(
        self,
        results_dir: str = "results",
        seed: int = 42,
        use_wandb: bool = True,
        wandb_project: str = "dr3l-pv-bess",
        wandb_entity: Optional[str] = None,
        checkpoint_root: Optional[Union[str, Path]] = None,
        resume: bool = False,
        checkpoint_every: int = 50,
        force_retrain: bool = False,
        debug_violation_env: bool = False,
        env_wandb_log_interval: int = 0,
        env_debug_print_interval: int = 0,
        baseline_w_const: float = 5.0,
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.checkpoint_root = Path(checkpoint_root) if checkpoint_root else None
        self.resume = resume
        self.checkpoint_every = max(1, int(checkpoint_every))
        self.force_retrain = force_retrain
        self.debug_violation_env = debug_violation_env
        self.env_wandb_log_interval = int(env_wandb_log_interval)
        self.env_debug_print_interval = int(env_debug_print_interval)
        self.baseline_w_const = float(baseline_w_const)

        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize wandb if enabled (will be initialized per experiment)
        if self.use_wandb:
            print(f"✅ WandB enabled (project: {wandb_project})")
            print("   WandB runs will be created for each training session")
        elif use_wandb and not WANDB_AVAILABLE:
            print("⚠️  WandB requested but not available. Install with: pip install wandb")
        if self.checkpoint_root:
            print(f"✅ 检查点目录: {self.checkpoint_root}（每 {self.checkpoint_every} episode 保存；--resume 可续训）")
            print("   注意：DDPG 回放缓存不保存，续训后 replay 为空。")
        if self.debug_violation_env:
            print("⚠️  PV 环境 debug_violation=ON：p_desired×2，仅用于调试。论文/正式实验请勿使用。")

    def _exp5_wconst_tag(self) -> str:
        """Experiment 5 输出文件名 / checkpoint 后缀（如 3 -> '3', 3.5 -> '3p5'）。"""
        return f"{self.baseline_w_const:g}".replace(".", "p")

    def _make_env(self, data_path: str, mode: str = "simple", **kwargs):
        return make_env(
            data_path,
            mode=mode,
            debug_violation=self.debug_violation_env,
            wandb_log_interval=self.env_wandb_log_interval,
            debug_print_interval=self.env_debug_print_interval,
            **kwargs,
        )
    
    def train_agent(
        self,
        agent,
        env,
        n_episodes: int = 1000,
        agent_name: str = "agent",
        wandb_run_name: Optional[str] = None,
        checkpoint_dir: Optional[Path] = None,
        resume: Optional[bool] = None,
        env_lambda_phases: Optional[List[Tuple[int, float]]] = None,
        env_kw_phases: Optional[List[Tuple[int, Dict[str, float]]]] = None,
    ) -> Dict:
        """Train an agent with WandB logging、可选检查点与断点续训。

        env_lambda_phases: 例如 [(200, 0.0), (100, 0.1), (100, 0.5), (100, 1.0)]，
        每段 (episode 数, lambda_cvar)。若提供，n_episodes 将与各段长度之和强制对齐。
        
        env_kw_phases: 例如 [(0, {...}), (200, {...})]，在每个 episode 开始前原地更新 env 的可设超参。
        用于 DR3L/约束分段等实验；不会重建 env，从而保留 loss_buffer 等跨 episode 状态。
        """
        print(f"\nTraining {agent_name}...")
        run_wandb = self.use_wandb
        run_slug = (wandb_run_name or agent_name).replace("/", "_").replace(" ", "_")
        ckpt_dir = checkpoint_dir
        if ckpt_dir is None and self.checkpoint_root is not None:
            ckpt_dir = self.checkpoint_root / run_slug
        do_resume = self.resume if resume is None else resume

        if env_lambda_phases:
            phases_total = sum(int(p[0]) for p in env_lambda_phases)
            if n_episodes != phases_total:
                print(
                    f"⚠️  env_lambda_phases 总长 {phases_total} 与 n_episodes={n_episodes} 不一致，"
                    f"将 n_episodes 设为 {phases_total}。"
                )
                n_episodes = phases_total

        env_kw_phases_sorted: Optional[List[Tuple[int, Dict[str, float]]]] = None
        if env_kw_phases:
            env_kw_phases_sorted = sorted(
                [(int(ep), kw) for ep, kw in env_kw_phases],
                key=lambda x: x[0],
            )

        start_episode = 0
        metrics = {
            'episode_returns': [],
            'episode_rewards': [],
            'episode_cvars': [],
            'episode_max_losses': []
        }
        q_values_list: List[float] = []
        recent_critic_losses: List[float] = []
        recent_actor_losses: List[float] = []
        critic_grad_norms: List[float] = []  # 新增：收集critic梯度norm
        actor_grad_norms: List[float] = []   # 新增：收集actor梯度norm
        wandb_resume_id: Optional[str] = None
        prev_lam_for_lr: Optional[float] = None

        _SETTABLE = {
            "w_const",
            "const_intent_flat",
            "w_soc_center",
            "lambda_cvar",
            "w_track",
            "w_econ",
            "w_risk",
            "w_smooth",
        }

        def _apply_env_kw(env: object, kw: Dict[str, float]) -> None:
            """原地更新 env 超参（不重建 env；保留 loss_buffer 等内部状态）。"""
            nonlocal prev_lam_for_lr
            for k, v in kw.items():
                if k not in _SETTABLE:
                    continue
                if not hasattr(env, k):
                    continue

                v = float(v)
                if k == "lambda_cvar" and hasattr(env, "set_lambda_cvar"):
                    # λ_cvar 变化时做一次学习率 warm restart（稳定性）
                    if prev_lam_for_lr is not None and abs(v - prev_lam_for_lr) > 1e-12:
                        _warm_restart_agent_lr(agent)
                    env.set_lambda_cvar(v)
                    prev_lam_for_lr = v
                else:
                    setattr(env, k, v)

        if ckpt_dir is not None and do_resume and checkpoint_exists(ckpt_dir):
            try:
                start_episode, metrics, wandb_resume_id, extra = load_training_checkpoint(
                    agent, Path(ckpt_dir), self.device
                )
                q_values_list = [float(x) for x in extra.get("q_values_list", [])]
                recent_critic_losses = [float(x) for x in extra.get("recent_critic_losses", [])]
                recent_actor_losses = [float(x) for x in extra.get("recent_actor_losses", [])]
                print(f"📂 已从检查点恢复: {ckpt_dir}，从 episode {start_episode + 1}/{n_episodes} 继续")
            except Exception as e:
                print(f"⚠️ 加载检查点失败，将从头训练: {e}")
                start_episode = 0
                wandb_resume_id = None

        if start_episode >= n_episodes:
            print(f"⏭️ 检查点显示训练已完成（{start_episode} >= {n_episodes}），跳过训练。")
            return metrics

        # Initialize wandb run for this agent if enabled
        if run_wandb:
            run_name = wandb_run_name or f"{agent_name}_{self.seed}"
            try:
                if wandb.run is not None:
                    wandb.finish()

                init_kwargs = {
                    'project': self.wandb_project,
                    'name': run_name,
                    'config': {
                        'agent': agent_name,
                        'n_episodes': n_episodes,
                        'seed': self.seed,
                        'device': self.device,
                        'resume_from_episode': start_episode,
                        'env_lambda_phases': env_lambda_phases,
                    }
                }
                if self.wandb_entity:
                    init_kwargs['entity'] = self.wandb_entity
                if wandb_resume_id:
                    init_kwargs['id'] = wandb_resume_id
                    init_kwargs['resume'] = 'allow'

                wandb.init(**init_kwargs)
                print(f"📊 WandB run: {wandb.run.name} ({wandb.run.url if wandb.run else 'N/A'})")
            except Exception as e:
                print(f"⚠️  WandB initialization failed: {e}")
                print("   Continuing without WandB logging...")
                run_wandb = False

        def _save_ckpt(last_ep: int) -> None:
            if ckpt_dir is None:
                return
            wid = wandb.run.id if (run_wandb and wandb.run is not None) else wandb_resume_id
            extra = {
                "q_values_list": q_values_list[-2000:],
                "recent_critic_losses": recent_critic_losses[-2000:],
                "recent_actor_losses": recent_actor_losses[-2000:],
            }
            save_training_checkpoint(
                agent,
                Path(ckpt_dir),
                last_completed_episode=last_ep,
                metrics=metrics,
                wandb_run_id=wid,
                device=self.device,
                extra=extra,
            )
            print(f"💾 检查点已保存: {ckpt_dir}（已完成 episode {last_ep + 1}）")

        for episode in tqdm(
            range(start_episode, n_episodes),
            initial=start_episode,
            total=n_episodes,
            desc=agent_name[:24],
        ):
            lam_ep: Optional[float] = None
            if env_lambda_phases is not None and hasattr(env, "set_lambda_cvar"):
                lam_ep = _env_lambda_at_episode(episode, env_lambda_phases)
                if prev_lam_for_lr is not None and lam_ep != prev_lam_for_lr:
                    _warm_restart_agent_lr(agent)
                    print(
                        f"🔁 env λ_cvar 分段切换 {prev_lam_for_lr} → {lam_ep}："
                        f"已重置 {agent_name} 优化器学习率至 3e-4"
                    )
                env.set_lambda_cvar(lam_ep)
                prev_lam_for_lr = float(lam_ep)

            # 原地切换 DR3L 的约束/意图惩罚超参（不重建 env，保留 loss_buffer 等状态）
            if env_kw_phases_sorted is not None:
                active_kw: Optional[Dict[str, float]] = None
                prev_phase_ep: Optional[int] = None
                for phase_ep, kw in env_kw_phases_sorted:
                    if episode >= phase_ep:
                        active_kw = kw
                        prev_phase_ep = phase_ep
                    else:
                        break
                if active_kw is not None:
                    # 检测阶段切换：清空 replay buffer 并重置 PER β
                    if (prev_phase_ep is not None and episode == prev_phase_ep 
                        and hasattr(agent, 'replay') and hasattr(agent.replay, 'clear')):
                        agent.replay.clear()
                        agent._per_learn_count = 0  # 重置 PER β 退火计数器
                        print(
                            f"🔄 阶段切换 (episode {episode})：已清空 replay buffer "
                            f"并重置 PER β 退火，避免旧经验污染新惩罚信号"
                        )
                    _apply_env_kw(env, active_kw)

            obs = env.reset()
            if isinstance(agent, DDPG):
                agent.reset_ou_noise()
            done = False
            episode_reward = 0  # 每个episode开始时重置
            step_count = 0
            ep_soc_viol: List[float] = []
            ep_ramp_viol: List[float] = []
            ep_r_total: List[float] = []
            ep_r_track: List[float] = []
            ep_r_econ: List[float] = []
            ep_r_risk: List[float] = []
            ep_r_smooth: List[float] = []
            ep_r_soc_center: List[float] = []
            ep_costs: List[float] = []  # Lagrangian: 每步约束违规幅度
            
            while not done:
                # Select action
                if isinstance(agent, DDPG):
                    # OU 噪声 + 分段衰减（正式实验增强探索）
                    noise_level = 0.42 if episode < 350 else (0.22 if episode < 450 else 0.12)
                    action = agent.select_action(obs, noise=noise_level)
                    obs_next, reward, done, info = env.step(action)
                    agent.buffer.push(obs, action, reward, obs_next, done)
                    
                    if len(agent.buffer) > 256:
                        update_metrics = agent.update(batch_size=256)
                        # 收集Q值统计和loss（用于episode结束时的统计）
                        if update_metrics:
                            if 'q_mean' in update_metrics:
                                q_values_list.append(update_metrics['q_mean'])
                            elif 'q_value' in update_metrics:
                                q_values_list.append(update_metrics['q_value'])
                            
                            # 收集loss
                            if 'critic_loss' in update_metrics:
                                recent_critic_losses.append(update_metrics['critic_loss'])
                            if 'actor_loss' in update_metrics:
                                recent_actor_losses.append(update_metrics['actor_loss'])
                
                elif isinstance(agent, PPOAgent):
                    action, log_prob = agent.select_action(obs)
                    obs_next, reward, done, info = env.step(action)
                    agent.store_transition(obs, action, reward, obs_next, done, log_prob)
                
                elif isinstance(agent, DR3L):
                    # Multi-scale observation
                    short_term = obs['short_term']
                    long_term = obs['long_term']
                    action, log_prob = agent.select_action(short_term, long_term)
                    obs_next, reward, done, info = env.step(action)
                    
                    # Extract next state observations
                    next_short_term = obs_next['short_term']
                    next_long_term = obs_next['long_term']

                    # Lagrangian: 提取约束违规幅度作为 cost 信号
                    # use_lagrangian_constraint=False 时 violation_magnitude=0，不影响原训练
                    step_cost = float(info.get("violation_magnitude", 0.0))
                    ep_costs.append(step_cost)

                    # Store transition with next state (含 cost，向后兼容旧 checkpoint)
                    agent.store_transition(
                        short_term, long_term, action, reward, log_prob,
                        next_short_term, next_long_term, done, step_cost
                    )
                    if getattr(agent, "off_policy", False) and len(agent.replay) >= 256:
                        every = max(1, int(getattr(agent, "replay_learn_every", 1)))
                        if agent._global_learn_step % every == 0:
                            learn_metrics = agent.learn_from_replay(batch_size=256)
                            # 收集梯度norm
                            if 'critic_grad_norm' in learn_metrics:
                                critic_grad_norms.append(float(learn_metrics['critic_grad_norm']))
                            if 'actor_grad_norm' in learn_metrics:
                                actor_grad_norms.append(float(learn_metrics['actor_grad_norm']))
                
                obs = obs_next
                episode_reward += reward  # 累加reward
                step_count += 1
                ep_soc_viol.append(float(info.get("soc_violation", 0)))
                ep_ramp_viol.append(float(info.get("ramp_violation", 0)))
                ep_r_total.append(float(info.get("total_reward", reward)))
                ep_r_track.append(float(info.get("reward_term_track", 0.0)))
                ep_r_econ.append(float(info.get("reward_term_econ", 0.0)))
                ep_r_risk.append(float(info.get("reward_term_risk", 0.0)))
                ep_r_smooth.append(float(info.get("reward_term_smooth", 0.0)))
                ep_r_soc_center.append(float(info.get("reward_term_soc_center", 0.0)))
            
            # Update policy at end of episode
            update_metrics = {}
            if isinstance(agent, PPOAgent):
                update_metrics = agent.update(n_epochs=10)
            elif isinstance(agent, DR3L):
                if getattr(agent, "off_policy", False):
                    update_metrics = dict(getattr(agent, "_last_learn_metrics", {}))
                else:
                    update_metrics = agent.update(n_epochs=10)

                # ── Lagrangian 对偶上升更新（每 episode 末执行一次）────────────
                # avg_cost = episode 内平均约束违规幅度
                # λ ← max(0, λ + lr * (avg_cost - ε))
                # use_lagrangian_constraint=False 时 update_lambda 返回 {} 无副作用
                if ep_costs:
                    avg_cost_ep = float(np.mean(ep_costs))
                    lambda_metrics = agent.update_lambda(avg_cost_ep)
                    update_metrics.update(lambda_metrics)
            
            # Record metrics
            episode_return = info.get('episode_return', episode_reward)
            episode_cvar = info.get('cvar_0.1', 0)
            episode_max_loss = info.get('max_loss', 0)
            
            metrics['episode_returns'].append(episode_return)
            metrics['episode_rewards'].append(episode_reward)  # 记录原始reward
            metrics['episode_cvars'].append(episode_cvar)
            metrics['episode_max_losses'].append(episode_max_loss)
            
            # 计算移动平均reward
            moving_avg_reward = np.mean(metrics['episode_rewards'][-100:]) if len(metrics['episode_rewards']) >= 100 else np.mean(metrics['episode_rewards'])
            
            if ckpt_dir is not None:
                if (episode + 1) % self.checkpoint_every == 0 or (episode + 1) == n_episodes:
                    _save_ckpt(episode)

            # Log to wandb (每个episode结束时记录)
            if run_wandb and wandb.run is not None:
                log_dict = {
                    'train/episode_reward': episode_reward,  # 原始reward（最重要）
                    'train/episode_return': episode_return,  # 环境返回的return
                    'train/moving_avg_reward': moving_avg_reward,  # 移动平均reward
                    'train/episode_cvar': episode_cvar,
                    'train/episode_max_loss': episode_max_loss,
                    'train/episode': episode
                }
                if lam_ep is not None:
                    log_dict['train/env_lambda_cvar'] = float(lam_ep)
                if ep_r_total:
                    log_dict['violation/soc'] = float(np.mean(ep_soc_viol))
                    log_dict['violation/ramp'] = float(np.mean(ep_ramp_viol))
                    log_dict['reward/total'] = float(np.mean(ep_r_total))
                    log_dict['reward/tracking'] = float(np.mean(ep_r_track))
                    log_dict['reward/economic'] = float(np.mean(ep_r_econ))
                    log_dict['reward/risk'] = float(np.mean(ep_r_risk))
                    log_dict['reward/smooth'] = float(np.mean(ep_r_smooth))
                    log_dict['reward/soc_center'] = float(np.mean(ep_r_soc_center))
                
                # 添加算法特定指标
                if isinstance(agent, DDPG):
                    # DDPG的Q值统计（使用最近100个batch的统计）
                    if len(q_values_list) > 0:
                        recent_q = q_values_list[-100:] if len(q_values_list) >= 100 else q_values_list
                        log_dict['train/Q_mean'] = np.mean(recent_q)
                        log_dict['train/Q_std'] = np.std(recent_q)
                    
                    # Loss统计（使用最近100个batch的平均）
                    if len(recent_critic_losses) > 0:
                        recent_critic = recent_critic_losses[-100:] if len(recent_critic_losses) >= 100 else recent_critic_losses
                        log_dict['train/critic_loss'] = np.mean(recent_critic)
                    if len(recent_actor_losses) > 0:
                        recent_actor = recent_actor_losses[-100:] if len(recent_actor_losses) >= 100 else recent_actor_losses
                        log_dict['train/actor_loss'] = np.mean(recent_actor)
                    
                    noise_level = 0.42 if episode < 350 else (0.22 if episode < 450 else 0.12)
                    log_dict['train/exploration_noise'] = noise_level
                
                elif isinstance(agent, (PPOAgent, DR3L)) and update_metrics:
                    # PPO/DR3L的指标
                    for key, value in update_metrics.items():
                        if isinstance(value, (int, float)):
                            # Lagrangian 指标已含 "lagrangian/" 前缀，直接上报
                            if key.startswith("lagrangian/"):
                                log_dict[key] = value
                            else:
                                log_dict[f'train/{key}'] = value

                # ── Lagrangian 专项日志（DR3L + use_lagrangian_constraint=True）──
                if isinstance(agent, DR3L) and getattr(agent, 'use_lagrangian_constraint', False):
                    avg_cost_log = float(np.mean(ep_costs)) if ep_costs else 0.0
                    viol_rate_log = float(np.mean([1.0 if c > 0 else 0.0 for c in ep_costs])) if ep_costs else 0.0
                    ema_risk_log = float(getattr(env, 'loss_ema', 0.0))
                    log_dict['lagrangian/lambda_constraint'] = float(agent.lambda_constraint)
                    log_dict['lagrangian/avg_cost'] = avg_cost_log
                    log_dict['lagrangian/violation_rate'] = viol_rate_log
                    log_dict['lagrangian/ema_risk'] = ema_risk_log
                    log_dict['lagrangian/target_violation'] = float(agent.target_violation)

                wandb.log(log_dict, step=episode)
            
            # Log progress
            if (episode + 1) % 100 == 0:
                recent_return = np.mean(metrics['episode_returns'][-100:])
                recent_reward = np.mean(metrics['episode_rewards'][-100:])
                recent_cvar = np.mean(metrics['episode_cvars'][-100:])
                print(f"Episode {episode+1}: Reward={recent_reward:.2f}, Return={recent_return:.2f}, CVaR={recent_cvar:.4f}")
                
                # Log rolling averages to wandb
                if run_wandb and wandb.run is not None:
                    wandb.log({
                        'train/reward_100ep_avg': recent_reward,
                        'train/return_100ep_avg': recent_return,
                        'train/cvar_100ep_avg': recent_cvar
                    }, step=episode)

        # Final summary to wandb (don't finish yet, evaluation will log to same run)
        if run_wandb and wandb.run is not None:
            final_log = {
                'train/final_reward_mean': np.mean(metrics['episode_rewards']),
                'train/final_reward_std': np.std(metrics['episode_rewards']),
                'train/final_return_mean': np.mean(metrics['episode_returns']),
                'train/final_return_std': np.std(metrics['episode_returns']),
                'train/final_cvar_mean': np.mean(metrics['episode_cvars']),
                'train/final_cvar_std': np.std(metrics['episode_cvars'])
            }
            
            # 添加Q值最终统计（DDPG）
            if len(q_values_list) > 0:
                final_log['train/final_Q_mean'] = np.mean(q_values_list)
                final_log['train/final_Q_std'] = np.std(q_values_list)
            
            wandb.log(final_log)
            # Note: Don't finish here, let evaluation log to the same run
        
        return metrics
    
    def evaluate_agent(self, agent, env, n_episodes: int = 50, 
                      split: str = "test", loss_buffer_warmup_episodes: int = 5) -> Dict:
        """Evaluate trained agent；统计 test 违规率供论文 Figure 与 WandB。

        loss_buffer_warmup_episodes: 正式统计前用若干 episode 滚动 loss_buffer，
        减轻训练期跨 episode buffer 与测试独立 env 的分布差异（不计入指标）。
        """
        print(f"\nEvaluating on {split} set...")
        if loss_buffer_warmup_episodes > 0:
            print(f"  Warm-up: {loss_buffer_warmup_episodes} episodes（填充 loss_buffer，不计入统计）...")
        for _ in range(loss_buffer_warmup_episodes):
            obs = env.reset()
            if isinstance(agent, DDPG):
                agent.reset_ou_noise()
            done = False
            while not done:
                if isinstance(agent, DDPG):
                    action = agent.select_action(obs, noise=0.0)
                elif isinstance(agent, PPOAgent):
                    action, _ = agent.select_action(obs)
                elif isinstance(agent, DR3L):
                    action, _ = agent.select_action(obs["short_term"], obs["long_term"])
                else:
                    raise TypeError(f"Unsupported agent type: {type(agent)}")
                obs, _, done, _ = env.step(action)

        metrics = {
            'episode_returns': [],
            'episode_cvars': [],
            'episode_max_losses': [],
            'episode_mean_losses': [],
            'episode_viol_soc': [],
            'episode_viol_ramp': [],
            'episode_viol_any': [],
            'episode_viol_soc_intent': [],
        }
        
        for episode in range(n_episodes):
            obs = env.reset()
            if isinstance(agent, DDPG):
                agent.reset_ou_noise()
            done = False
            soc_flags: List[float] = []
            ramp_flags: List[float] = []
            any_flags: List[float] = []
            intent_soc_flags: List[float] = []
            
            while not done:
                if isinstance(agent, DDPG):
                    action = agent.select_action(obs, noise=0.0)
                elif isinstance(agent, PPOAgent):
                    action, _ = agent.select_action(obs)
                elif isinstance(agent, DR3L):
                    short_term = obs['short_term']
                    long_term = obs['long_term']
                    action, _ = agent.select_action(short_term, long_term)
                
                obs, reward, done, info = env.step(action)
                sv = float(info.get("soc_violation", 0))
                rv = float(info.get("ramp_violation", 0))
                pv = float(info.get("power_violation", 0))
                soc_flags.append(sv)
                ramp_flags.append(rv)
                any_flags.append(1.0 if (sv > 0 or rv > 0 or pv > 0) else 0.0)
                intent_soc_flags.append(float(info.get("intent_soc_violation", 0)))
            
            metrics['episode_returns'].append(info['episode_return'])
            metrics['episode_cvars'].append(info['cvar_0.1'])
            metrics['episode_max_losses'].append(info['max_loss'])
            metrics['episode_mean_losses'].append(info['mean_loss'])
            metrics['episode_viol_soc'].append(float(np.mean(soc_flags)) if soc_flags else 0.0)
            metrics['episode_viol_ramp'].append(float(np.mean(ramp_flags)) if ramp_flags else 0.0)
            metrics['episode_viol_any'].append(float(np.mean(any_flags)) if any_flags else 0.0)
            metrics['episode_viol_soc_intent'].append(
                float(np.mean(intent_soc_flags)) if intent_soc_flags else 0.0
            )
        
        results: Dict = {}
        for key, values in metrics.items():
            results[f'{key}_mean'] = float(np.mean(values))
            results[f'{key}_std'] = float(np.std(values))

        results['violation_soc_mean'] = results['episode_viol_soc_mean']
        results['violation_ramp_mean'] = results['episode_viol_ramp_mean']
        results['violation_any_rate_mean'] = results['episode_viol_any_mean']
        results['violation_soc_intent_mean'] = results['episode_viol_soc_intent_mean']
        
        if self.use_wandb and wandb.run is not None:
            wandb.log({
                f'{split}/return_mean': results['episode_returns_mean'],
                f'{split}/return_std': results['episode_returns_std'],
                f'{split}/cvar_mean': results['episode_cvars_mean'],
                f'{split}/cvar_std': results['episode_cvars_std'],
                f'{split}/max_loss_mean': results['episode_max_losses_mean'],
                f'{split}/mean_loss_mean': results['episode_mean_losses_mean'],
                'test/return_mean': results['episode_returns_mean'],
                'test/cvar_mean': results['episode_cvars_mean'],
                'test/violation_soc': results['violation_soc_mean'],
                'test/violation_ramp': results['violation_ramp_mean'],
                'test/violation_any_rate': results['violation_any_rate_mean'],
                'test/violation_soc_intent': results['violation_soc_intent_mean'],
                'violation/soc': results['violation_soc_mean'],
                'violation/ramp': results['violation_ramp_mean'],
            })
            wandb.finish()
        
        return results
    
    def experiment_1_lambda_tradeoff(self, data_mode: str = "strict"):
        """Experiment 1: λ trade-off analysis (With Resume Support)"""
        print("\n" + "="*80)
        print("EXPERIMENT 1: λ Trade-off Analysis (Resumable Mode)")
        print("="*80)
        
        # 定义结果文件路径
        output_dir = self.results_dir / data_mode
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "exp1_lambda_tradeoff.json"
        
        # 加载已有结果
        results = {}
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    results = json.load(f)
                print(f"✅ Found existing results: {list(results.keys())}")
            except Exception as e:
                print(f"⚠️  Error loading existing results: {e}. Starting fresh.")
        
        # 保存进度函数
        def save_progress():
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            serializable_results[key][sub_key] = {}
                            for metric_key, metric_value in sub_value.items():
                                if isinstance(metric_value, (np.ndarray, list)):
                                    serializable_results[key][sub_key][metric_key] = (
                                        [float(x) for x in metric_value] if isinstance(metric_value, np.ndarray)
                                        else [float(x) for x in metric_value]
                                    )
                                elif isinstance(metric_value, (np.integer, np.floating)):
                                    serializable_results[key][sub_key][metric_key] = float(metric_value)
                                else:
                                    serializable_results[key][sub_key][metric_key] = metric_value
                        else:
                            serializable_results[key][sub_key] = sub_value
                else:
                    serializable_results[key] = value
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"✅ Progress saved to {output_file}")
        
        lambdas = [0.0, 0.1, 0.5, 1.0, 2.0]
        data_path = f"processed_data/{data_mode}/alice/train_rl.pkl.gz"
        test_path = f"processed_data/{data_mode}/alice/test_rl.pkl.gz"
        
        for lam in lambdas:
            key = f"lambda_{lam}"
            if key in results:
                print(f"\n⏭️  Skipping λ={lam} (Already trained)")
                continue
            
            print(f"\nTraining DR3L with λ={lam}")
            try:
                env = self._make_env(data_path, mode='multiscale')
                agent = DR3L(
                    state_dim=10,
                    lambda_scale=lam,
                    cvar_alpha=0.05,
                    device=self.device
                )
                
                train_metrics = self.train_agent(
                    agent, env, n_episodes=500, 
                    agent_name=f"DR3L_lambda_{lam}",
                    wandb_run_name=f"DR3L_lambda_{lam}_{data_mode}"
                )
                test_env = self._make_env(test_path, mode='multiscale')
                eval_metrics = self.evaluate_agent(agent, test_env, n_episodes=50, split='test')
                
                results[key] = {'train': train_metrics, 'test': eval_metrics}
                
                # Save agent
                save_dir = output_dir / f"lambda_{lam}"
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(agent.feature_net.state_dict(), save_dir / "feature_net.pt")
                torch.save(agent.actor.state_dict(), save_dir / "actor.pt")
                torch.save(agent.critic.state_dict(), save_dir / "critic.pt")
                
                save_progress()  # 每个λ跑完立刻存！
            except Exception as e:
                print(f"❌ Error training λ={lam}: {e}")
                import traceback
                traceback.print_exc()
                raise e
        
        # Plot results
        if len(results) > 0:
            try:
                self.plot_lambda_tradeoff(results, data_mode)
            except Exception as e:
                print(f"⚠️  Error plotting: {e}")
        
        return results
    
    def experiment_2_robust_comparison(self, data_mode: str = "strict"):
        """Experiment 2: Robust vs Non-robust (With Resume Support)"""
        print("\n" + "="*80)
        print("EXPERIMENT 2: Robust vs Non-robust Comparison (Resumable Mode)")
        print("="*80)
        
        output_dir = self.results_dir / data_mode
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "exp2_robust_comparison.json"
        
        # 加载已有结果
        results = {}
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    results = json.load(f)
                print(f"✅ Found existing results: {list(results.keys())}")
            except Exception as e:
                print(f"⚠️  Error loading existing results: {e}. Starting fresh.")
        
        def save_progress():
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            serializable_results[key][sub_key] = {}
                            for metric_key, metric_value in sub_value.items():
                                if isinstance(metric_value, (np.ndarray, list)):
                                    serializable_results[key][sub_key][metric_key] = (
                                        [float(x) for x in metric_value] if isinstance(metric_value, np.ndarray)
                                        else [float(x) for x in metric_value]
                                    )
                                elif isinstance(metric_value, (np.integer, np.floating)):
                                    serializable_results[key][sub_key][metric_key] = float(metric_value)
                                else:
                                    serializable_results[key][sub_key][metric_key] = metric_value
                        else:
                            serializable_results[key][sub_key] = sub_value
                else:
                    serializable_results[key] = value
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"✅ Progress saved to {output_file}")
        
        rhos = [0.0, 0.01, 0.05]
        data_path = f"processed_data/{data_mode}/alice/train_rl.pkl.gz"
        test_path = f"processed_data/{data_mode}/alice/test_rl.pkl.gz"
        
        for rho in rhos:
            key = f"rho_{rho}"
            if key in results:
                print(f"\n⏭️  Skipping ρ={rho} (Already trained)")
                continue
            
            print(f"\nTraining DR3L with ρ={rho}")
            try:
                env = self._make_env(data_path, mode='multiscale')
                agent = DR3L(
                    state_dim=10,
                    lambda_scale=0.5,
                    cvar_alpha=rho,
                    device=self.device
                )
                
                train_metrics = self.train_agent(
                    agent, env, n_episodes=500,
                    agent_name=f"DR3L_rho_{rho}",
                    wandb_run_name=f"DR3L_rho_{rho}_{data_mode}"
                )
                test_env = self._make_env(test_path, mode='multiscale')
                eval_metrics = self.evaluate_agent(agent, test_env, n_episodes=50, split='test')
                
                results[key] = {'train': train_metrics, 'test': eval_metrics}
                save_progress()  # 每个ρ跑完立刻存！
            except Exception as e:
                print(f"❌ Error training ρ={rho}: {e}")
                import traceback
                traceback.print_exc()
                raise e
        
        if len(results) > 0:
            try:
                self.plot_robust_comparison(results, data_mode)
            except Exception as e:
                print(f"⚠️  Error plotting: {e}")
        
        return results
    
    def experiment_3_distribution_shift(self):
        """Experiment 3: Distribution shift test"""
        print("\n" + "="*80)
        print("EXPERIMENT 3: Distribution Shift Test")
        print("="*80)
        
        # Train on Alice STRICT
        print("\nTraining on Alice Springs...")
        train_path = "processed_data/strict/alice/train_rl.pkl.gz"
        train_env = self._make_env(train_path, mode='multiscale')
        
        agent = DR3L(
            state_dim=10,
            lambda_scale=0.5,  # lambda_risk equivalent
            cvar_alpha=0.05,  # rho equivalent (CVaR confidence level)
            device=self.device
        )
        
        train_metrics = self.train_agent(
            agent, train_env, n_episodes=500,
            agent_name="DR3L_alice",
            wandb_run_name="DR3L_distribution_shift_alice"
        )
        
        # Test on Alice
        print("\nTesting on Alice Springs...")
        alice_test_path = "processed_data/strict/alice/test_rl.pkl.gz"
        alice_test_env = self._make_env(alice_test_path, mode='multiscale')
        alice_results = self.evaluate_agent(agent, alice_test_env, n_episodes=50, split='test_alice')
        
        # Test on Yulara
        print("\nTesting on Yulara...")
        try:
            yulara_test_path = "processed_data/strict/yulara/test_rl.pkl.gz"
            yulara_test_env = self._make_env(yulara_test_path, mode='multiscale')
            yulara_results = self.evaluate_agent(agent, yulara_test_env, n_episodes=50, split='test_yulara')
        except:
            print("Warning: Yulara data not available")
            yulara_results = {}
        
        results = {
            'train': train_metrics,
            'alice_test': alice_results,
            'yulara_test': yulara_results
        }
        
        with open(self.results_dir / "exp3_distribution_shift.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def experiment_4_data_quality(self):
        """Experiment 4: Data quality ablation (With Resume Support)"""
        print("\n" + "="*80)
        print("EXPERIMENT 4: Data Quality Ablation (Resumable Mode)")
        print("="*80)
        
        output_file = self.results_dir / "exp4_data_quality.json"
        
        # 加载已有结果
        results = {}
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    results = json.load(f)
                print(f"✅ Found existing results: {list(results.keys())}")
            except Exception as e:
                print(f"⚠️  Error loading existing results: {e}. Starting fresh.")
        
        def save_progress():
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            serializable_results[key][sub_key] = {}
                            for metric_key, metric_value in sub_value.items():
                                if isinstance(metric_value, (np.ndarray, list)):
                                    serializable_results[key][sub_key][metric_key] = (
                                        [float(x) for x in metric_value] if isinstance(metric_value, np.ndarray)
                                        else [float(x) for x in metric_value]
                                    )
                                elif isinstance(metric_value, (np.integer, np.floating)):
                                    serializable_results[key][sub_key][metric_key] = float(metric_value)
                                else:
                                    serializable_results[key][sub_key][metric_key] = metric_value
                        else:
                            serializable_results[key][sub_key] = sub_value
                else:
                    serializable_results[key] = value
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"✅ Progress saved to {output_file}")
        
        modes = ['strict', 'light', 'raw']
        
        for mode in modes:
            if mode in results and 'error' not in results[mode] and 'test' in results[mode]:
                print(f"\n⏭️  Skipping {mode.upper()} (Already trained)")
                continue
            
            print(f"\nTraining on {mode.upper()} dataset...")
            train_path = f"processed_data/{mode}/alice/train_rl.pkl.gz"
            test_path = f"processed_data/{mode}/alice/test_rl.pkl.gz"
            
            # Check if files exist before attempting to load
            from pathlib import Path
            train_file = Path(train_path)
            test_file = Path(test_path)
            
            if not train_file.exists():
                error_msg = f"Training file not found: {train_path}"
                print(f"❌ {error_msg}")
                results[mode] = {'error': error_msg}
                save_progress()
                continue
            
            if not test_file.exists():
                error_msg = f"Test file not found: {test_path}"
                print(f"❌ {error_msg}")
                print(f"💡 Hint: Run preprocessing for {mode.upper()} mode:")
                print(f"   python preprocess_dkasc_data_v2.py {mode}")
                results[mode] = {'error': error_msg}
                save_progress()
                continue
            
            try:
                train_env = self._make_env(train_path, mode='multiscale')
                test_env = self._make_env(test_path, mode='multiscale')
                
                agent = DR3L(
                    state_dim=10,
                    lambda_scale=0.5,
                    cvar_alpha=0.05,
                    device=self.device
                )
                
                train_metrics = self.train_agent(
                    agent, train_env, n_episodes=500,
                    agent_name=f"DR3L_{mode}",
                    wandb_run_name=f"DR3L_data_quality_{mode}"
                )
                eval_metrics = self.evaluate_agent(agent, test_env, n_episodes=50, split='test')
                
                results[mode] = {'train': train_metrics, 'test': eval_metrics}
                save_progress()  # 每个mode跑完立刻存！
            except Exception as e:
                print(f"❌ Error with {mode} dataset: {e}")
                results[mode] = {'error': str(e)}
                save_progress()  # 即使出错也保存，避免重复尝试
                import traceback
                traceback.print_exc()
        
        if len(results) > 0:
            try:
                self.plot_data_quality(results)
            except Exception as e:
                print(f"⚠️  Error plotting: {e}")
        
        return results
    
    def experiment_5_baselines(self, data_mode: str = "strict"):
        """Experiment 5: Baseline comparisons (With Resume Support)"""
        print("\n" + "="*80)
        print("EXPERIMENT 5: Baseline Comparisons (Resumable Mode)")
        print(f"   env 覆盖: w_const={self.baseline_w_const}（仅 Experiment 5；pv_env 默认未改）")
        print("="*80)
        
        output_dir = self.results_dir / data_mode
        output_dir.mkdir(parents=True, exist_ok=True)
        w_tag = self._exp5_wconst_tag()
        output_file = output_dir / f"exp5_baselines_wconst_{w_tag}.json"
        
        # 1. 加载已有结果（跳过已跑完的算法）；--force_retrain 时清空并备份当前 w_const 对应 JSON
        results: Dict = {}
        if self.force_retrain:
            if output_file.exists():
                bak = output_dir / f"{output_file.stem}.backup.{int(time.time())}.json"
                shutil.copy2(output_file, bak)
                print(f"⚠️  --force_retrain: 已备份 {output_file.name} -> {bak.name}，将重新训练全部 baseline。")
        elif output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    raw = json.load(f)
                file_w = raw.get("_baseline_w_const")
                if file_w is not None and abs(float(file_w) - self.baseline_w_const) > 1e-6:
                    print(
                        f"⚠️  文件内 _baseline_w_const={file_w} 与当前 {self.baseline_w_const} 不一致，忽略缓存。"
                    )
                    results = {}
                else:
                    results = {
                        k: v
                        for k, v in raw.items()
                        if not str(k).startswith("_") and isinstance(v, dict)
                    }
                print(f"✅ Found existing results: {list(results.keys())}")
            except Exception as e:
                print(f"⚠️  Error loading existing results: {e}. Starting fresh.")
        
        # 辅助函数：保存进度（将numpy数组转换为列表以便JSON序列化）
        def save_progress():
            # 转换numpy数组为列表以便JSON序列化
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            serializable_results[key][sub_key] = {}
                            for metric_key, metric_value in sub_value.items():
                                if isinstance(metric_value, (np.ndarray, list)):
                                    serializable_results[key][sub_key][metric_key] = (
                                        [float(x) for x in metric_value] if isinstance(metric_value, np.ndarray)
                                        else [float(x) for x in metric_value]
                                    )
                                elif isinstance(metric_value, (np.integer, np.floating)):
                                    serializable_results[key][sub_key][metric_key] = float(metric_value)
                                else:
                                    serializable_results[key][sub_key][metric_key] = metric_value
                        else:
                            serializable_results[key][sub_key] = sub_value
                else:
                    serializable_results[key] = value
            serializable_results["_baseline_w_const"] = float(self.baseline_w_const)

            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"✅ Progress saved to {output_file}")

        train_path = f"processed_data/{data_mode}/alice/train_rl.pkl.gz"
        test_path = f"processed_data/{data_mode}/alice/test_rl.pkl.gz"

        baseline_env_kw = dict(
            lambda_cvar=1.0,
            cvar_alpha=0.1,
            loss_buffer_maxlen=400,
            w_const=self.baseline_w_const,
        )
        _wb = f"_{data_mode}_wc{w_tag}"
        
        # --- DDPG ---
        if 'DDPG' in results:
            print("\n⏭️  Skipping DDPG (Already trained)")
        else:
            print("\nTraining DDPG...")
            try:
                env_ddpg = self._make_env(train_path, mode='simple', **baseline_env_kw)
                # Derive state_dim from environment to keep compatibility with state augmentation
                state_dim = int(np.prod(env_ddpg.observation_space.shape))
                agent_ddpg = DDPG(state_dim=state_dim, device=self.device)
                train_metrics_ddpg = self.train_agent(
                    agent_ddpg, env_ddpg, n_episodes=500, 
                    agent_name="DDPG",
                    wandb_run_name=f"DDPG{_wb}",
                )
                test_env_ddpg = self._make_env(test_path, mode='simple', **baseline_env_kw)
                eval_metrics_ddpg = self.evaluate_agent(agent_ddpg, test_env_ddpg, n_episodes=50, split='test')
                
                results['DDPG'] = {'train': train_metrics_ddpg, 'test': eval_metrics_ddpg}
                save_progress()  # 跑完立刻存！
            except Exception as e:
                print(f"❌ Error training DDPG: {e}")
                import traceback
                traceback.print_exc()
                raise e  # 抛出异常停止，修复后下次跑会自动跳过之前的
        
        # --- PPO ---
        if 'PPO' in results:
            print("\n⏭️  Skipping PPO (Already trained)")
        else:
            print("\nTraining PPO...")
            try:
                env_ppo = self._make_env(train_path, mode='simple', **baseline_env_kw)
                state_dim = int(np.prod(env_ppo.observation_space.shape))
                agent_ppo = PPOAgent(state_dim=state_dim, device=self.device)
                train_metrics_ppo = self.train_agent(
                    agent_ppo, env_ppo, n_episodes=500, 
                    agent_name="PPO",
                    wandb_run_name=f"PPO{_wb}",
                )
                test_env_ppo = self._make_env(test_path, mode='simple', **baseline_env_kw)
                eval_metrics_ppo = self.evaluate_agent(agent_ppo, test_env_ppo, n_episodes=50, split='test')
                
                results['PPO'] = {'train': train_metrics_ppo, 'test': eval_metrics_ppo}
                save_progress()  # 跑完立刻存！
            except Exception as e:
                print(f"❌ Error training PPO: {e}")
                import traceback
                traceback.print_exc()
                raise e  # 抛出异常停止，修复后下次跑会自动跳过之前的
        
        # --- DR3L (ρ=0, no robustness) ---
        if 'DR3L_rho0' in results:
            print("\n⏭️  Skipping DR3L_rho0 (Already trained)")
        else:
            print("\nTraining DR3L (ρ=0)...")
            try:
                # 训练与测试同一套 env 风险超参（lambda_cvar=1.0），取消分阶段 λ
                env_dr3l_0 = self._make_env(train_path, mode='multiscale', **baseline_env_kw)
                agent_dr3l_0 = DR3L(
                    state_dim=10,
                    cvar_alpha=0.0,
                    lambda_scale=0.5,
                    rho_scale=0.0,
                    device=self.device
                )
                train_metrics_dr3l_0 = self.train_agent(
                    agent_dr3l_0,
                    env_dr3l_0,
                    n_episodes=500,
                    agent_name="DR3L_rho0",
                    wandb_run_name=f"DR3L_rho0{_wb}",
                )
                test_env_dr3l_0 = self._make_env(test_path, mode='multiscale', **baseline_env_kw)
                eval_metrics_dr3l_0 = self.evaluate_agent(agent_dr3l_0, test_env_dr3l_0, n_episodes=50, split='test')
                
                results['DR3L_rho0'] = {'train': train_metrics_dr3l_0, 'test': eval_metrics_dr3l_0}
                save_progress()  # 跑完立刻存！
            except Exception as e:
                print(f"❌ Error training DR3L_rho0: {e}")
                import traceback
                traceback.print_exc()
                raise e
        
        # --- DR3L (full) ---
        if 'DR3L_full' in results:
            print("\n⏭️  Skipping DR3L_full (Already trained)")
        else:
            print("\nTraining DR3L (full)...")
            try:
                env_dr3l = self._make_env(train_path, mode='multiscale', **baseline_env_kw)
                agent_dr3l = DR3L(
                    state_dim=10,
                    cvar_alpha=0.1,
                    lambda_scale=0.5,
                    rho_scale=2.0,
                    device=self.device
                )
                train_metrics_dr3l = self.train_agent(
                    agent_dr3l,
                    env_dr3l,
                    n_episodes=500,
                    agent_name="DR3L_full",
                    wandb_run_name=f"DR3L_full{_wb}",
                )
                test_env_dr3l = self._make_env(test_path, mode='multiscale', **baseline_env_kw)
                eval_metrics_dr3l = self.evaluate_agent(agent_dr3l, test_env_dr3l, n_episodes=50, split='test')
                
                results['DR3L_full'] = {'train': train_metrics_dr3l, 'test': eval_metrics_dr3l}
                save_progress()  # 跑完立刻存！
            except Exception as e:
                print(f"❌ Error training DR3L_full: {e}")
                import traceback
                traceback.print_exc()
                raise e
        
        # 绘图部分也加上检查，防止因为前面的失败而画不出图
        if len(results) >= 2:  # 至少2个结果才绘图
            try:
                self.plot_baselines(results, data_mode)
            except Exception as e:
                print(f"⚠️  Error plotting: {e}")
        
        return results

    def experiment_5b_dr3l_phased(self, data_mode: str = "strict"):
        """Experiment 5b：DR3L 分阶段约束意图惩罚（方案B）"""
        print("\n" + "=" * 80)
        print("EXPERIMENT 5b: DR3L Phased Constraint Intent (Plan B)")
        print(
            f"   env base 覆盖: w_const={self.baseline_w_const}（DDPG/PPO & 测试）；"
            f" DR3L 阶段1 300ep（FIXED Plan B: w_const=2, intent=1.0, w_soc_center=1.5, λ_cvar=0.1），"
            f"阶段2 w_const=3, intent=0.3, w_soc_center=1.0, λ_cvar=0.5"
        )
        print("=" * 80)

        output_dir = self.results_dir / data_mode
        output_dir.mkdir(parents=True, exist_ok=True)
        w_tag = self._exp5_wconst_tag()
        output_file = output_dir / f"exp5b_dr3l_phased_wconst_{w_tag}.json"

        # 1. 加载已有结果（跳过已跑完的算法）
        results: Dict = {}
        if self.force_retrain:
            if output_file.exists():
                bak = output_dir / f"{output_file.stem}.backup.{int(time.time())}.json"
                shutil.copy2(output_file, bak)
                print(f"⚠️  --force_retrain: 已备份 {output_file.name} -> {bak.name}，将重新训练全部 baseline。")
        elif output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    raw = json.load(f)
                file_w = raw.get("_baseline_w_const")
                if file_w is not None and abs(float(file_w) - self.baseline_w_const) > 1e-6:
                    print(
                        f"⚠️  文件内 _baseline_w_const={file_w} 与当前 {self.baseline_w_const} 不一致，忽略缓存。"
                    )
                    results = {}
                else:
                    results = {k: v for k, v in raw.items() if not str(k).startswith("_") and isinstance(v, dict)}
                print(f"✅ Found existing results: {list(results.keys())}")
            except Exception as e:
                print(f"⚠️  Error loading existing results: {e}. Starting fresh.")

        train_path = f"processed_data/{data_mode}/alice/train_rl.pkl.gz"
        test_path = f"processed_data/{data_mode}/alice/test_rl.pkl.gz"

        baseline_env_kw = dict(
            lambda_cvar=1.0,
            cvar_alpha=0.1,
            loss_buffer_maxlen=400,
            w_const=self.baseline_w_const,
        )
        # FIXED Plan B（与 results/strict/FIXED_exp5b_dr3l_phased_results.md 一致）
        dr3l_phase1_env_kw = {
            **baseline_env_kw,
            "w_const": 2.0,
            "const_intent_flat": 1.0,
            "w_soc_center": 1.5,
            "lambda_cvar": 0.1,
        }
        dr3l_phase2_env_kw = {
            **baseline_env_kw,
            "w_const": 3.0,
            "const_intent_flat": 0.3,
            "w_soc_center": 1.0,
            "lambda_cvar": 0.5,
        }
        PHASE1_EPISODES = 300
        env_kw_phases = [
            (0, dr3l_phase1_env_kw),
            (PHASE1_EPISODES, dr3l_phase2_env_kw),
        ]
        _wb = f"_{data_mode}_wc{w_tag}_phaseB"

        print("=== FIXED CONFIG CHECK (exp5b) ===")
        print(f"PHASE1_EPISODES: {PHASE1_EPISODES}")
        print(f"phase1 const_intent_flat: {dr3l_phase1_env_kw['const_intent_flat']}")
        print(f"phase1 lambda_cvar: {dr3l_phase1_env_kw['lambda_cvar']}")
        print(f"phase1 w_soc_center: {dr3l_phase1_env_kw['w_soc_center']}")
        print(f"phase2 const_intent_flat: {dr3l_phase2_env_kw['const_intent_flat']}")
        print(f"phase2 lambda_cvar: {dr3l_phase2_env_kw['lambda_cvar']}")
        print("====================================")

        def save_progress():
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            serializable_results[key][sub_key] = {}
                            for metric_key, metric_value in sub_value.items():
                                if isinstance(metric_value, (np.ndarray, list)):
                                    serializable_results[key][sub_key][metric_key] = (
                                        [float(x) for x in metric_value]
                                        if isinstance(metric_value, np.ndarray)
                                        else [float(x) for x in metric_value]
                                    )
                                elif isinstance(metric_value, (np.integer, np.floating)):
                                    serializable_results[key][sub_key][metric_key] = float(metric_value)
                                else:
                                    serializable_results[key][sub_key][metric_key] = metric_value
                        else:
                            serializable_results[key][sub_key] = sub_value
                else:
                    serializable_results[key] = value
            serializable_results["_baseline_w_const"] = float(self.baseline_w_const)
            serializable_results["_planB"] = True
            serializable_results["_planB_phase1_ep"] = PHASE1_EPISODES
            serializable_results["_planB_phase1_env"] = {
                "w_const": 2.0,
                "const_intent_flat": 1.0,
                "w_soc_center": 1.5,
                "lambda_cvar": 0.1,
            }
            serializable_results["_planB_phase2_env"] = {
                "w_const": 3.0,
                "const_intent_flat": 0.3,
                "w_soc_center": 1.0,
                "lambda_cvar": 0.5,
            }
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"✅ Progress saved to {output_file}")

        # --- DDPG ---
        if 'DDPG' not in results:
            print("\nTraining DDPG...")
            env_ddpg = self._make_env(train_path, mode='simple', **baseline_env_kw)
            state_dim = int(np.prod(env_ddpg.observation_space.shape))
            agent_ddpg = DDPG(state_dim=state_dim, device=self.device)
            train_metrics_ddpg = self.train_agent(
                agent_ddpg,
                env_ddpg,
                n_episodes=500,
                agent_name="DDPG",
                wandb_run_name=f"DDPG{_wb}",
            )
            test_env_ddpg = self._make_env(test_path, mode='simple', **baseline_env_kw)
            eval_metrics_ddpg = self.evaluate_agent(agent_ddpg, test_env_ddpg, n_episodes=50, split='test')
            results['DDPG'] = {'train': train_metrics_ddpg, 'test': eval_metrics_ddpg}
            save_progress()
        else:
            print("\n⏭️  Skipping DDPG (Already trained)")

        # --- PPO ---
        if 'PPO' not in results:
            print("\nTraining PPO...")
            env_ppo = self._make_env(train_path, mode='simple', **baseline_env_kw)
            state_dim = int(np.prod(env_ppo.observation_space.shape))
            agent_ppo = PPOAgent(state_dim=state_dim, device=self.device)
            train_metrics_ppo = self.train_agent(
                agent_ppo,
                env_ppo,
                n_episodes=500,
                agent_name="PPO",
                wandb_run_name=f"PPO{_wb}",
            )
            test_env_ppo = self._make_env(test_path, mode='simple', **baseline_env_kw)
            eval_metrics_ppo = self.evaluate_agent(agent_ppo, test_env_ppo, n_episodes=50, split='test')
            results['PPO'] = {'train': train_metrics_ppo, 'test': eval_metrics_ppo}
            save_progress()
        else:
            print("\n⏭️  Skipping PPO (Already trained)")

        # --- DR3L (rho=0) ---
        if 'DR3L_rho0' not in results:
            print("\nTraining DR3L (rho=0) with phased env (Plan B)...")
            env_dr3l_0 = self._make_env(train_path, mode='multiscale', **dr3l_phase1_env_kw)
            agent_dr3l_0 = DR3L(
                state_dim=10,
                cvar_alpha=0.0,
                lambda_scale=0.5,
                rho_scale=0.0,
                device=self.device,
            )
            train_metrics_dr3l_0 = self.train_agent(
                agent_dr3l_0,
                env_dr3l_0,
                n_episodes=500,
                agent_name="DR3L_rho0",
                wandb_run_name=f"DR3L_rho0{_wb}",
                env_kw_phases=env_kw_phases,
            )
            test_env_dr3l_0 = self._make_env(test_path, mode='multiscale', **baseline_env_kw)
            eval_metrics_dr3l_0 = self.evaluate_agent(agent_dr3l_0, test_env_dr3l_0, n_episodes=50, split='test')
            results['DR3L_rho0'] = {'train': train_metrics_dr3l_0, 'test': eval_metrics_dr3l_0}
            save_progress()
        else:
            print("\n⏭️  Skipping DR3L_rho0 (Already trained)")

        # --- DR3L (full) ---
        if 'DR3L_full' not in results:
            print("\nTraining DR3L (full) with phased env (Plan B)...")
            env_dr3l = self._make_env(train_path, mode='multiscale', **dr3l_phase1_env_kw)
            agent_dr3l = DR3L(
                state_dim=10,
                cvar_alpha=0.1,
                lambda_scale=0.5,
                rho_scale=2.0,
                device=self.device,
            )
            train_metrics_dr3l = self.train_agent(
                agent_dr3l,
                env_dr3l,
                n_episodes=500,
                agent_name="DR3L_full",
                wandb_run_name=f"DR3L_full{_wb}",
                env_kw_phases=env_kw_phases,
            )
            test_env_dr3l = self._make_env(test_path, mode='multiscale', **baseline_env_kw)
            eval_metrics_dr3l = self.evaluate_agent(agent_dr3l, test_env_dr3l, n_episodes=50, split='test')
            results['DR3L_full'] = {'train': train_metrics_dr3l, 'test': eval_metrics_dr3l}
            save_progress()
        else:
            print("\n⏭️  Skipping DR3L_full (Already trained)")

        # 绘图：避免覆盖 baseline_comparison
        if len(results) >= 2:
            try:
                self.plot_baselines(results, data_mode, plot_suffix="_phaseB")
            except Exception as e:
                print(f"⚠️  Error plotting: {e}")

        return results

    def experiment_5c_multiseed(
        self,
        data_mode: str = "strict",
        seeds: list = None,
    ):
        """
        Experiment 5c：多种子统计验证。

        对 DR3L_full 和 DDPG 各跑 N 个种子，
        汇报均值 ± 标准差，验证 exp5b 结果的可重复性。
        DR3L_rho0 和 PPO 只跑 seed=42（已有结果，不重复跑）。
        """
        if seeds is None:
            seeds = [0, 1, 2, 42]   # 默认4个种子，含原始seed

        print("\n" + "=" * 80)
        print(f"EXPERIMENT 5c: Multi-Seed Validation (seeds={seeds})")
        print("=" * 80)

        output_dir = self.results_dir / data_mode
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "exp5c_multiseed.json"

        # 加载已有结果（支持断点续跑）
        all_results: dict = {}
        if output_file.exists() and not self.force_retrain:
            try:
                with open(output_file, "r") as f:
                    all_results = json.load(f)
                print(f"✅ 已加载已有结果: {list(all_results.keys())}")
            except Exception as e:
                print(f"⚠️  加载失败，从头开始: {e}")

        def save():
            with open(output_file, "w") as f:
                json.dump(all_results, f, indent=2, default=float)
            print(f"✅ 已保存: {output_file}")

        train_path = f"processed_data/{data_mode}/alice/train_rl.pkl.gz"
        test_path = f"processed_data/{data_mode}/alice/test_rl.pkl.gz"

        # 公共 env 参数（与 exp5b 完全一致）
        baseline_env_kw = dict(
            lambda_cvar=1.0,
            cvar_alpha=0.1,
            loss_buffer_maxlen=400,
            w_const=self.baseline_w_const,
        )
        # 与 experiment_5b FIXED Plan B 一致
        dr3l_phase1_env_kw = {
            **baseline_env_kw,
            "w_const": 2.0,
            "const_intent_flat": 1.0,
            "w_soc_center": 1.5,
            "lambda_cvar": 0.1,
        }
        dr3l_phase2_env_kw = {
            **baseline_env_kw,
            "w_const": 3.0,
            "const_intent_flat": 0.3,
            "w_soc_center": 1.0,
            "lambda_cvar": 0.5,
        }
        phase1_episodes = 300
        env_kw_phases = [
            (0, dr3l_phase1_env_kw),
            (phase1_episodes, dr3l_phase2_env_kw),
        ]

        # ── 对每个 seed 分别跑 DDPG 和 DR3L_full ──────────────────────
        for seed in seeds:
            seed_key = f"seed_{seed}"
            if seed_key not in all_results:
                all_results[seed_key] = {}

            # 设置种子
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            # —— DDPG ——
            ddpg_key = f"DDPG_seed{seed}"
            if ddpg_key not in all_results[seed_key]:
                print(f"\n[seed={seed}] Training DDPG...")
                try:
                    env_ddpg = self._make_env(train_path, mode='simple',
                                              **baseline_env_kw)
                    state_dim = int(np.prod(env_ddpg.observation_space.shape))
                    agent_ddpg = DDPG(state_dim=state_dim, device=self.device)
                    train_m = self.train_agent(
                        agent_ddpg, env_ddpg, n_episodes=500,
                        agent_name=f"DDPG_s{seed}",
                        wandb_run_name=f"DDPG_multiseed_s{seed}_{data_mode}",
                    )
                    test_env = self._make_env(test_path, mode='simple',
                                              **baseline_env_kw)
                    eval_m = self.evaluate_agent(agent_ddpg, test_env,
                                                 n_episodes=50, split='test')
                    all_results[seed_key][ddpg_key] = {
                        'train': train_m, 'test': eval_m, 'seed': seed
                    }
                    save()
                except Exception as e:
                    print(f"❌ DDPG seed={seed} 失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"\n⏭️  跳过 DDPG seed={seed}（已有结果）")

            # —— DR3L_full ——
            dr3l_key = f"DR3L_full_seed{seed}"
            if dr3l_key not in all_results[seed_key]:
                print(f"\n[seed={seed}] Training DR3L_full...")
                try:
                    env_dr3l = self._make_env(train_path, mode='multiscale',
                                              **dr3l_phase1_env_kw)
                    agent_dr3l = DR3L(
                        state_dim=10, cvar_alpha=0.1,
                        lambda_scale=0.5, rho_scale=2.0,
                        device=self.device,
                    )
                    train_m = self.train_agent(
                        agent_dr3l, env_dr3l, n_episodes=500,
                        agent_name=f"DR3L_full_s{seed}",
                        wandb_run_name=f"DR3L_full_multiseed_s{seed}_{data_mode}",
                        env_kw_phases=env_kw_phases,
                    )
                    test_env = self._make_env(test_path, mode='multiscale',
                                              **baseline_env_kw)
                    eval_m = self.evaluate_agent(agent_dr3l, test_env,
                                                 n_episodes=50, split='test')
                    all_results[seed_key][dr3l_key] = {
                        'train': train_m, 'test': eval_m, 'seed': seed
                    }
                    save()
                except Exception as e:
                    print(f"❌ DR3L_full seed={seed} 失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"\n⏭️  跳过 DR3L_full seed={seed}（已有结果）")

        # ── 汇总统计 ─────────────────────────────────────────────────
        print("\n" + "=" * 80)
        print("MULTI-SEED SUMMARY")
        print("=" * 80)

        summary = {}
        for algo_prefix in ["DDPG", "DR3L_full"]:
            returns, cvars, viols, ramps = [], [], [], []
            for seed in seeds:
                seed_key = f"seed_{seed}"
                algo_key = f"{algo_prefix}_seed{seed}"
                if seed_key in all_results and algo_key in all_results[seed_key]:
                    t = all_results[seed_key][algo_key]["test"]
                    returns.append(t.get("episode_returns_mean", float("nan")))
                    cvars.append(t.get("episode_cvars_mean", float("nan")))
                    viols.append(t.get("violation_any_rate_mean",
                                  t.get("episode_viol_any_mean", float("nan"))))
                    ramps.append(t.get("episode_viol_ramp_mean", float("nan")))

            if returns:
                summary[algo_prefix] = {
                    "n_seeds": len(returns),
                    "return_mean":  float(np.nanmean(returns)),
                    "return_std":   float(np.nanstd(returns)),
                    "return_all":   [float(x) for x in returns],
                    "cvar_mean":    float(np.nanmean(cvars)),
                    "cvar_std":     float(np.nanstd(cvars)),
                    "viol_mean":    float(np.nanmean(viols)),
                    "viol_std":     float(np.nanstd(viols)),
                    "viol_all":     [float(x) for x in viols],
                    "ramp_mean":    float(np.nanmean(ramps)),
                    "ramp_std":     float(np.nanstd(ramps)),
                }
                print(f"\n{algo_prefix} ({len(returns)} seeds):")
                print(f"  Return : {np.nanmean(returns):.1f} ± {np.nanstd(returns):.1f}  "
                      f"(per seed: {[f'{r:.0f}' for r in returns]})")
                print(f"  CVaR   : {np.nanmean(cvars):.3f} ± {np.nanstd(cvars):.3f}")
                print(f"  Viol%  : {100*np.nanmean(viols):.1f}% ± {100*np.nanstd(viols):.1f}%  "
                      f"(per seed: {[f'{100*v:.1f}%' for v in viols]})")
                print(f"  Ramp%  : {100*np.nanmean(ramps):.2f}% ± {100*np.nanstd(ramps):.2f}%")

        all_results["_summary"] = summary
        all_results["_seeds"] = seeds
        all_results["_config"] = {
            "phase1_episodes": phase1_episodes,
            "phase2_episodes": 500 - phase1_episodes,
            "max_norm": 1.0,
            "baseline_w_const": float(self.baseline_w_const),
            "dr3l_phase1": dr3l_phase1_env_kw,
            "dr3l_phase2": dr3l_phase2_env_kw,
        }
        save()
        return all_results

    def experiment_5d_stability(
        self,
        data_mode: str = "strict",
        seeds: list = None,
    ):
        """
        Experiment 5d：多种子下仅重训 DR3L_full（配置与 exp5b FIXED Plan B 相同，
        见 results/strict/FIXED_exp5b_dr3l_phased_results.md）。
        历史版本曾使用更强的 phase1 超参；现已与 5b/5c 对齐。
        """
        if seeds is None:
            seeds = [0, 1, 2, 42]

        print("\n" + "=" * 80)
        print(f"EXPERIMENT 5d: Stability Fix Validation (seeds={seeds})")
        print(
            "   DR3L 阶段1 300ep: w_const=2, intent=1.0, w_soc_center=1.5, λ_cvar=0.1；"
            "阶段2: w_const=3, intent=0.3, w_soc_center=1.0, λ_cvar=0.5"
        )
        print("=" * 80)

        output_dir = self.results_dir / data_mode
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "exp5d_stability.json"

        all_results: dict = {}
        if output_file.exists() and not self.force_retrain:
            try:
                with open(output_file, "r") as f:
                    all_results = json.load(f)
                print(f"✅ 已加载已有结果: {list(all_results.keys())}")
            except Exception as e:
                print(f"⚠️  加载失败，从头开始: {e}")

        def save():
            with open(output_file, "w") as f:
                json.dump(all_results, f, indent=2, default=float)
            print(f"✅ 已保存: {output_file}")

        train_path = f"processed_data/{data_mode}/alice/train_rl.pkl.gz"
        test_path = f"processed_data/{data_mode}/alice/test_rl.pkl.gz"

        baseline_env_kw = dict(
            lambda_cvar=1.0,
            cvar_alpha=0.1,
            loss_buffer_maxlen=400,
            w_const=self.baseline_w_const,
        )
        dr3l_phase1_env_kw = {
            **baseline_env_kw,
            "w_const": 2.0,
            "const_intent_flat": 1.0,
            "w_soc_center": 1.5,
            "lambda_cvar": 0.1,
        }
        dr3l_phase2_env_kw = {
            **baseline_env_kw,
            "w_const": 3.0,
            "const_intent_flat": 0.3,
            "w_soc_center": 1.0,
            "lambda_cvar": 0.5,
        }
        phase1_episodes = 300
        env_kw_phases = [
            (0, dr3l_phase1_env_kw),
            (phase1_episodes, dr3l_phase2_env_kw),
        ]

        for seed in seeds:
            seed_key = f"seed_{seed}"
            if seed_key not in all_results:
                all_results[seed_key] = {}

            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            dr3l_key = f"DR3L_full_seed{seed}"
            if dr3l_key not in all_results[seed_key]:
                print(f"\n[seed={seed}] Training DR3L_full (5d config)...")
                try:
                    env_dr3l = self._make_env(
                        train_path, mode='multiscale', **dr3l_phase1_env_kw
                    )
                    agent_dr3l = DR3L(
                        state_dim=10, cvar_alpha=0.1,
                        lambda_scale=0.5, rho_scale=2.0,
                        device=self.device,
                    )
                    train_m = self.train_agent(
                        agent_dr3l, env_dr3l,
                        n_episodes=500,
                        agent_name=f"DR3L_full_5d_s{seed}",
                        wandb_run_name=f"DR3L_full_5d_s{seed}_{data_mode}",
                        env_kw_phases=env_kw_phases,
                    )
                    test_env = self._make_env(
                        test_path, mode='multiscale', **baseline_env_kw
                    )
                    eval_m = self.evaluate_agent(
                        agent_dr3l, test_env, n_episodes=50, split='test'
                    )
                    all_results[seed_key][dr3l_key] = {
                        'train': train_m, 'test': eval_m, 'seed': seed
                    }
                    save()
                except Exception as e:
                    print(f"❌ DR3L_full 5d seed={seed} 失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"\n⏭️  跳过 DR3L_full seed={seed}（已有结果）")

        print("\n" + "=" * 80)
        print("EXPERIMENT 5d SUMMARY")
        print("=" * 80)

        returns, cvars, viols, ramps = [], [], [], []
        for seed in seeds:
            seed_key = f"seed_{seed}"
            algo_key = f"DR3L_full_seed{seed}"
            if seed_key in all_results and algo_key in all_results[seed_key]:
                t = all_results[seed_key][algo_key]["test"]
                returns.append(t.get("episode_returns_mean", float("nan")))
                cvars.append(t.get("episode_cvars_mean", float("nan")))
                viols.append(t.get("violation_any_rate_mean",
                              t.get("episode_viol_any_mean", float("nan"))))
                ramps.append(t.get("episode_viol_ramp_mean", float("nan")))

        if returns:
            print(f"\nDR3L_full 5d ({len(returns)} seeds):")
            print(f"  Return : {np.nanmean(returns):.1f} ± "
                  f"{np.nanstd(returns):.1f}  "
                  f"(per seed: {[f'{r:.0f}' for r in returns]})")
            print(f"  CVaR   : {np.nanmean(cvars):.3f} ± "
                  f"{np.nanstd(cvars):.3f}")
            print(f"  Viol%  : {100*np.nanmean(viols):.1f}% ± "
                  f"{100*np.nanstd(viols):.1f}%  "
                  f"(per seed: {[f'{100*v:.1f}%' for v in viols]})")
            print(f"  Ramp%  : {100*np.nanmean(ramps):.2f}% ± "
                  f"{100*np.nanstd(ramps):.2f}%")

            print(f"\n对比基线 DDPG (from exp5c):")
            print(f"  Return : -1897 ± 55")
            print(f"  CVaR   : 3.203 ± 0.090")
            print(f"  Viol%  : 6.4% ± 1.8%")

            viol_std = 100 * np.nanstd(viols)
            if viol_std < 15.0:
                print(f"\n✅ 稳定性达标：Viol% std={viol_std:.1f}% < 15%")
            else:
                print(f"\n⚠️  稳定性未达标：Viol% std={viol_std:.1f}% ≥ 15%，"
                      f"考虑进一步提高 const_intent_flat 或延长阶段1")

            all_results["_summary_5d"] = {
                "n_seeds": len(returns),
                "return_mean": float(np.nanmean(returns)),
                "return_std":  float(np.nanstd(returns)),
                "return_all":  [float(x) for x in returns],
                "cvar_mean":   float(np.nanmean(cvars)),
                "cvar_std":    float(np.nanstd(cvars)),
                "viol_mean":   float(np.nanmean(viols)),
                "viol_std":    float(np.nanstd(viols)),
                "viol_all":    [float(x) for x in viols],
                "ramp_mean":   float(np.nanmean(ramps)),
                "ramp_std":    float(np.nanstd(ramps)),
            }
            save()

        return all_results

    # ------------------------------------------------------------------
    # Ablation: DDPG vs DR3L-EMA (full) vs DR3L-Quantile (N=8, no adaptive)
    # ------------------------------------------------------------------
    def experiment_ablation(
        self,
        data_mode: str = "strict",
        seeds: list = None,
    ):
        """Multi-seed ablation comparing DDPG, DR3L_full (EMA), and
        DR3L_Quantile (8-quantile, fixed-lambda, no adversarial).
        """
        if seeds is None:
            seeds = [0, 1, 2, 42]

        print("\n" + "=" * 80)
        print(f"ABLATION: DDPG vs DR3L_full vs DR3L_Quantile  (seeds={seeds})")
        print("=" * 80)

        output_dir = self.results_dir / data_mode
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "ablation_quantile.json"

        all_results: dict = {}
        if output_file.exists() and not self.force_retrain:
            try:
                with open(output_file) as f:
                    all_results = json.load(f)
                print(f"✅ 已加载已有结果: {list(all_results.keys())}")
            except Exception as e:
                print(f"⚠️  加载失败: {e}")

        def save():
            with open(output_file, "w") as f:
                json.dump(all_results, f, indent=2, default=float)
            print(f"✅ 已保存: {output_file}")

        train_path = f"processed_data/{data_mode}/alice/train_rl.pkl.gz"
        test_path = f"processed_data/{data_mode}/alice/test_rl.pkl.gz"

        baseline_env_kw = dict(
            lambda_cvar=1.0, cvar_alpha=0.1,
            loss_buffer_maxlen=400,
            w_const=self.baseline_w_const,
        )
        # FIXED Plan B phases (same as exp5b)
        dr3l_phase1_env_kw = {
            **baseline_env_kw,
            "w_const": 2.0, "const_intent_flat": 1.0,
            "w_soc_center": 1.5, "lambda_cvar": 0.1,
        }
        dr3l_phase2_env_kw = {
            **baseline_env_kw,
            "w_const": 3.0, "const_intent_flat": 0.3,
            "w_soc_center": 1.0, "lambda_cvar": 0.5,
        }
        phase1_episodes = 300
        env_kw_phases = [
            (0, dr3l_phase1_env_kw),
            (phase1_episodes, dr3l_phase2_env_kw),
        ]

        ALGOS = [
            ("DDPG",          "simple",     None),
            ("DR3L_full",     "multiscale", env_kw_phases),
            ("DR3L_Quantile", "multiscale", env_kw_phases),
        ]

        for seed in seeds:
            seed_key = f"seed_{seed}"
            if seed_key not in all_results:
                all_results[seed_key] = {}

            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            for algo_name, obs_mode, phases in ALGOS:
                run_key = f"{algo_name}_seed{seed}"
                if run_key in all_results[seed_key]:
                    print(f"\n⏭️  跳过 {run_key}（已有结果）")
                    continue

                print(f"\n[seed={seed}] Training {algo_name}...")
                try:
                    init_env_kw = dr3l_phase1_env_kw if phases else baseline_env_kw
                    env = self._make_env(
                        train_path, mode=obs_mode, **init_env_kw,
                    )

                    if algo_name == "DDPG":
                        state_dim = int(np.prod(env.observation_space.shape))
                        agent = DDPG(state_dim=state_dim, device=self.device)
                    elif algo_name == "DR3L_full":
                        agent = DR3L(
                            state_dim=10, cvar_alpha=0.1,
                            lambda_scale=0.5, rho_scale=2.0,
                            device=self.device,
                        )
                    elif algo_name == "DR3L_Quantile":
                        agent = DR3LQuantile(
                            state_dim=10, num_quantiles=8,
                            cvar_alpha=0.1, fixed_lambda=0.5,
                            device=self.device,
                        )
                    else:
                        raise ValueError(algo_name)

                    kw_train = {}
                    if phases:
                        kw_train["env_kw_phases"] = phases

                    train_m = self.train_agent(
                        agent, env, n_episodes=500,
                        agent_name=f"{algo_name}_s{seed}",
                        wandb_run_name=f"{algo_name}_ablation_s{seed}_{data_mode}",
                        **kw_train,
                    )
                    test_env = self._make_env(
                        test_path, mode=obs_mode, **baseline_env_kw,
                    )
                    eval_m = self.evaluate_agent(
                        agent, test_env, n_episodes=50, split="test",
                    )
                    all_results[seed_key][run_key] = {
                        "train": train_m, "test": eval_m, "seed": seed,
                    }
                    save()
                except Exception as e:
                    print(f"❌ {run_key} 失败: {e}")
                    import traceback; traceback.print_exc()

        # ── Summary ──
        print("\n" + "=" * 80)
        print("ABLATION SUMMARY")
        print("=" * 80)

        summary = {}
        for algo_name, _, _ in ALGOS:
            rets, cvars, maxls, viols = [], [], [], []
            for seed in seeds:
                sk, rk = f"seed_{seed}", f"{algo_name}_seed{seed}"
                if sk in all_results and rk in all_results[sk]:
                    t = all_results[sk][rk]["test"]
                    rets.append(t.get("episode_returns_mean", float("nan")))
                    cvars.append(t.get("episode_cvars_mean", float("nan")))
                    maxls.append(t.get("episode_max_losses_mean", float("nan")))
                    viols.append(
                        t.get("violation_any_rate_mean",
                              t.get("episode_viol_any_mean", float("nan")))
                    )
            if rets:
                summary[algo_name] = {
                    "return_mean":  float(np.nanmean(rets)),
                    "return_std":   float(np.nanstd(rets)),
                    "cvar_mean":    float(np.nanmean(cvars)),
                    "cvar_std":     float(np.nanstd(cvars)),
                    "max_loss_mean": float(np.nanmean(maxls)),
                    "max_loss_std":  float(np.nanstd(maxls)),
                    "viol_mean":    float(np.nanmean(viols)),
                    "viol_std":     float(np.nanstd(viols)),
                    "viol_all":     [float(v) for v in viols],
                }
                print(f"\n{algo_name} ({len(rets)} seeds):")
                print(f"  Return   : {np.nanmean(rets):.1f} ± {np.nanstd(rets):.1f}")
                print(f"  CVaR     : {np.nanmean(cvars):.3f} ± {np.nanstd(cvars):.3f}")
                print(f"  Max Loss : {np.nanmean(maxls):.3f} ± {np.nanstd(maxls):.3f}")
                print(f"  Viol%    : {100*np.nanmean(viols):.1f}% ± {100*np.nanstd(viols):.1f}%")

        all_results["_summary"] = summary
        all_results["_seeds"] = seeds
        all_results["_config"] = {
            "phase1_episodes": phase1_episodes,
            "baseline_w_const": float(self.baseline_w_const),
            "dr3l_full": {"rho_scale": 2.0, "lambda_scale": 0.5, "num_quantiles": 51},
            "dr3l_quantile": {"num_quantiles": 8, "fixed_lambda": 0.5,
                              "rho_scale": 0, "epsilon": 0},
        }
        save()
        return all_results

    def plot_lambda_tradeoff(self, results: Dict, data_mode: str):
        """Plot λ trade-off curves (handles incomplete results)"""
        lambdas = [0.0, 0.1, 0.5, 1.0, 2.0]
        available_lambdas = []
        cvars = []
        returns = []
        
        for lam in lambdas:
            key = f"lambda_{lam}"
            if key in results and 'test' in results[key] and 'episode_cvars_mean' in results[key]['test']:
                available_lambdas.append(lam)
                cvars.append(results[key]['test']['episode_cvars_mean'])
                returns.append(results[key]['test']['episode_returns_mean'])
        
        if not available_lambdas:
            print("⚠️  No complete results to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(available_lambdas, cvars, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('λ (Risk Weight)', fontsize=12)
        ax1.set_ylabel('CVaR₀.₁', fontsize=12)
        ax1.set_title('Risk vs λ', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(available_lambdas, returns, 's-', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('λ (Risk Weight)', fontsize=12)
        ax2.set_ylabel('Episode Return', fontsize=12)
        ax2.set_title('Return vs λ', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / data_mode / "lambda_tradeoff.png", dpi=300)
        plt.close()
        print(f"✅ Plot saved: {self.results_dir / data_mode / 'lambda_tradeoff.png'}")
    
    def plot_robust_comparison(self, results: Dict, data_mode: str):
        """Plot robust vs non-robust comparison (handles incomplete results)"""
        rhos = [0.0, 0.01, 0.05]
        available_rhos = []
        cvars = []
        
        for rho in rhos:
            key = f"rho_{rho}"
            if key in results and 'test' in results[key] and 'episode_cvars_mean' in results[key]['test']:
                available_rhos.append(rho)
                cvars.append(results[key]['test']['episode_cvars_mean'])
        
        if not available_rhos:
            print("⚠️  No complete results to plot")
            return
        
        colors = ['red', 'orange', 'green'][:len(available_rhos)]
        plt.figure(figsize=(8, 5))
        plt.bar(range(len(available_rhos)), cvars, color=colors)
        plt.xticks(range(len(available_rhos)), [f'ρ={rho}' for rho in available_rhos])
        plt.ylabel('CVaR₀.₁', fontsize=12)
        plt.title('Robustness Comparison', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.results_dir / data_mode / "robust_comparison.png", dpi=300)
        plt.close()
        print(f"✅ Plot saved: {self.results_dir / data_mode / 'robust_comparison.png'}")
    
    def plot_data_quality(self, results: Dict):
        """Plot data quality comparison (handles incomplete results)"""
        modes = ['strict', 'light', 'raw']
        available_modes = []
        cvars = []
        
        for mode in modes:
            if mode in results and 'test' in results[mode] and 'episode_cvars_mean' in results[mode]['test']:
                available_modes.append(mode)
                cvars.append(results[mode]['test']['episode_cvars_mean'])
        
        if not available_modes:
            print("⚠️  No complete results to plot")
            return
        
        colors = ['green', 'orange', 'red'][:len(available_modes)]
        plt.figure(figsize=(8, 5))
        plt.bar(range(len(available_modes)), cvars, color=colors)
        plt.xticks(range(len(available_modes)), [m.upper() for m in available_modes])
        plt.ylabel('CVaR₀.₁', fontsize=12)
        plt.title('Data Quality Impact', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.results_dir / "data_quality_comparison.png", dpi=300)
        plt.close()
        print(f"✅ Plot saved: {self.results_dir / 'data_quality_comparison.png'}")
    
    def plot_baselines(self, results: Dict, data_mode: str, plot_suffix: str = ""):
        """论文级三图：Return、CVaR、Test 集逐步违规率（SoC 或 Ramp 任一）。"""
        methods = ['DDPG', 'PPO', 'DR3L_rho0', 'DR3L_full']
        available_methods = [m for m in methods if m in results and 'test' in results[m] and 'episode_cvars_mean' in results[m]['test']]
        
        if not available_methods:
            print("⚠️  No complete results to plot")
            return
        
        cvars = [results[m]['test']['episode_cvars_mean'] for m in available_methods]
        returns = [results[m]['test']['episode_returns_mean'] for m in available_methods]
        viol_pct = []
        for m in available_methods:
            t = results[m]['test']
            r = float(t.get('violation_any_rate_mean', t.get('episode_viol_any_mean', 0.0)))
            if np.isnan(r):
                r = 0.0
            viol_pct.append(100.0 * r)
        
        colors = ['#4477AA', '#EE7733', '#CC3311', '#009988'][:len(available_methods)]
        
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14.5, 4.2))
        
        x = np.arange(len(available_methods))
        ax0.bar(x, returns, color=colors, edgecolor='black', linewidth=0.4)
        ax0.set_xticks(x)
        ax0.set_xticklabels(available_methods, rotation=18, ha='right')
        ax0.set_ylabel('Mean episode return (test)', fontsize=11)
        ax0.set_title('Return', fontsize=13)
        ax0.grid(True, alpha=0.3, axis='y')
        
        ax1.bar(x, cvars, color=colors, edgecolor='black', linewidth=0.4)
        ax1.set_xticks(x)
        ax1.set_xticklabels(available_methods, rotation=18, ha='right')
        ax1.set_ylabel('CVaR₀.₁ (episode loss tail)', fontsize=11)
        ax1.set_title('Tail risk (CVaR)', fontsize=13)
        ax1.grid(True, alpha=0.3, axis='y')
        
        ax2.bar(x, viol_pct, color=colors, edgecolor='black', linewidth=0.4)
        ax2.set_xticks(x)
        ax2.set_xticklabels(available_methods, rotation=18, ha='right')
        ax2.set_ylabel('Violation rate (%)', fontsize=11)
        ax2.set_title('Constraint intent (test)', fontsize=13)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        w_tag = self._exp5_wconst_tag()
        plot_name = f"baseline_comparison_wconst_{w_tag}{plot_suffix}.png"
        plt.savefig(self.results_dir / data_mode / plot_name, dpi=300)
        plt.close()
        print(f"✅ Plot saved: {self.results_dir / data_mode / plot_name}")


def main():
    parser = argparse.ArgumentParser(description='IEEE TSG DR3L Experiments with WandB Support')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'lambda', 'robust', 'shift', 'quality',
                                'baselines', 'baselines_phaseB', 'multiseed', 'stability',
                                'ablation'],
                       help='Experiment to run')
    parser.add_argument('--data_mode', type=str, default='strict',
                       choices=['strict', 'light', 'raw'],
                       help='Data preprocessing mode')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable WandB logging (default: WandB on if installed)')
    parser.add_argument('--wandb_project', type=str, default='dr3l-pv-bess',
                       help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='WandB entity/team name (optional)')
    parser.add_argument('--checkpoint_root', type=str, default='results/checkpoints',
                       help='Directory for per-run training checkpoints (agent.pt + training_meta.json)')
    parser.add_argument('--no_checkpoint', action='store_true',
                       help='Disable saving training checkpoints')
    parser.add_argument('--resume', action='store_true',
                       help='Resume train_agent from checkpoint under checkpoint_root/<run_name>')
    parser.add_argument('--checkpoint_every', type=int, default=50,
                       help='Save checkpoint every N completed episodes')
    parser.add_argument('--force_retrain', action='store_true',
                       help='Experiment baselines: 忽略当前 w_const 下 JSON 的跳过逻辑，备份后全量重训')
    parser.add_argument('--baseline_w_const', type=float, default=5.0,
                       help='Experiment 5：经 baseline_env_kw 传给环境的 w_const（默认 5.0；可与 3.0 对比试跑）')
    parser.add_argument('--debug_violation_env', action='store_true',
                       help='仅调试：p_desired×2。论文/正式实验不要加此参数。')
    parser.add_argument('--env_wandb_log_interval', type=int, default=0,
                       help='If >0, env step() logs violation/* and debug/p_* to WandB every N global env steps')
    parser.add_argument('--env_debug_print_interval', type=int, default=0,
                       help='If >0, print p_desired buffer min/max and soc every N global env steps')
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        seed=args.seed,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        checkpoint_root=None if args.no_checkpoint else args.checkpoint_root,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        force_retrain=args.force_retrain,
        debug_violation_env=args.debug_violation_env,
        env_wandb_log_interval=args.env_wandb_log_interval,
        env_debug_print_interval=args.env_debug_print_interval,
        baseline_w_const=args.baseline_w_const,
    )
    
    if args.experiment == 'all':
        runner.experiment_5_baselines(args.data_mode)
        runner.experiment_1_lambda_tradeoff(args.data_mode)
        runner.experiment_2_robust_comparison(args.data_mode)
        runner.experiment_3_distribution_shift()
        runner.experiment_4_data_quality()
    elif args.experiment == 'lambda':
        runner.experiment_1_lambda_tradeoff(args.data_mode)
    elif args.experiment == 'robust':
        runner.experiment_2_robust_comparison(args.data_mode)
    elif args.experiment == 'shift':
        runner.experiment_3_distribution_shift()
    elif args.experiment == 'quality':
        runner.experiment_4_data_quality()
    elif args.experiment == 'baselines':
        runner.experiment_5_baselines(args.data_mode)
    elif args.experiment == 'baselines_phaseB':
        runner.experiment_5b_dr3l_phased(args.data_mode)
    elif args.experiment == 'multiseed':
        runner.experiment_5c_multiseed(
            data_mode=args.data_mode,
            seeds=[0, 1, 2, 42],
        )
    elif args.experiment == 'stability':
        runner.experiment_5d_stability(
            data_mode=args.data_mode,
            seeds=[0, 1, 2, 42],
        )
    elif args.experiment == 'ablation':
        runner.experiment_ablation(
            data_mode=args.data_mode,
            seeds=[0, 1, 2, 42],
        )

    print("\n" + "="*80)
    print("All experiments completed!")
    print(f"Results saved to: {runner.results_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
