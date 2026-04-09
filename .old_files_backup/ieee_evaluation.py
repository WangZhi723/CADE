"""
IEEE TSG级别的统一评估函数
用于生成论文级别的评估指标
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle
import gzip


def evaluate_policy_ieee(
    agent,
    env,
    n_episodes: int = 50,
    cvar_alpha: float = 0.1,
    loss_threshold: float = 0.8,
    device: str = 'cpu'
) -> Dict:
    """
    IEEE TSG标准评估函数
    
    Args:
        agent: 训练好的智能体（DDPG/PPO/DR3L）
        env: 测试环境
        n_episodes: 评估episode数量
        cvar_alpha: CVaR置信水平（默认0.1表示最差10%）
        loss_threshold: 尾部风险阈值
        device: 计算设备
    
    Returns:
        评估指标字典，包含：
        - mean_reward: 平均奖励 ↑
        - std_reward: 奖励标准差
        - cvar_0.1: CVaR@10% ↑ (越大越好，表示风险越低)
        - rmse: PV预测RMSE ↓
        - tail_prob: P(loss > threshold) ↓
        - max_loss: 最大损失
        - mean_loss: 平均损失
        - sharpe_ratio: Sharpe比率 ↑
    """
    from algorithms import DDPG, PPOAgent, DR3L
    
    # 收集所有episode的指标
    episode_rewards = []
    episode_losses = []
    episode_pv_errors = []
    episode_returns = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step_losses = []
        step_pv_errors = []
        
        while not done:
            # 根据智能体类型选择动作（确定性策略）
            if isinstance(agent, DDPG):
                action = agent.select_action(obs, noise=0.0)
            elif isinstance(agent, PPOAgent):
                action, _ = agent.select_action(obs)
            elif isinstance(agent, DR3L):
                short_term = obs['short_term']
                long_term = obs['long_term']
                action, _ = agent.select_action(short_term, long_term)
            else:
                raise ValueError(f"Unknown agent type: {type(agent)}")
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            # 收集步级指标
            if 'loss' in info:
                step_losses.append(info['loss'])
            if 'pv_error' in info:
                step_pv_errors.append(info['pv_error'])
        
        # 记录episode级指标
        episode_rewards.append(episode_reward)
        episode_returns.append(info.get('episode_return', episode_reward))
        
        if step_losses:
            episode_losses.extend(step_losses)
        if step_pv_errors:
            episode_pv_errors.extend(step_pv_errors)
    
    # 转换为numpy数组
    episode_rewards = np.array(episode_rewards)
    episode_losses = np.array(episode_losses) if episode_losses else np.array([0.0])
    episode_pv_errors = np.array(episode_pv_errors) if episode_pv_errors else np.array([0.0])
    
    # 计算核心指标
    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    
    # CVaR@10%: 最差10%的平均奖励
    # 注意：奖励是负数，所以最差10%是最小的10%
    sorted_rewards = np.sort(episode_rewards)
    n_worst = max(1, int(cvar_alpha * len(sorted_rewards)))
    cvar_worst_rewards = sorted_rewards[:n_worst]
    cvar_value = float(np.mean(cvar_worst_rewards))
    
    # 为了使CVaR指标更直观（越大越好），我们计算"风险归一化CVaR"
    # 将其转换为[0,1]范围，其中1表示最好
    # 使用损失的归一化版本
    if len(episode_losses) > 0:
        cvar_loss = float(np.quantile(episode_losses, 1 - cvar_alpha))  # 90%分位数
        # 归一化到[0,1]，越小越好的损失转换为越大越好的指标
        cvar_metric = 1.0 - min(cvar_loss, 1.0)
    else:
        cvar_metric = 0.5  # 默认值
    
    # RMSE: PV预测误差
    rmse = float(np.sqrt(np.mean(episode_pv_errors**2))) if len(episode_pv_errors) > 0 else 0.0
    
    # 尾部概率: P(loss > threshold)
    tail_prob = float(np.mean(episode_losses > loss_threshold)) if len(episode_losses) > 0 else 0.0
    
    # 最大和平均损失
    max_loss = float(np.max(episode_losses)) if len(episode_losses) > 0 else 0.0
    mean_loss = float(np.mean(episode_losses)) if len(episode_losses) > 0 else 0.0
    
    # Sharpe比率: 风险调整后的收益
    sharpe_ratio = float(mean_reward / std_reward) if std_reward > 0 else 0.0
    
    # 返回IEEE标准指标
    return {
        # 主要性能指标
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'cvar_0.1': cvar_metric,  # 归一化的CVaR指标（越大越好）
        'cvar_reward': cvar_value,  # 原始CVaR奖励值
        
        # 风险指标
        'rmse': rmse,
        'tail_prob': tail_prob,
        'max_loss': max_loss,
        'mean_loss': mean_loss,
        
        # 综合指标
        'sharpe_ratio': sharpe_ratio,
        
        # 原始数据（用于进一步分析）
        'episode_rewards': episode_rewards.tolist(),
        'episode_losses': episode_losses.tolist()[:100],  # 限制长度
    }


def load_agent_from_checkpoint(
    agent_type: str,
    checkpoint_dir: Path,
    state_dim: int = 10,
    device: str = 'cpu'
):
    """
    从检查点加载智能体
    
    Args:
        agent_type: 'DDPG', 'PPO', 'DR3L'
        checkpoint_dir: 检查点目录
        state_dim: 状态维度
        device: 计算设备
    
    Returns:
        加载好的智能体
    """
    from algorithms import DDPG, PPOAgent, DR3L
    
    if agent_type == 'DDPG':
        agent = DDPG(state_dim=state_dim, device=device)
        # DDPG通常保存actor和critic
        if (checkpoint_dir / 'actor.pt').exists():
            agent.actor.load_state_dict(torch.load(checkpoint_dir / 'actor.pt', map_location=device))
        if (checkpoint_dir / 'critic.pt').exists():
            agent.critic.load_state_dict(torch.load(checkpoint_dir / 'critic.pt', map_location=device))
    
    elif agent_type == 'PPO':
        agent = PPOAgent(state_dim=state_dim, device=device)
        if (checkpoint_dir / 'actor.pt').exists():
            agent.actor.load_state_dict(torch.load(checkpoint_dir / 'actor.pt', map_location=device))
        if (checkpoint_dir / 'critic.pt').exists():
            agent.critic.load_state_dict(torch.load(checkpoint_dir / 'critic.pt', map_location=device))
    
    elif agent_type == 'DR3L':
        agent = DR3L(state_dim=state_dim, device=device)
        if (checkpoint_dir / 'feature_net.pt').exists():
            agent.feature_net.load_state_dict(torch.load(checkpoint_dir / 'feature_net.pt', map_location=device))
        if (checkpoint_dir / 'actor.pt').exists():
            agent.actor.load_state_dict(torch.load(checkpoint_dir / 'actor.pt', map_location=device))
        if (checkpoint_dir / 'critic.pt').exists():
            agent.critic.load_state_dict(torch.load(checkpoint_dir / 'critic.pt', map_location=device))
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent.eval()  # 设置为评估模式
    return agent


def batch_evaluate_experiments(
    results_dir: Path,
    data_mode: str,
    experiment_name: str,
    agent_configs: List[Dict],
    test_data_path: str,
    n_episodes: int = 50
) -> Dict:
    """
    批量评估实验配置
    
    Args:
        results_dir: 结果目录
        data_mode: 数据模式 ('strict', 'light', 'raw')
        experiment_name: 实验名称
        agent_configs: 智能体配置列表 [{'name': 'lambda_0.0', 'type': 'DR3L', 'dir': ...}, ...]
        test_data_path: 测试数据路径
        n_episodes: 评估episode数
    
    Returns:
        评估结果字典
    """
    from pv_env import make_env
    
    results = {}
    test_env = make_env(test_data_path, mode='multiscale')
    
    for config in agent_configs:
        agent_name = config['name']
        agent_type = config['type']
        checkpoint_dir = config['dir']
        
        print(f"Evaluating {agent_name}...")
        
        try:
            # 加载智能体
            agent = load_agent_from_checkpoint(agent_type, checkpoint_dir)
            
            # 评估
            metrics = evaluate_policy_ieee(agent, test_env, n_episodes=n_episodes)
            results[agent_name] = metrics
            
            print(f"  ✅ {agent_name}: Reward={metrics['mean_reward']:.2f}, CVaR={metrics['cvar_0.1']:.4f}")
        
        except Exception as e:
            print(f"  ❌ {agent_name}: {e}")
            results[agent_name] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    """测试评估函数"""
    import sys
    from pv_env import make_env
    from algorithms import DR3L
    
    # 创建测试环境
    test_path = "processed_data/strict/alice/test_rl.pkl.gz"
    test_env = make_env(test_path, mode='multiscale')
    
    # 创建测试智能体
    agent = DR3L(state_dim=10, device='cpu')
    
    # 评估
    print("Testing IEEE evaluation function...")
    metrics = evaluate_policy_ieee(agent, test_env, n_episodes=5)
    
    print("\n评估结果:")
    for key, value in metrics.items():
        if not isinstance(value, list):
            print(f"  {key}: {value}")
    
    print("\n✅ 评估函数测试完成！")
