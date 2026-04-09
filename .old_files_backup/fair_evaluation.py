"""
公平评估：使用统一的评估指标对比所有算法
不使用任何算法内部的penalty，只评估实际物理性能
"""

import numpy as np
import torch
import json
import pickle
import gzip
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from pv_env import make_env
from algorithms import DDPG, PPOAgent, DR3L


class FairEvaluator:
    """公平评估器 - 使用统一的物理指标"""
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def compute_base_reward_only(self, pv_actual, p_battery, load, hour):
        """
        纯粹的经济收益，不含任何penalty
        
        Args:
            pv_actual: 实际光伏输出 (MW)
            p_battery: 储能功率 (MW, 正为放电)
            load: 负荷需求 (MW)
            hour: 当前小时
        
        Returns:
            base_reward: 纯经济收益
        """
        # 电价（简化版）
        if 9 <= hour <= 21:  # 白天电价高
            price_sell = 0.8  # 元/kWh
            price_buy = 0.6
        else:  # 夜间电价低
            price_sell = 0.4
            price_buy = 0.3
        
        # 净输出功率
        p_net = pv_actual + p_battery  # 正为向电网输出
        
        # 满足负荷后的剩余/缺口
        p_surplus = p_net - load
        
        if p_surplus > 0:
            # 有剩余，卖电
            revenue = p_surplus * price_sell * 1000  # MW -> kW
        else:
            # 有缺口，买电
            revenue = p_surplus * price_buy * 1000  # 负数，表示成本
        
        return revenue
    
    def compute_tracking_rmse(self, pv_actual_list, pv_forecast_list):
        """
        计算光伏预测的RMSE
        
        Args:
            pv_actual_list: 实际光伏输出列表
            pv_forecast_list: 预测光伏输出列表
        
        Returns:
            rmse: 均方根误差
        """
        errors = np.array(pv_actual_list) - np.array(pv_forecast_list)
        rmse = np.sqrt(np.mean(errors**2))
        return rmse
    
    def compute_cvar_energy_gap(self, energy_gaps, alpha=0.1):
        """
        计算能量缺口的CVaR（最差10%的平均）
        
        Args:
            energy_gaps: 能量缺口列表
            alpha: CVaR置信水平
        
        Returns:
            cvar: CVaR值
        """
        if len(energy_gaps) == 0:
            return 0.0
        
        sorted_gaps = np.sort(energy_gaps)[::-1]  # 降序
        n_tail = max(1, int(alpha * len(sorted_gaps)))
        cvar = sorted_gaps[:n_tail].mean()
        
        return cvar
    
    def compute_violation_rate(self, soc_history, p_battery_history, 
                               soc_min=0.1, soc_max=0.9, 
                               power_rated=0.5, ramp_rate_max=0.3):
        """
        计算约束违反率
        
        Args:
            soc_history: SoC历史
            p_battery_history: 储能功率历史
            soc_min, soc_max: SoC约束
            power_rated: 额定功率
            ramp_rate_max: 最大爬坡率
        
        Returns:
            violation_dict: 各类违反的统计
        """
        n_steps = len(soc_history)
        
        # SoC违反
        soc_violations = sum(1 for soc in soc_history 
                            if soc < soc_min or soc > soc_max)
        
        # 功率违反
        power_violations = sum(1 for p in p_battery_history 
                              if abs(p) > power_rated)
        
        # 爬坡违反
        ramp_violations = 0
        for i in range(1, len(p_battery_history)):
            ramp = abs(p_battery_history[i] - p_battery_history[i-1]) / power_rated
            if ramp > ramp_rate_max:
                ramp_violations += 1
        
        return {
            'soc_violation_rate': soc_violations / n_steps,
            'power_violation_rate': power_violations / n_steps,
            'ramp_violation_rate': ramp_violations / max(1, n_steps - 1),
            'total_violation_rate': (soc_violations + power_violations + ramp_violations) / (n_steps * 3)
        }
    
    def evaluate_agent_fair(self, agent, env, n_episodes=50):
        """
        公平评估一个智能体
        
        Returns:
            metrics: 统一的评估指标
        """
        print(f"  公平评估 {n_episodes} episodes...")
        
        # 收集所有episode的数据
        all_base_rewards = []
        all_pv_actuals = []
        all_pv_forecasts = []
        all_energy_gaps = []
        all_soc_histories = []
        all_p_battery_histories = []
        
        for episode in tqdm(range(n_episodes), desc="  Evaluating"):
            obs = env.reset()
            done = False
            
            episode_base_rewards = []
            episode_pv_actuals = []
            episode_pv_forecasts = []
            episode_energy_gaps = []
            episode_soc = []
            episode_p_battery = []
            
            step = 0
            
            while not done:
                # 选择动作（确定性）
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
                
                # 执行动作
                obs_next, reward, done, info = env.step(action)
                
                # 提取物理量
                pv_actual = info.get('pv_actual', 0)
                load = info.get('load', 0)
                p_battery = info.get('p_battery', 0)
                soc = info.get('soc', 0.5)
                
                # 计算纯经济收益
                hour = (step * 5 / 60.0) % 24
                base_reward = self.compute_base_reward_only(
                    pv_actual, p_battery, load, int(hour)
                )
                
                # 计算能量缺口
                energy_gap = max(0, load - pv_actual - p_battery)
                
                # 记录
                episode_base_rewards.append(base_reward)
                episode_pv_actuals.append(pv_actual)
                episode_energy_gaps.append(energy_gap)
                episode_soc.append(soc)
                episode_p_battery.append(p_battery)
                
                obs = obs_next
                step += 1
            
            # 保存episode数据
            all_base_rewards.extend(episode_base_rewards)
            all_pv_actuals.extend(episode_pv_actuals)
            all_energy_gaps.extend(episode_energy_gaps)
            all_soc_histories.append(episode_soc)
            all_p_battery_histories.append(episode_p_battery)
        
        # 计算统一指标
        metrics = {
            # 1. 纯经济收益
            'base_reward_mean': float(np.mean(all_base_rewards)),
            'base_reward_std': float(np.std(all_base_rewards)),
            'base_reward_total': float(np.sum(all_base_rewards)),
            
            # 2. 能量缺口CVaR
            'cvar_energy_gap': float(self.compute_cvar_energy_gap(all_energy_gaps)),
            'mean_energy_gap': float(np.mean(all_energy_gaps)),
            'max_energy_gap': float(np.max(all_energy_gaps)),
            
            # 3. 约束违反率
            'violation_rates': {},
            
            # 4. 稳定性指标
            'base_reward_cv': float(np.std(all_base_rewards) / abs(np.mean(all_base_rewards))) if np.mean(all_base_rewards) != 0 else 0,
        }
        
        # 计算所有episode的平均违反率
        all_violations = []
        for soc_hist, p_hist in zip(all_soc_histories, all_p_battery_histories):
            viol = self.compute_violation_rate(soc_hist, p_hist)
            all_violations.append(viol)
        
        metrics['violation_rates'] = {
            'soc_violation_rate': float(np.mean([v['soc_violation_rate'] for v in all_violations])),
            'power_violation_rate': float(np.mean([v['power_violation_rate'] for v in all_violations])),
            'ramp_violation_rate': float(np.mean([v['ramp_violation_rate'] for v in all_violations])),
            'total_violation_rate': float(np.mean([v['total_violation_rate'] for v in all_violations]))
        }
        
        return metrics
    
    def load_trained_model(self, model_type, model_dir, state_dim=10):
        """加载训练好的模型"""
        if model_type == 'DDPG':
            agent = DDPG(state_dim=state_dim, device=self.device)
            if (model_dir / 'actor.pt').exists():
                agent.actor.load_state_dict(torch.load(model_dir / 'actor.pt', map_location=self.device))
            if (model_dir / 'critic.pt').exists():
                agent.critic.load_state_dict(torch.load(model_dir / 'critic.pt', map_location=self.device))
        
        elif model_type == 'PPO':
            agent = PPOAgent(state_dim=state_dim, device=self.device)
            if (model_dir / 'actor.pt').exists():
                agent.actor.load_state_dict(torch.load(model_dir / 'actor.pt', map_location=self.device))
            if (model_dir / 'critic.pt').exists():
                agent.critic.load_state_dict(torch.load(model_dir / 'critic.pt', map_location=self.device))
        
        elif model_type == 'DR3L':
            agent = DR3L(state_dim=state_dim, device=self.device)
            if (model_dir / 'feature_net.pt').exists():
                agent.feature_net.load_state_dict(torch.load(model_dir / 'feature_net.pt', map_location=self.device))
            if (model_dir / 'actor.pt').exists():
                agent.actor.load_state_dict(torch.load(model_dir / 'actor.pt', map_location=self.device))
            if (model_dir / 'critic.pt').exists():
                agent.critic.load_state_dict(torch.load(model_dir / 'critic.pt', map_location=self.device))
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        agent.eval()
        return agent


def run_fair_comparison():
    """运行公平对比实验"""
    print("="*80)
    print("公平对比评估：使用统一的物理指标")
    print("="*80)
    
    evaluator = FairEvaluator(device='cpu')
    results_dir = Path('results')
    
    # 测试数据路径
    test_data_path = "processed_data/strict/alice/test_rl.pkl.gz"
    
    # 要评估的模型配置
    models_to_evaluate = [
        # 从实验5的baseline对比中选择模型
        {
            'name': 'DDPG',
            'type': 'DDPG',
            'dir': None,  # DDPG没有保存检查点
            'mode': 'simple'
        },
        {
            'name': 'PPO',
            'type': 'PPO',
            'dir': None,  # PPO没有保存检查点
            'mode': 'simple'
        },
        {
            'name': 'DR3L (λ=0.0)',
            'type': 'DR3L',
            'dir': results_dir / 'strict' / 'lambda_0.0',
            'mode': 'multiscale'
        },
        {
            'name': 'DR3L (λ=0.5)',
            'type': 'DR3L',
            'dir': results_dir / 'strict' / 'lambda_0.5',
            'mode': 'multiscale'
        },
        {
            'name': 'DR3L (λ=1.0)',
            'type': 'DR3L',
            'dir': results_dir / 'strict' / 'lambda_1.0',
            'mode': 'multiscale'
        },
    ]
    
    fair_results = {}
    
    for model_config in models_to_evaluate:
        print(f"\n{'='*80}")
        print(f"评估: {model_config['name']}")
        print(f"{'='*80}")
        
        try:
            # 创建环境
            env = make_env(test_data_path, mode=model_config['mode'])
            
            # 加载或创建模型
            if model_config['dir'] is not None and model_config['dir'].exists():
                print(f"  加载模型: {model_config['dir']}")
                agent = evaluator.load_trained_model(
                    model_config['type'],
                    model_config['dir']
                )
            else:
                print(f"  创建新模型（未训练）: {model_config['type']}")
                if model_config['type'] == 'DDPG':
                    agent = DDPG(state_dim=10, device='cpu')
                elif model_config['type'] == 'PPO':
                    agent = PPOAgent(state_dim=10, device='cpu')
                elif model_config['type'] == 'DR3L':
                    agent = DR3L(state_dim=10, device='cpu')
            
            # 公平评估
            metrics = evaluator.evaluate_agent_fair(agent, env, n_episodes=50)
            fair_results[model_config['name']] = metrics
            
            # 打印结果
            print(f"\n  结果:")
            print(f"    纯经济收益: {metrics['base_reward_mean']:.2f} ± {metrics['base_reward_std']:.2f}")
            print(f"    能量缺口CVaR: {metrics['cvar_energy_gap']:.4f}")
            print(f"    平均能量缺口: {metrics['mean_energy_gap']:.4f}")
            print(f"    约束违反率: {metrics['violation_rates']['total_violation_rate']:.2%}")
            
        except Exception as e:
            print(f"  ❌ 评估失败: {e}")
            import traceback
            traceback.print_exc()
            fair_results[model_config['name']] = {'error': str(e)}
    
    # 保存结果
    output_file = results_dir / 'fair_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(fair_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ 公平对比完成！结果已保存: {output_file}")
    print(f"{'='*80}")
    
    # 生成对比表
    generate_comparison_table(fair_results, results_dir)
    
    return fair_results


def generate_comparison_table(results, results_dir):
    """生成公平对比表"""
    print("\n" + "="*80)
    print("公平对比表")
    print("="*80)
    
    table_md = """# 公平对比评估结果

**评估说明**: 使用统一的物理指标，不含任何算法内部的penalty

---

## 对比表

| 算法 | 纯经济收益 | 标准差 | CVaR能量缺口 | 平均能量缺口 | 约束违反率 |
|------|-----------|--------|-------------|-------------|-----------|
"""
    
    for name, metrics in results.items():
        if 'error' not in metrics:
            table_md += f"| {name} | "
            table_md += f"{metrics['base_reward_mean']:.2f} | "
            table_md += f"{metrics['base_reward_std']:.2f} | "
            table_md += f"{metrics['cvar_energy_gap']:.4f} | "
            table_md += f"{metrics['mean_energy_gap']:.4f} | "
            table_md += f"{metrics['violation_rates']['total_violation_rate']:.2%} |\n"
    
    table_md += """
---

## 指标说明

1. **纯经济收益**: 不含任何penalty的基础收益（售电收入-购电成本）
2. **标准差**: 收益的波动性
3. **CVaR能量缺口**: 最差10%情况下的平均能量缺口
4. **平均能量缺口**: 所有时刻的平均能量缺口
5. **约束违反率**: SoC、功率、爬坡约束的总违反率

---

**生成时间**: 2026-02-20
"""
    
    output_file = results_dir / 'FAIR_COMPARISON_TABLE.md'
    with open(output_file, 'w') as f:
        f.write(table_md)
    
    print(f"✅ 对比表已保存: {output_file}")
    print("\n" + table_md)


if __name__ == "__main__":
    results = run_fair_comparison()
