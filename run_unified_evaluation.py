"""
运行统一评估：对所有已训练模型使用相同的评估标准
确保公平对比，解决"目标函数不一致"问题
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict
import sys

# 添加evaluation模块到路径
sys.path.insert(0, str(Path(__file__).parent))

from pv_env import make_env
from algorithms import DDPG, PPOAgent, DR3L
from evaluation.unified_evaluator import evaluate_policy


class UnifiedEvaluationRunner:
    """统一评估运行器"""
    
    def __init__(self, results_dir: str = "results", device: str = 'cpu'):
        self.results_dir = Path(results_dir)
        self.device = device
    
    def load_agent(self, agent_type: str, model_dir: Path, state_dim: int = 14):
        """加载训练好的智能体（DDPG/PPO 的 state_dim 须与当前 make_env simple 观测维一致）。"""
        print(f"  加载模型: {model_dir}")
        
        if agent_type == 'DDPG':
            agent = DDPG(state_dim=state_dim, device=self.device)
            if (model_dir / 'actor.pt').exists():
                agent.actor.load_state_dict(
                    torch.load(model_dir / 'actor.pt', map_location=self.device)
                )
        
        elif agent_type == 'PPO':
            agent = PPOAgent(state_dim=state_dim, device=self.device)
            if (model_dir / 'actor.pt').exists():
                agent.actor.load_state_dict(
                    torch.load(model_dir / 'actor.pt', map_location=self.device)
                )
        
        elif agent_type == 'DR3L':
            agent = DR3L(state_dim=state_dim, device=self.device)
            if (model_dir / 'feature_net.pt').exists():
                agent.feature_net.load_state_dict(
                    torch.load(model_dir / 'feature_net.pt', map_location=self.device)
                )
            if (model_dir / 'actor.pt').exists():
                agent.actor.load_state_dict(
                    torch.load(model_dir / 'actor.pt', map_location=self.device)
                )
            if (model_dir / 'critic.pt').exists():
                agent.critic.load_state_dict(
                    torch.load(model_dir / 'critic.pt', map_location=self.device)
                )
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # 将模型设置为评估模式（只对PyTorch模型调用）
        if agent_type == 'DDPG':
            if hasattr(agent, 'actor'):
                agent.actor.eval()
        elif agent_type == 'PPO':
            if hasattr(agent, 'actor'):
                agent.actor.eval()
        elif agent_type == 'DR3L':
            # DR3L可能有多个网络
            if hasattr(agent, 'feature_net'):
                agent.feature_net.eval()
            if hasattr(agent, 'actor'):
                agent.actor.eval()
            if hasattr(agent, 'critic'):
                agent.critic.eval()
        
        return agent
    
    def run_unified_evaluation(self, data_mode: str = 'strict'):
        """
        对指定数据模式的所有模型运行统一评估
        
        Args:
            data_mode: 'strict', 'light', 或 'raw'
        """
        print("="*80)
        print(f"统一评估 - {data_mode.upper()} 模式")
        print("="*80)
        
        # 测试数据路径
        test_data_path = f"processed_data/{data_mode}/alice/test_rl.pkl.gz"
        
        # 要评估的模型配置
        models_to_evaluate = [
            # Lambda系列（实验1）
            {
                'name': 'DR3L_lambda_0.0',
                'type': 'DR3L',
                'dir': self.results_dir / data_mode / 'lambda_0.0',
                'env_mode': 'multiscale'
            },
            {
                'name': 'DR3L_lambda_0.1',
                'type': 'DR3L',
                'dir': self.results_dir / data_mode / 'lambda_0.1',
                'env_mode': 'multiscale'
            },
            {
                'name': 'DR3L_lambda_0.5',
                'type': 'DR3L',
                'dir': self.results_dir / data_mode / 'lambda_0.5',
                'env_mode': 'multiscale'
            },
            {
                'name': 'DR3L_lambda_1.0',
                'type': 'DR3L',
                'dir': self.results_dir / data_mode / 'lambda_1.0',
                'env_mode': 'multiscale'
            },
            {
                'name': 'DR3L_lambda_2.0',
                'type': 'DR3L',
                'dir': self.results_dir / data_mode / 'lambda_2.0',
                'env_mode': 'multiscale'
            },
        ]
        
        unified_results = {}
        
        for model_config in models_to_evaluate:
            print(f"\n{'='*80}")
            print(f"评估: {model_config['name']}")
            print(f"{'='*80}")
            
            try:
                # 检查模型是否存在
                if not model_config['dir'].exists():
                    print(f"  ⚠️  模型不存在，跳过: {model_config['dir']}")
                    continue
                
                # 创建环境
                env = make_env(test_data_path, mode=model_config['env_mode'])
                if model_config['type'] in ('DDPG', 'PPO'):
                    sd = int(np.prod(env.observation_space.shape))
                else:
                    sd = 14
                agent = self.load_agent(
                    model_config['type'],
                    model_config['dir'],
                    state_dim=sd,
                )
                
                # 统一评估（不使用训练reward）
                metrics = evaluate_policy(
                    agent,
                    env,
                    agent_type=model_config['type'],
                    n_episodes=50
                )
                
                unified_results[model_config['name']] = metrics
                
            except Exception as e:
                print(f"  ❌ 评估失败: {e}")
                import traceback
                traceback.print_exc()
                unified_results[model_config['name']] = {'error': str(e)}
        
        # 保存结果
        output_file = self.results_dir / f'unified_metrics_{data_mode}.json'
        with open(output_file, 'w') as f:
            json.dump(unified_results, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✅ 统一评估完成！")
        print(f"结果已保存: {output_file}")
        print(f"{'='*80}")
        
        # 生成对比表
        self.generate_comparison_table(unified_results, data_mode)
        
        return unified_results
    
    def generate_comparison_table(self, results: Dict, data_mode: str):
        """生成统一评估对比表"""
        print(f"\n{'='*80}")
        print(f"统一评估对比表 - {data_mode.upper()}")
        print(f"{'='*80}")
        
        # Markdown表格
        table_md = f"""# 统一评估对比表 - {data_mode.upper()} 模式

**评估说明**: 
- 使用统一的物理指标，不使用训练reward
- 所有算法使用相同的测试环境和评估标准
- 确保公平对比

---

## 对比表

| Method | Base Return | CVaR↓ | Max Loss↓ | Mean Gap↓ | Violation Rate↓ |
|--------|-------------|-------|-----------|-----------|-----------------|
"""
        
        for name, metrics in results.items():
            if 'error' not in metrics:
                table_md += f"| {name} | "
                table_md += f"{metrics.get('base_return', 0):.2f} | "
                table_md += f"{metrics.get('cvar_0.1', 0):.4f} | "
                table_md += f"{metrics.get('max_loss', 0):.4f} | "
                table_md += f"{metrics.get('mean_energy_gap', 0):.4f} | "
                table_md += f"{metrics.get('violation_rate', 0):.2%} |\n"
        
        table_md += """
---

## 指标说明

1. **Base Return**: 纯经济收益（不含任何penalty）
2. **CVaR**: 能量缺口的CVaR（最差10%），越低越好
3. **Max Loss**: 最大能量缺口，越低越好
4. **Mean Gap**: 平均能量缺口，越低越好
5. **Violation Rate**: 约束违反率，越低越好

---

## 关键发现

"""
        
        # 找出最佳配置
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if valid_results:
            # 最低CVaR
            best_cvar = min(valid_results.items(), key=lambda x: x[1].get('cvar_0.1', float('inf')))
            table_md += f"- **最低CVaR**: {best_cvar[0]} ({best_cvar[1]['cvar_0.1']:.4f})\n"
            
            # 最低Max Loss
            best_max_loss = min(valid_results.items(), key=lambda x: x[1].get('max_loss', float('inf')))
            table_md += f"- **最低Max Loss**: {best_max_loss[0]} ({best_max_loss[1]['max_loss']:.4f})\n"
            
            # 最低违反率
            best_violation = min(valid_results.items(), key=lambda x: x[1].get('violation_rate', float('inf')))
            table_md += f"- **最低违反率**: {best_violation[0]} ({best_violation[1]['violation_rate']:.2%})\n"
        
        table_md += f"""

---

**生成时间**: 2026-02-20  
**数据模式**: {data_mode.upper()}  
**评估方法**: 统一物理指标（与训练reward解耦）
"""
        
        # 保存表格
        output_file = self.results_dir / f'UNIFIED_COMPARISON_{data_mode.upper()}.md'
        with open(output_file, 'w') as f:
            f.write(table_md)
        
        print(f"✅ 对比表已保存: {output_file}")
        print("\n" + table_md)
    
    def run_all_modes(self):
        """运行所有数据模式的统一评估"""
        modes = ['strict', 'light', 'raw']
        all_results = {}
        
        for mode in modes:
            print(f"\n\n{'#'*80}")
            print(f"# 开始评估 {mode.upper()} 模式")
            print(f"{'#'*80}\n")
            
            try:
                results = self.run_unified_evaluation(mode)
                all_results[mode] = results
            except Exception as e:
                print(f"❌ {mode.upper()} 模式评估失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 保存汇总结果
        summary_file = self.results_dir / 'unified_metrics_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n\n{'#'*80}")
        print(f"# ✅ 所有模式评估完成")
        print(f"# 汇总结果: {summary_file}")
        print(f"{'#'*80}")
        
        return all_results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='统一评估：使用相同标准评估所有算法'
    )
    parser.add_argument('--mode', type=str, default='all',
                       choices=['strict', 'light', 'raw', 'all'],
                       help='数据模式')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='结果目录')
    
    args = parser.parse_args()
    
    runner = UnifiedEvaluationRunner(results_dir=args.results_dir)
    
    if args.mode == 'all':
        runner.run_all_modes()
    else:
        runner.run_unified_evaluation(args.mode)


if __name__ == "__main__":
    main()
