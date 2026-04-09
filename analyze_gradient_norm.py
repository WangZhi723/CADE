#!/usr/bin/env python3
"""
分析梯度L2 Norm的脚本
用于诊断梯度爆炸和约束惩罚的影响
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

def load_wandb_data(run_dir: str) -> Dict:
    """从WandB run目录加载数据"""
    run_path = Path(run_dir)
    
    # 查找wandb run目录
    wandb_dirs = list(Path('./wandb').glob('run-*'))
    if not wandb_dirs:
        print("❌ 未找到WandB run目录")
        return {}
    
    # 使用最新的run
    latest_run = sorted(wandb_dirs)[-1]
    print(f"✅ 找到最新run: {latest_run.name}")
    
    # 尝试从wandb-metadata.json读取
    metadata_file = latest_run / 'wandb-metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            print(f"   Run ID: {metadata.get('run_id', 'N/A')}")
            print(f"   项目: {metadata.get('project', 'N/A')}")
    
    return {'run_dir': latest_run}

def analyze_gradient_norm_from_json(json_file: str = 'results/strict/exp5b_dr3l_phased_wconst_5.json'):
    """从JSON结果文件中分析梯度norm（如果有的话）"""
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 文件不存在: {json_file}")
        return
    
    print("\n" + "="*70)
    print("梯度L2 Norm分析")
    print("="*70)
    
    for algo in ['DDPG', 'PPO', 'DR3L_rho0', 'DR3L_full']:
        if algo not in data:
            continue
        
        print(f"\n【{algo}】")
        print("-" * 70)
        
        # 检查是否有梯度norm数据
        if 'train' in data[algo]:
            train_data = data[algo]['train']
            
            # 检查是否有梯度norm字段
            has_grad_norm = any(k.endswith('_grad_norm') for k in train_data.keys())
            
            if has_grad_norm:
                for key in ['critic_grad_norm', 'actor_grad_norm']:
                    if key in train_data:
                        norms = train_data[key]
                        if isinstance(norms, list) and len(norms) > 0:
                            norms = np.array(norms)
                            print(f"\n{key}:")
                            print(f"  平均值: {np.mean(norms):.4f}")
                            print(f"  标准差: {np.std(norms):.4f}")
                            print(f"  最小值: {np.min(norms):.4f}")
                            print(f"  最大值: {np.max(norms):.4f}")
                            print(f"  中位数: {np.median(norms):.4f}")
                            
                            # 统计超过阈值的比例
                            over_threshold = np.sum(norms > 1.0) / len(norms) * 100
                            print(f"  超过1.0的比例: {over_threshold:.1f}%")
                            
                            # 检测梯度爆炸
                            if np.max(norms) > 5.0:
                                print(f"  ⚠️  警告: 检测到梯度爆炸（最大值{np.max(norms):.2f}）")
                            elif np.max(norms) > 3.0:
                                print(f"  ⚠️  注意: 梯度较大（最大值{np.max(norms):.2f}）")
                            else:
                                print(f"  ✅ 梯度正常")
            else:
                print("  ℹ️  本次训练未记录梯度norm数据")
                print("     请重新运行训练以获取梯度norm信息")

def print_gradient_norm_guide():
    """打印梯度norm诊断指南"""
    
    print("\n" + "="*70)
    print("梯度L2 Norm诊断指南")
    print("="*70)
    
    guide = """
【正常范围】
- Critic梯度norm: 0.3 - 2.0
- Actor梯度norm: 0.2 - 1.5
- 超过1.0的比例: < 50%

【异常信号】
1. 梯度norm > 5.0
   → 梯度爆炸，需要降低惩罚幅度或增加梯度裁剪阈值

2. 梯度norm < 0.1
   → 梯度过小，可能学习率太低或梯度裁剪过严

3. 梯度norm在阶段2突然飙升
   → 约束惩罚幅度过大，需要渐进式退火

4. 梯度norm方差很大（std > mean）
   → 训练不稳定，需要增加batch size或调整学习率

【优化建议】
- 如果梯度norm > 3.0频繁出现：
  ✓ 降低w_const（3.0 → 2.0）
  ✓ 降低const_intent_flat（1.5 → 1.0）
  ✓ 增加梯度裁剪阈值（1.0 → 1.5）

- 如果梯度norm < 0.2：
  ✓ 增加梯度裁剪阈值（1.0 → 2.0）
  ✓ 增加学习率（3e-4 → 5e-4）
  ✓ 检查loss是否为0

【WandB查看方法】
1. 打开WandB项目: https://wandb.ai/your-entity/dr3l-pv-bess
2. 进入最新run
3. 在Charts中搜索: critic_grad_norm 或 actor_grad_norm
4. 查看梯度随episode的变化趋势
"""
    
    print(guide)

def main():
    print("\n" + "="*70)
    print("DR3L梯度L2 Norm分析工具")
    print("="*70)
    
    # 加载WandB数据
    wandb_info = load_wandb_data('./wandb')
    
    # 分析JSON结果
    analyze_gradient_norm_from_json()
    
    # 打印诊断指南
    print_gradient_norm_guide()
    
    print("\n" + "="*70)
    print("💡 提示: 运行以下命令查看完整的WandB数据:")
    print("   wandb sync ./wandb/run-*/logs/")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
