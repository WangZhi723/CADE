#!/usr/bin/env python3
"""
IEEE TSG Experiments Runner - Multi-process Parallel Version
支持单GPU多进程并行运行实验
"""

import numpy as np
import torch
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
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
from algorithms import DDPG, PPOAgent, DR3L

# 注意: 单GPU多进程时，PyTorch会自动管理GPU内存
# 不需要额外的锁机制


def run_single_experiment(exp_config: Dict, results_queue: Queue):
    """
    在单独进程中运行单个实验
    
    Args:
        exp_config: 实验配置字典
        results_queue: 结果队列
    """
    exp_name = exp_config['name']
    exp_type = exp_config['type']
    data_mode = exp_config.get('data_mode', 'strict')
    seed = exp_config.get('seed', 42)
    use_wandb = exp_config.get('use_wandb', False)
    wandb_project = exp_config.get('wandb_project', 'dr3l-pv-bess')
    wandb_entity = exp_config.get('wandb_entity', None)
    
    # 设置进程特定的随机种子
    process_seed = seed + hash(exp_name) % 1000
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    
    # 检查CUDA可用性（不需要锁，只是检查）
    if torch.cuda.is_available():
        # 注意：在单GPU多进程情况下，多个进程会共享同一个GPU
        # PyTorch会自动管理GPU内存，但需要注意内存限制
        device = 'cuda'
        torch.cuda.manual_seed(process_seed)
        # 设置当前GPU（单GPU情况下都是0）
        torch.cuda.set_device(0)
    else:
        device = 'cpu'
    
    print(f"[进程 {os.getpid()}] 开始运行: {exp_name} (设备: {device})")
    
    try:
        # 导入ExperimentRunner（需要在这里导入以避免序列化问题）
        from run_experiments import ExperimentRunner
        
        # 创建实验运行器
        runner = ExperimentRunner(
            results_dir="results",
            seed=seed,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity
        )
        
        # 运行对应的实验
        if exp_type == 'lambda':
            results = runner.experiment_1_lambda_tradeoff(data_mode)
            result_file = f"results/{data_mode}/exp1_lambda_tradeoff.json"
        elif exp_type == 'robust':
            results = runner.experiment_2_robust_comparison(data_mode)
            result_file = f"results/{data_mode}/exp2_robust_comparison.json"
        elif exp_type == 'shift':
            results = runner.experiment_3_distribution_shift()
            result_file = "results/exp3_distribution_shift.json"
        elif exp_type == 'quality':
            results = runner.experiment_4_data_quality()
            result_file = "results/exp4_data_quality.json"
        elif exp_type == 'baselines':
            results = runner.experiment_5_baselines(data_mode)
            result_file = f"results/{data_mode}/exp5_baselines.json"
        else:
            raise ValueError(f"Unknown experiment type: {exp_type}")
        
        print(f"[进程 {os.getpid()}] ✅ 完成: {exp_name}")
        
        # 将结果放入队列
        results_queue.put({
            'name': exp_name,
            'type': exp_type,
            'data_mode': data_mode,
            'status': 'success',
            'result_file': result_file,
            'results': results
        })
        
    except Exception as e:
        print(f"[进程 {os.getpid()}] ❌ 失败: {exp_name} - {str(e)}")
        import traceback
        traceback.print_exc()
        
        results_queue.put({
            'name': exp_name,
            'type': exp_type,
            'data_mode': data_mode,
            'status': 'error',
            'error': str(e)
        })


def main_parallel(max_workers: int = 2, use_wandb: bool = False, 
                  wandb_project: str = "dr3l-pv-bess", wandb_entity: Optional[str] = None):
    """
    并行运行所有剩余实验
    
    Args:
        max_workers: 最大并行进程数
        use_wandb: 是否使用wandb
        wandb_project: wandb项目名
        wandb_entity: wandb实体名
    """
    print("=" * 80)
    print("并行运行所有剩余实验")
    print("=" * 80)
    print(f"最大并行进程数: {max_workers}")
    print(f"使用设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("")
    
    # 定义所有需要运行的实验
    experiments = []
    
    # 实验1: Lambda Tradeoff
    for mode in ['strict', 'light', 'raw']:
        file_path = Path(f"results/{mode}/exp1_lambda_tradeoff.json")
        need_run = False
        
        if not file_path.exists():
            need_run = True
        else:
            # 检查是否完整（需要5个lambda值都有test结果）
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                expected_lambdas = ['lambda_0.0', 'lambda_0.1', 'lambda_0.5', 'lambda_1.0', 'lambda_2.0']
                has_all = all(lam in data for lam in expected_lambdas)
                has_all_tests = all(lam in data and 'test' in data[lam] for lam in expected_lambdas)
                if not has_all or not has_all_tests:
                    need_run = True
            except:
                need_run = True
        
        if need_run:
            experiments.append({
                'name': f'实验1_Lambda_Tradeoff_{mode}',
                'type': 'lambda',
                'data_mode': mode,
                'seed': 42,
                'use_wandb': use_wandb,
                'wandb_project': wandb_project,
                'wandb_entity': wandb_entity
            })
    
    # 实验2: Robust Comparison
    for mode in ['strict', 'light', 'raw']:
        file_path = Path(f"results/{mode}/exp2_robust_comparison.json")
        need_run = False
        
        if not file_path.exists():
            need_run = True
        else:
            # 检查是否完整（需要3个rho值都有test结果）
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                expected_rhos = ['rho_0.0', 'rho_0.01', 'rho_0.05']
                has_all = all(rho in data for rho in expected_rhos)
                has_all_tests = all(rho in data and 'test' in data[rho] for rho in expected_rhos)
                if not has_all or not has_all_tests:
                    need_run = True
            except:
                need_run = True
        
        if need_run:
            experiments.append({
                'name': f'实验2_Robust_Comparison_{mode}',
                'type': 'robust',
                'data_mode': mode,
                'seed': 42,
                'use_wandb': use_wandb,
                'wandb_project': wandb_project,
                'wandb_entity': wandb_entity
            })
    
    # 实验3: Distribution Shift (检查Yulara测试是否完成)
    exp3_file = Path("results/exp3_distribution_shift.json")
    if exp3_file.exists():
        with open(exp3_file, 'r') as f:
            exp3_data = json.load(f)
            if not (exp3_data.get('yulara_test') and len(exp3_data.get('yulara_test', {})) > 0):
                experiments.append({
                    'name': '实验3_Distribution_Shift',
                    'type': 'shift',
                    'data_mode': 'strict',
                    'seed': 42,
                    'use_wandb': use_wandb,
                    'wandb_project': wandb_project,
                    'wandb_entity': wandb_entity
                })
    else:
        experiments.append({
            'name': '实验3_Distribution_Shift',
            'type': 'shift',
            'data_mode': 'strict',
            'seed': 42,
            'use_wandb': use_wandb,
            'wandb_project': wandb_project,
            'wandb_entity': wandb_entity
        })
    
    # 实验5: Baselines
    for mode in ['strict', 'light', 'raw']:
        file_path = Path(f"results/{mode}/exp5_baselines.json")
        need_run = False
        
        if not file_path.exists():
            need_run = True
        else:
            # 检查是否完整（需要4个agent都有test结果）
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                expected_agents = ['DDPG', 'PPO', 'DR3L_rho0', 'DR3L_full']
                has_all = all(agent in data for agent in expected_agents)
                has_all_tests = all(agent in data and 'test' in data[agent] for agent in expected_agents)
                if not has_all or not has_all_tests:
                    need_run = True
            except:
                need_run = True
        
        if need_run:
            experiments.append({
                'name': f'实验5_Baselines_{mode}',
                'type': 'baselines',
                'data_mode': mode,
                'seed': 42,
                'use_wandb': use_wandb,
                'wandb_project': wandb_project,
                'wandb_entity': wandb_entity
            })
    
    if len(experiments) == 0:
        print("✅ 所有实验已完成，无需运行")
        return
    
    print(f"待运行实验数: {len(experiments)}")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['name']}")
    print("")
    
    # 创建Manager和共享资源
    manager = Manager()
    results_queue = manager.Queue()
    
    # 启动进程池
    processes = []
    active_experiments = experiments.copy()
    
    print("开始并行运行实验...")
    print("=" * 80)
    
    start_time = time.time()
    completed = 0
    failed = 0
    
    # 使用进程池模式：保持max_workers个进程运行
    while active_experiments or processes:
        # 启动新进程（如果还有待运行实验且进程数未满）
        while len(processes) < max_workers and active_experiments:
            exp_config = active_experiments.pop(0)
            p = Process(
                target=run_single_experiment,
                args=(exp_config, results_queue)
            )
            p.start()
            processes.append((p, exp_config['name']))
            print(f"[主进程] 启动: {exp_config['name']} (PID: {p.pid})")
        
        # 检查是否有进程完成
        for i, (p, exp_name) in enumerate(processes):
            if not p.is_alive():
                p.join()  # 确保进程完全结束
                processes.pop(i)
                
                # 获取结果
                try:
                    result = results_queue.get_nowait()
                    if result['status'] == 'success':
                        completed += 1
                        print(f"[主进程] ✅ 完成: {result['name']}")
                    else:
                        failed += 1
                        print(f"[主进程] ❌ 失败: {result['name']} - {result.get('error', 'Unknown error')}")
                except:
                    pass
                
                break
        
        # 短暂休眠，避免CPU占用过高
        time.sleep(0.1)
    
    # 等待所有进程完成
    for p, exp_name in processes:
        p.join()
        try:
            result = results_queue.get_nowait()
            if result['status'] == 'success':
                completed += 1
                print(f"[主进程] ✅ 完成: {result['name']}")
            else:
                failed += 1
                print(f"[主进程] ❌ 失败: {result['name']} - {result.get('error', 'Unknown error')}")
        except:
            pass
    
    elapsed_time = time.time() - start_time
    
    print("")
    print("=" * 80)
    print("所有实验运行完成！")
    print("=" * 80)
    print(f"总计: {len(experiments)} 个实验")
    print(f"成功: {completed} 个")
    print(f"失败: {failed} 个")
    print(f"耗时: {elapsed_time/3600:.2f} 小时 ({elapsed_time/60:.1f} 分钟)")
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='并行运行所有剩余实验')
    parser.add_argument('--max_workers', type=int, default=2,
                       help='最大并行进程数（默认: 2）')
    parser.add_argument('--use_wandb', action='store_true',
                       help='启用WandB日志')
    parser.add_argument('--wandb_project', type=str, default='dr3l-pv-bess',
                       help='WandB项目名')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='WandB实体名（可选）')
    
    args = parser.parse_args()
    
    # 设置multiprocessing启动方法
    mp.set_start_method('spawn', force=True)
    
    main_parallel(
        max_workers=args.max_workers,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity
    )
