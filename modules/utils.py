"""
Utility Functions
工具函数：日志、可视化、评估等
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import torch
import sys
sys.path.append('/home/zhi/Risk')
from config import Config

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def set_seed(seed: int = Config.SEED):
    """
    设置随机种子（确保可复现性）
    
    Args:
        seed: 随机种子
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_config(config_path: str):
    """
    保存配置到 JSON 文件
    
    Args:
        config_path: 保存路径
    """
    config_dict = {}
    for key, value in vars(Config).items():
        # 跳过私有属性和方法
        if key.startswith('_'):
            continue
        
        # 跳过函数、方法、classmethod 等可调用对象
        if callable(value):
            continue
        
        # 检查是否是 classmethod、staticmethod 等
        if isinstance(value, (classmethod, staticmethod)):
            continue
        
        # 转换不可序列化的对象
        try:
            if isinstance(value, torch.device):
                value = str(value)
            # 测试是否可以序列化
            json.dumps(value)
            config_dict[key] = value
        except (TypeError, ValueError):
            # 如果无法序列化，转换为字符串
            config_dict[key] = str(value)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"配置已保存到: {config_path}")


def plot_training_curves(log_data: Dict[str, List], save_path: str = None):
    """
    绘制训练曲线
    
    Args:
        log_data: 日志数据字典 {'reward': [...], 'loss': [...], ...}
        save_path: 保存路径（如果为 None，则显示图像）
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 奖励曲线
    if 'reward' in log_data:
        axes[0, 0].plot(log_data['reward'], alpha=0.6)
        axes[0, 0].set_title('Episode Reward')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
    
    # CVaR 曲线
    if 'cvar' in log_data:
        axes[0, 1].plot(log_data['cvar'], alpha=0.6, color='orange')
        axes[0, 1].set_title('CVaR (Risk Metric)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('CVaR')
        axes[0, 1].grid(True)
    
    # 能量缺口
    if 'energy_gap' in log_data:
        axes[1, 0].plot(log_data['energy_gap'], alpha=0.6, color='red')
        axes[1, 0].set_title('Energy Gap')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Mean Energy Gap (MWh)')
        axes[1, 0].grid(True)
    
    # SoC 违反次数
    if 'soc_violations' in log_data:
        axes[1, 1].plot(log_data['soc_violations'], alpha=0.6, color='green')
        axes[1, 1].set_title('SoC Violations')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Number of Violations')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_episode_details(episode_data: Dict, save_path: str = None):
    """
    绘制单个 episode 的详细信息
    
    Args:
        episode_data: episode 数据字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    steps = range(len(episode_data.get('p_pv', [])))
    
    # 功率曲线
    axes[0].plot(steps, episode_data.get('p_pv', []), label='光伏输出', linewidth=2)
    axes[0].plot(steps, episode_data.get('p_load', []), label='负荷需求', linewidth=2)
    axes[0].plot(steps, episode_data.get('p_battery', []), label='储能功率', linewidth=2)
    axes[0].set_title('功率曲线 (MW)', fontsize=14)
    axes[0].set_xlabel('时间步（5分钟/步）')
    axes[0].set_ylabel('功率 (MW)')
    axes[0].legend()
    axes[0].grid(True)
    
    # SoC 曲线
    axes[1].plot(steps, episode_data.get('soc', []), label='SoC', color='blue', linewidth=2)
    axes[1].axhline(y=Config.BESS_SOC_MIN, color='red', linestyle='--', label='SoC 下限')
    axes[1].axhline(y=Config.BESS_SOC_MAX, color='red', linestyle='--', label='SoC 上限')
    axes[1].set_title('储能 SoC', fontsize=14)
    axes[1].set_xlabel('时间步（5分钟/步）')
    axes[1].set_ylabel('SoC')
    axes[1].legend()
    axes[1].grid(True)
    
    # 奖励曲线
    axes[2].plot(steps, episode_data.get('reward', []), label='即时奖励', linewidth=2)
    axes[2].set_title('即时奖励', fontsize=14)
    axes[2].set_xlabel('时间步（5分钟/步）')
    axes[2].set_ylabel('奖励')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Episode 详情已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_policy(env, model, n_episodes: int = 10) -> Dict:
    """
    评估策略性能
    
    Args:
        env: 环境
        model: 训练好的模型
        n_episodes: 评估 episode 数量
    
    Returns:
        评估结果字典
    """
    all_rewards = []
    all_energy_gaps = []
    all_soc_violations = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        
        # 修复：处理 Gym API 返回的 tuple (observation, info)
        if isinstance(obs, tuple):
            obs = obs[0]
        
        done = False
        episode_reward = 0
        episode_energy_gaps = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            
            # 修复：处理不同版本的 Gym API
            if len(step_result) == 4:
                # 旧版 Gym: (obs, reward, done, info)
                obs, reward, done, info = step_result
            else:
                # 新版 Gym: (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            # 处理 obs 可能是 tuple 的情况
            if isinstance(obs, tuple):
                obs = obs[0]
            
            # 处理 reward 可能是数组的情况
            if isinstance(reward, (list, np.ndarray)):
                reward = reward[0] if len(reward) > 0 else 0
            
            episode_reward += reward
            episode_energy_gaps.append(info.get('energy_gap', 0))
        
        all_rewards.append(episode_reward)
        all_energy_gaps.extend(episode_energy_gaps)
        
        # 尝试获取环境统计
        try:
            stats = env.get_episode_stats()
            all_soc_violations.append(stats.get('soc_violations', 0))
        except AttributeError:
            # 向量化环境可能没有这个方法
            all_soc_violations.append(0)
    
    # 计算风险指标
    energy_gaps = np.array(all_energy_gaps)
    if len(energy_gaps) > 0:
        cvar_005 = np.mean(np.sort(energy_gaps)[-max(1, int(0.05*len(energy_gaps))):])
    else:
        cvar_005 = 0.0
    
    results = {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_energy_gap': np.mean(energy_gaps) if len(energy_gaps) > 0 else 0.0,
        'max_energy_gap': np.max(energy_gaps) if len(energy_gaps) > 0 else 0.0,
        'cvar_005': cvar_005,
        'mean_soc_violations': np.mean(all_soc_violations),
        'total_soc_violations': np.sum(all_soc_violations)
    }
    
    return results


def print_evaluation_results(results: Dict):
    """
    打印评估结果
    
    Args:
        results: 评估结果字典
    """
    print("\n" + "=" * 80)
    print("评估结果:")
    print("=" * 80)
    for key, value in results.items():
        print(f"  {key:30s}: {value:12.4f}")
    print("=" * 80)


def create_result_dirs():
    """
    创建结果目录
    """
    dirs = [
        Config.LOG_DIR,
        Config.MODEL_DIR,
        Config.FIGURE_DIR,
        Config.TENSORBOARD_LOG
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("结果目录已创建")


def save_episode_trajectory(episode_data: Dict, save_path: str):
    """
    保存 episode 轨迹到 JSON
    
    Args:
        episode_data: episode 数据
        save_path: 保存路径
    """
    # 转换 numpy 数组为列表
    serializable_data = {}
    for key, value in episode_data.items():
        if isinstance(value, np.ndarray):
            serializable_data[key] = value.tolist()
        elif isinstance(value, list):
            serializable_data[key] = value
        else:
            serializable_data[key] = str(value)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    print(f"Episode 轨迹已保存到: {save_path}")


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("=" * 80)
    print("测试工具函数")
    print("=" * 80)
    
    # 测试设置种子
    set_seed(42)
    print("\n随机种子已设置")
    
    # 测试创建目录
    create_result_dirs()
    
    # 测试保存配置
    save_config(os.path.join(Config.LOG_DIR, 'config.json'))
    
    # 测试绘图（模拟数据）
    print("\n生成测试图表...")
    log_data = {
        'reward': np.cumsum(np.random.randn(100)),
        'cvar': np.abs(np.random.randn(100)) * 0.1,
        'energy_gap': np.abs(np.random.randn(100)) * 0.05,
        'soc_violations': np.random.randint(0, 5, 100)
    }
    
    plot_training_curves(
        log_data, 
        save_path=os.path.join(Config.FIGURE_DIR, 'test_training_curves.png')
    )
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
