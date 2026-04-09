"""
测试新的违约惩罚奖励函数
Test the new violation penalty reward function
"""
import numpy as np
import sys
sys.path.append('/home/zhi/Risk')
from config import Config
from modules.reward import RiskAwareReward


def test_violation_penalty():
    """测试违约惩罚功能"""
    print("=" * 80)
    print("测试违约惩罚奖励函数")
    print("=" * 80)
    
    # 创建奖励函数（使用新的违约惩罚参数）
    reward_func = RiskAwareReward(
        lambda_violation=1.0,  # 违约惩罚权重
        lambda_ramp_violation=0.5  # ramp违约惩罚权重
    )
    
    print(f"\n配置参数:")
    print(f"  违约率阈值: {Config.VIOLATION_THRESHOLD * 100:.1f}%")
    print(f"  Ramp违约率阈值: {Config.RAMP_VIOLATION_THRESHOLD * 100:.1f}%")
    print(f"  违约惩罚权重 (λ_1): {reward_func.lambda_violation}")
    print(f"  Ramp违约惩罚权重 (λ_2): {reward_func.lambda_ramp_violation}")
    
    # 模拟一个 episode
    np.random.seed(42)
    n_steps = 288  # 24小时，每步5分钟
    
    print(f"\n开始模拟 {n_steps} 步...")
    
    total_rewards = []
    violation_counts = 0
    ramp_violation_counts = 0
    
    for t in range(n_steps):
        # 模拟数据
        hour = t // 12  # 5分钟间隔，12步=1小时
        
        # 光伏输出（白天高，夜晚低）
        p_pv_actual = np.random.uniform(0, 2.0) * (1 if 6 <= hour < 18 else 0.1)
        p_pv_forecast = p_pv_actual + np.random.normal(0, 0.2)
        
        # 储能功率（随机）
        p_battery = np.random.uniform(-0.5, 0.5)
        p_battery_prev = p_battery + np.random.normal(0, 0.1)
        
        # 负荷（基于时间的正弦模式）
        p_load = 0.8 + 0.3 * np.sin(2 * np.pi * hour / 24)
        
        # SoC（模拟一些越界情况）
        if t % 50 == 0:  # 每50步故意制造一次越界
            soc = np.random.choice([0.05, 0.95])  # 越界
            violation_counts += 1
        else:
            soc = np.clip(0.5 + np.random.normal(0, 0.1), Config.BESS_SOC_MIN, Config.BESS_SOC_MAX)
        
        # 计算奖励
        reward, info = reward_func.compute_total_reward(
            p_pv_actual, p_pv_forecast, p_battery, p_battery_prev,
            p_load, soc, hour
        )
        
        total_rewards.append(reward)
        
        # 每小时打印一次
        if t % 12 == 0 and t > 0:
            print(f"\n  步骤 {t:3d} (小时 {hour:2d}):")
            print(f"    总奖励: {reward:7.2f}")
            print(f"    基础收益: {info['reward/base']:7.2f}")
            print(f"    CVaR惩罚: {info['reward/cvar_penalty']:7.2f}")
            print(f"    违约惩罚: {info['reward/violation_penalty']:7.2f}")
            print(f"    Ramp违约惩罚: {info['reward/ramp_violation_penalty']:7.2f}")
            print(f"    违约率: {info['risk/violation_rate']*100:5.2f}%")
            print(f"    Ramp违约率: {info['risk/ramp_violation_rate']*100:5.2f}%")
    
    # 打印最终统计
    print("\n" + "=" * 80)
    print("最终统计信息:")
    print("=" * 80)
    
    stats = reward_func.get_statistics()
    print(f"\n奖励组成:")
    print(f"  总奖励:           {stats['total_reward']:12.2f}")
    print(f"  基础收益:         {stats['base_reward']:12.2f}")
    print(f"  CVaR惩罚:         {stats['cvar_penalty']:12.2f}")
    print(f"  极端偏差惩罚:     {stats['extreme_penalty']:12.2f}")
    print(f"  SoC惩罚:          {stats['soc_penalty']:12.2f}")
    print(f"  Ramp惩罚:         {stats['ramp_penalty']:12.2f}")
    print(f"  违约惩罚:         {stats['violation_penalty']:12.2f}  ← 新增")
    print(f"  Ramp违约惩罚:     {stats['ramp_violation_penalty']:12.2f}  ← 新增")
    
    print(f"\n违约统计:")
    print(f"  总违约次数:       {stats['num_violations']:12d}")
    print(f"  最终违约率:       {stats['violation_rate']*100:11.2f}%")
    print(f"  最终Ramp违约率:   {stats['ramp_violation_rate']*100:11.2f}%")
    print(f"  违约率阈值:       {Config.VIOLATION_THRESHOLD*100:11.2f}%")
    print(f"  Ramp违约率阈值:   {Config.RAMP_VIOLATION_THRESHOLD*100:11.2f}%")
    
    # 打印风险指标
    print("\n" + "=" * 80)
    print("风险指标:")
    print("=" * 80)
    risk_metrics = reward_func.get_risk_metrics()
    for key, value in risk_metrics.items():
        print(f"  {key:25s}: {value:12.4f}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    
    # 验证违约率控制效果
    final_violation_rate = stats['violation_rate']
    final_ramp_violation_rate = stats['ramp_violation_rate']
    
    print(f"\n验证结果:")
    if final_violation_rate <= Config.VIOLATION_THRESHOLD:
        print(f"  ✓ 违约率 {final_violation_rate*100:.2f}% <= {Config.VIOLATION_THRESHOLD*100:.2f}% (目标达成)")
    else:
        print(f"  ✗ 违约率 {final_violation_rate*100:.2f}% > {Config.VIOLATION_THRESHOLD*100:.2f}% (需要调整惩罚权重)")
    
    if final_ramp_violation_rate <= Config.RAMP_VIOLATION_THRESHOLD:
        print(f"  ✓ Ramp违约率 {final_ramp_violation_rate*100:.2f}% <= {Config.RAMP_VIOLATION_THRESHOLD*100:.2f}% (目标达成)")
    else:
        print(f"  ✗ Ramp违约率 {final_ramp_violation_rate*100:.2f}% > {Config.RAMP_VIOLATION_THRESHOLD*100:.2f}% (需要调整惩罚权重)")


def test_hyperparameter_tuning():
    """测试不同超参数对违约率的影响"""
    print("\n" + "=" * 80)
    print("超参数调优测试")
    print("=" * 80)
    
    # 测试不同的惩罚权重
    lambda_values = [0.5, 1.0, 2.0, 5.0]
    
    print(f"\n测试不同的违约惩罚权重 (λ_1):")
    print(f"{'λ_1':>6} | {'违约率':>8} | {'Ramp违约率':>12} | {'平均奖励':>10}")
    print("-" * 45)
    
    for lambda_1 in lambda_values:
        reward_func = RiskAwareReward(
            lambda_violation=lambda_1,
            lambda_ramp_violation=0.5
        )
        
        # 快速模拟
        np.random.seed(42)
        rewards = []
        for t in range(100):
            hour = t // 12
            p_pv_actual = np.random.uniform(0, 2.0) * (1 if 6 <= hour < 18 else 0.1)
            p_pv_forecast = p_pv_actual + np.random.normal(0, 0.2)
            p_battery = np.random.uniform(-0.5, 0.5)
            p_battery_prev = p_battery + np.random.normal(0, 0.1)
            p_load = 0.8 + 0.3 * np.sin(2 * np.pi * hour / 24)
            soc = 0.05 if t % 50 == 0 else np.clip(0.5 + np.random.normal(0, 0.1), 0.1, 0.9)
            
            reward, info = reward_func.compute_total_reward(
                p_pv_actual, p_pv_forecast, p_battery, p_battery_prev,
                p_load, soc, hour
            )
            rewards.append(reward)
        
        stats = reward_func.get_statistics()
        violation_rate = stats['violation_rate']
        ramp_violation_rate = stats['ramp_violation_rate']
        avg_reward = np.mean(rewards)
        
        print(f"{lambda_1:6.1f} | {violation_rate*100:7.2f}% | {ramp_violation_rate*100:11.2f}% | {avg_reward:10.2f}")
    
    print("\n建议:")
    print("  - 如果违约率过高，增大 λ_1")
    print("  - 如果ramp违约率过高，增大 λ_2")
    print("  - 权衡经济收益和违约率控制")


if __name__ == "__main__":
    test_violation_penalty()
    test_hyperparameter_tuning()
