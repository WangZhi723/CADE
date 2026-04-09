"""
示例：如何使用新的违约惩罚奖励函数

本示例展示如何在训练代码中使用新增的违约惩罚项
"""
import sys
sys.path.append('/home/zhi/Risk')
from config import Config
from modules.reward import RiskAwareReward


def example_basic_usage():
    """示例1: 基本使用"""
    print("=" * 80)
    print("示例1: 基本使用")
    print("=" * 80)
    
    # 创建奖励函数（使用默认参数）
    reward_func = RiskAwareReward()
    
    print(f"\n默认配置:")
    print(f"  违约惩罚权重 (λ_1): {reward_func.lambda_violation}")
    print(f"  Ramp违约惩罚权重 (λ_2): {reward_func.lambda_ramp_violation}")
    print(f"  违约率阈值: {Config.VIOLATION_THRESHOLD * 100:.1f}%")
    print(f"  Ramp违约率阈值: {Config.RAMP_VIOLATION_THRESHOLD * 100:.1f}%")
    
    # 模拟一个时间步
    p_pv_actual = 1.5  # MW
    p_pv_forecast = 1.4  # MW
    p_battery = 0.3  # MW (放电)
    p_battery_prev = 0.2  # MW
    p_load = 1.2  # MW
    soc = 0.6  # 60%
    hour = 10  # 10点
    
    # 计算奖励
    reward, info = reward_func.compute_total_reward(
        p_pv_actual=p_pv_actual,
        p_pv_forecast=p_pv_forecast,
        p_battery=p_battery,
        p_battery_prev=p_battery_prev,
        p_load=p_load,
        soc=soc,
        hour=hour
    )
    
    print(f"\n奖励计算结果:")
    print(f"  总奖励: {reward:.2f}")
    print(f"  基础收益: {info['reward/base']:.2f}")
    print(f"  CVaR惩罚: {info['reward/cvar_penalty']:.2f}")
    print(f"  违约惩罚: {info['reward/violation_penalty']:.2f}")
    print(f"  Ramp违约惩罚: {info['reward/ramp_violation_penalty']:.2f}")
    print(f"  违约率: {info['risk/violation_rate']*100:.2f}%")
    print(f"  Ramp违约率: {info['risk/ramp_violation_rate']*100:.2f}%")


def example_custom_parameters():
    """示例2: 自定义参数"""
    print("\n" + "=" * 80)
    print("示例2: 自定义参数")
    print("=" * 80)
    
    # 创建奖励函数（自定义参数）
    reward_func = RiskAwareReward(
        cvar_alpha=0.05,  # CVaR置信水平
        cvar_window=100,  # CVaR窗口大小
        lambda_cvar=1.0,  # CVaR惩罚权重
        lambda_extreme=0.5,  # 极端偏差惩罚权重
        lambda_violation=2.0,  # 违约惩罚权重（增大以更严格控制违约率）
        lambda_ramp_violation=1.0  # Ramp违约惩罚权重（增大以更严格控制ramp违约率）
    )
    
    print(f"\n自定义配置:")
    print(f"  违约惩罚权重 (λ_1): {reward_func.lambda_violation}")
    print(f"  Ramp违约惩罚权重 (λ_2): {reward_func.lambda_ramp_violation}")
    print(f"  → 更严格的违约率控制")


def example_training_loop():
    """示例3: 在训练循环中使用"""
    print("\n" + "=" * 80)
    print("示例3: 在训练循环中使用")
    print("=" * 80)
    
    # 创建奖励函数
    reward_func = RiskAwareReward(
        lambda_violation=1.0,
        lambda_ramp_violation=0.5
    )
    
    print(f"\n模拟训练循环...")
    
    # 模拟一个episode
    n_steps = 50
    episode_reward = 0
    
    for step in range(n_steps):
        # 模拟状态和动作（实际训练中从环境获取）
        p_pv_actual = 1.0 + 0.5 * (step / n_steps)
        p_pv_forecast = p_pv_actual + 0.1
        p_battery = 0.2
        p_battery_prev = 0.15
        p_load = 1.2
        
        # 模拟一些违约情况
        if step % 20 == 0:
            soc = 0.05  # 违约（低于0.1）
        else:
            soc = 0.5
        
        hour = (step // 12) % 24
        
        # 计算奖励
        reward, info = reward_func.compute_total_reward(
            p_pv_actual, p_pv_forecast, p_battery, p_battery_prev,
            p_load, soc, hour
        )
        
        episode_reward += reward
        
        # 每10步打印一次
        if step % 10 == 0:
            print(f"  Step {step:2d}: "
                  f"Reward={reward:6.2f}, "
                  f"Violation Rate={info['risk/violation_rate']*100:5.2f}%, "
                  f"Ramp Violation Rate={info['risk/ramp_violation_rate']*100:5.2f}%")
    
    print(f"\nEpisode总奖励: {episode_reward:.2f}")
    
    # 获取统计信息
    stats = reward_func.get_statistics()
    print(f"\nEpisode统计:")
    print(f"  违约率: {stats['violation_rate']*100:.2f}%")
    print(f"  Ramp违约率: {stats['ramp_violation_rate']*100:.2f}%")
    print(f"  违约惩罚: {stats['violation_penalty']:.2f}")
    print(f"  Ramp违约惩罚: {stats['ramp_violation_penalty']:.2f}")


def example_hyperparameter_comparison():
    """示例4: 超参数对比"""
    print("\n" + "=" * 80)
    print("示例4: 超参数对比")
    print("=" * 80)
    
    # 测试不同的λ_1值
    lambda_values = [0.5, 1.0, 2.0, 5.0]
    
    print(f"\n对比不同的违约惩罚权重 (λ_1):")
    print(f"{'λ_1':>6} | {'平均奖励':>10} | {'违约率':>8} | {'违约惩罚':>10}")
    print("-" * 45)
    
    for lambda_1 in lambda_values:
        # 创建奖励函数
        reward_func = RiskAwareReward(
            lambda_violation=lambda_1,
            lambda_ramp_violation=0.5
        )
        
        # 模拟20步
        rewards = []
        for step in range(20):
            p_pv_actual = 1.5
            p_pv_forecast = 1.4
            p_battery = 0.3
            p_battery_prev = 0.2
            p_load = 1.2
            soc = 0.05 if step % 10 == 0 else 0.5  # 每10步违约一次
            hour = 10
            
            reward, info = reward_func.compute_total_reward(
                p_pv_actual, p_pv_forecast, p_battery, p_battery_prev,
                p_load, soc, hour
            )
            rewards.append(reward)
        
        # 获取统计
        stats = reward_func.get_statistics()
        avg_reward = sum(rewards) / len(rewards)
        violation_rate = stats['violation_rate']
        violation_penalty = stats['violation_penalty']
        
        print(f"{lambda_1:6.1f} | {avg_reward:10.2f} | {violation_rate*100:7.2f}% | {violation_penalty:10.2f}")
    
    print(f"\n观察:")
    print(f"  - λ_1越大，违约惩罚越重")
    print(f"  - 平均奖励可能降低（因为惩罚增加）")
    print(f"  - 但违约率会得到更好的控制")


def example_monitoring():
    """示例5: 监控违约率"""
    print("\n" + "=" * 80)
    print("示例5: 监控违约率")
    print("=" * 80)
    
    reward_func = RiskAwareReward(
        lambda_violation=1.0,
        lambda_ramp_violation=0.5
    )
    
    print(f"\n模拟训练过程中的违约率变化...")
    print(f"{'Step':>5} | {'违约率':>8} | {'Ramp违约率':>12} | {'状态':>10}")
    print("-" * 45)
    
    # 模拟100步
    for step in range(0, 100, 10):
        # 模拟违约率逐渐降低的过程
        violation_prob = max(0.0, 0.1 - step * 0.001)  # 从10%降到0%
        
        # 模拟10步
        for _ in range(10):
            p_pv_actual = 1.5
            p_pv_forecast = 1.4
            p_battery = 0.3
            p_battery_prev = 0.2
            p_load = 1.2
            
            # 根据概率决定是否违约
            import random
            if random.random() < violation_prob:
                soc = 0.05  # 违约
            else:
                soc = 0.5  # 正常
            
            hour = 10
            
            reward, info = reward_func.compute_total_reward(
                p_pv_actual, p_pv_forecast, p_battery, p_battery_prev,
                p_load, soc, hour
            )
        
        # 获取当前违约率
        violation_rate = info['risk/violation_rate']
        ramp_violation_rate = info['risk/ramp_violation_rate']
        
        # 判断状态
        if violation_rate <= Config.VIOLATION_THRESHOLD:
            status = "✓ 达标"
        else:
            status = "✗ 超标"
        
        print(f"{step:5d} | {violation_rate*100:7.2f}% | {ramp_violation_rate*100:11.2f}% | {status}")
    
    print(f"\n目标:")
    print(f"  违约率 ≤ {Config.VIOLATION_THRESHOLD*100:.1f}%")
    print(f"  Ramp违约率 ≤ {Config.RAMP_VIOLATION_THRESHOLD*100:.1f}%")


if __name__ == "__main__":
    # 运行所有示例
    example_basic_usage()
    example_custom_parameters()
    example_training_loop()
    example_hyperparameter_comparison()
    example_monitoring()
    
    print("\n" + "=" * 80)
    print("所有示例完成！")
    print("=" * 80)
    print(f"\n下一步:")
    print(f"  1. 查看 VIOLATION_PENALTY_GUIDE.md 了解详细使用方法")
    print(f"  2. 在实际训练代码中应用新的奖励函数")
    print(f"  3. 根据训练效果调整超参数 λ_1 和 λ_2")
    print(f"  4. 监控违约率是否满足要求")
