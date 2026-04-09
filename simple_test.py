"""
简单测试：验证违约惩罚奖励函数的核心逻辑
不依赖其他模块，可以直接运行
"""

def compute_reward(economic_return, violation, ramp_violation, lambda_1=1.0, lambda_2=0.5):
    """
    计算包括违约惩罚项的奖励函数
    
    Args:
        economic_return: 当前经济回报
        violation: 当前违约率（0-1之间）
        ramp_violation: 当前 ramp 违约率（0-1之间）
        lambda_1: 违约惩罚项的权重
        lambda_2: ramp 违约惩罚项的权重
    
    Returns:
        Reward: 计算后的奖励
    """
    # 确保违约率不超过 1%
    violation_penalty = max(violation - 0.01, 0) * 1000
    
    # 控制 ramp violation 不超过 5%
    ramp_violation_penalty = max(ramp_violation - 0.05, 0) * 1000
    
    # 总奖励
    reward = economic_return - lambda_1 * violation_penalty - lambda_2 * ramp_violation_penalty
    
    return reward


def test_violation_penalty():
    """测试违约惩罚功能"""
    print("=" * 80)
    print("违约惩罚奖励函数测试")
    print("=" * 80)
    
    # 测试场景
    scenarios = [
        {
            "name": "场景1: 无违约",
            "economic_return": 100,
            "violation": 0.005,  # 0.5% < 1%
            "ramp_violation": 0.03,  # 3% < 5%
            "lambda_1": 1.0,
            "lambda_2": 0.5
        },
        {
            "name": "场景2: 轻微违约",
            "economic_return": 100,
            "violation": 0.015,  # 1.5% > 1%
            "ramp_violation": 0.06,  # 6% > 5%
            "lambda_1": 1.0,
            "lambda_2": 0.5
        },
        {
            "name": "场景3: 严重违约",
            "economic_return": 100,
            "violation": 0.05,  # 5% > 1%
            "ramp_violation": 0.10,  # 10% > 5%
            "lambda_1": 1.0,
            "lambda_2": 0.5
        },
        {
            "name": "场景4: 增大惩罚权重",
            "economic_return": 100,
            "violation": 0.02,  # 2% > 1%
            "ramp_violation": 0.08,  # 8% > 5%
            "lambda_1": 2.0,  # 增大
            "lambda_2": 1.0   # 增大
        }
    ]
    
    print(f"\n{'场景':<20} | {'经济收益':>8} | {'违约率':>8} | {'Ramp违约率':>12} | {'最终奖励':>10}")
    print("-" * 75)
    
    for scenario in scenarios:
        reward = compute_reward(
            economic_return=scenario["economic_return"],
            violation=scenario["violation"],
            ramp_violation=scenario["ramp_violation"],
            lambda_1=scenario["lambda_1"],
            lambda_2=scenario["lambda_2"]
        )
        
        print(f"{scenario['name']:<20} | {scenario['economic_return']:8.2f} | "
              f"{scenario['violation']*100:7.2f}% | {scenario['ramp_violation']*100:11.2f}% | "
              f"{reward:10.2f}")
    
    print("\n" + "=" * 80)
    print("观察:")
    print("  1. 场景1: 违约率都低于阈值，无惩罚，奖励 = 经济收益")
    print("  2. 场景2: 违约率超过阈值，有惩罚，奖励下降")
    print("  3. 场景3: 违约率更高，惩罚更重，奖励大幅下降")
    print("  4. 场景4: 增大惩罚权重，相同违约率下惩罚更重")
    print("=" * 80)


def demo_training_process():
    """演示训练过程中违约率的变化"""
    print("\n" + "=" * 80)
    print("模拟训练过程")
    print("=" * 80)
    
    print(f"\n假设训练过程中违约率逐渐降低...")
    print(f"{'Episode':>8} | {'违约率':>8} | {'Ramp违约率':>12} | {'经济收益':>10} | {'奖励':>10} | {'状态':>10}")
    print("-" * 75)
    
    lambda_1 = 1.0
    lambda_2 = 0.5
    
    # 模拟训练过程：违约率逐渐降低
    episodes = [0, 100, 200, 300, 400, 500]
    violation_rates = [0.10, 0.08, 0.05, 0.03, 0.015, 0.008]  # 从10%降到0.8%
    ramp_violation_rates = [0.15, 0.12, 0.09, 0.07, 0.06, 0.04]  # 从15%降到4%
    economic_returns = [80, 85, 90, 92, 95, 95]  # 经济收益逐渐提高
    
    for i, episode in enumerate(episodes):
        violation = violation_rates[i]
        ramp_violation = ramp_violation_rates[i]
        economic_return = economic_returns[i]
        
        reward = compute_reward(economic_return, violation, ramp_violation, lambda_1, lambda_2)
        
        # 判断状态
        if violation <= 0.01 and ramp_violation <= 0.05:
            status = "✓ 达标"
        else:
            status = "✗ 未达标"
        
        print(f"{episode:8d} | {violation*100:7.2f}% | {ramp_violation*100:11.2f}% | "
              f"{economic_return:10.2f} | {reward:10.2f} | {status}")
    
    print("\n" + "=" * 80)
    print("结论:")
    print("  - 训练初期: 违约率高，惩罚重，总奖励低")
    print("  - 训练中期: 违约率下降，惩罚减少，总奖励提高")
    print("  - 训练后期: 违约率达标，惩罚很小，总奖励接近经济收益")
    print("=" * 80)


def demo_hyperparameter_tuning():
    """演示超参数调优"""
    print("\n" + "=" * 80)
    print("超参数调优示例")
    print("=" * 80)
    
    print(f"\n固定违约率，测试不同的 λ_1 值:")
    print(f"{'λ_1':>6} | {'λ_2':>6} | {'违约率':>8} | {'Ramp违约率':>12} | {'奖励':>10}")
    print("-" * 55)
    
    economic_return = 100
    violation = 0.02  # 2%
    ramp_violation = 0.08  # 8%
    
    lambda_pairs = [(0.5, 0.25), (1.0, 0.5), (2.0, 1.0), (5.0, 2.0)]
    
    for lambda_1, lambda_2 in lambda_pairs:
        reward = compute_reward(economic_return, violation, ramp_violation, lambda_1, lambda_2)
        print(f"{lambda_1:6.1f} | {lambda_2:6.2f} | {violation*100:7.2f}% | "
              f"{ramp_violation*100:11.2f}% | {reward:10.2f}")
    
    print("\n" + "=" * 80)
    print("建议:")
    print("  - 如果违约率过高 (> 1%)，增大 λ_1")
    print("  - 如果ramp违约率过高 (> 5%)，增大 λ_2")
    print("  - 如果经济收益过低，适当降低 λ_1 和 λ_2")
    print("  - 推荐起始值: λ_1 = 1.0, λ_2 = 0.5")
    print("=" * 80)


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "违约惩罚奖励函数 - 简单测试" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # 运行测试
    test_violation_penalty()
    demo_training_process()
    demo_hyperparameter_tuning()
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    print("\n核心公式:")
    print("  reward = economic_return - λ_1 * max(violation - 0.01, 0) * 1000")
    print("                           - λ_2 * max(ramp_violation - 0.05, 0) * 1000")
    print("\n参数说明:")
    print("  - economic_return: 经济收益")
    print("  - violation: 违约率（目标 < 1% = 0.01）")
    print("  - ramp_violation: ramp违约率（目标 < 5% = 0.05）")
    print("  - λ_1: 违约惩罚权重（默认 1.0）")
    print("  - λ_2: ramp违约惩罚权重（默认 0.5）")
    print("\n下一步:")
    print("  1. 在实际训练代码中应用这个奖励函数")
    print("  2. 监控训练过程中的违约率")
    print("  3. 根据违约率调整 λ_1 和 λ_2")
    print("=" * 80)
    print()
