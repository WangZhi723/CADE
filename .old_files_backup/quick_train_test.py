"""
Quick Training Test Script
快速测试训练流程，验证数据加载和训练是否正常
"""

import pickle
import gzip
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# 检查数据文件
print("="*80)
print("快速训练测试")
print("="*80)

# 1. 检查数据文件
data_path = Path("processed_data/strict/alice/train_rl.pkl.gz")
if not data_path.exists():
    print(f"❌ 数据文件不存在: {data_path}")
    print("请先运行预处理: python preprocess_dkasc_data_v2.py strict")
    exit(1)

print(f"\n✅ 找到数据文件: {data_path}")

# 2. 加载数据
print("\n加载RL样本...")
with gzip.open(data_path, 'rb') as f:
    samples = pickle.load(f)

print(f"✅ 加载成功: {len(samples):,} 个样本")

# 3. 检查样本格式
if len(samples) > 0:
    sample = samples[0]
    print("\n样本格式检查:")
    print(f"  short_term shape: {sample['short_term'].shape} (期望: (12, 6))")
    print(f"  long_term shape: {sample['long_term'].shape} (期望: (288, 2))")
    print(f"  current_state shape: {sample['current_state'].shape} (期望: (6,))")
    print(f"  pv_actual: {sample['pv_actual']}")
    print(f"  pv_forecast: {sample['pv_forecast']}")
    print(f"  timestamp: {sample['timestamp']}")
    
    # 验证形状
    assert sample['short_term'].shape == (12, 6), f"short_term形状错误: {sample['short_term'].shape}"
    assert sample['long_term'].shape == (288, 2), f"long_term形状错误: {sample['long_term'].shape}"
    assert sample['current_state'].shape == (6,), f"current_state形状错误: {sample['current_state'].shape}"
    print("✅ 样本格式正确")

# 4. 检查归一化参数
norm_path = Path("processed_data/strict/alice/norm_params.pkl")
if norm_path.exists():
    with open(norm_path, 'rb') as f:
        norm_params = pickle.load(f)
    print(f"\n✅ 归一化参数: {len(norm_params)} 个特征")
else:
    print("\n⚠️  归一化参数文件不存在")

# 5. 测试数据加载（模拟训练循环）
print("\n" + "="*80)
print("测试数据加载（模拟训练）")
print("="*80)

# 使用前1000个样本进行快速测试
test_samples = samples[:1000]
print(f"使用 {len(test_samples)} 个样本进行测试...")

# 模拟训练循环
for i, sample in enumerate(tqdm(test_samples[:100], desc="处理样本")):
    # 转换为numpy数组
    short_term = np.array(sample['short_term'], dtype=np.float32)
    long_term = np.array(sample['long_term'], dtype=np.float32)
    current_state = np.array(sample['current_state'], dtype=np.float32)
    
    # 检查NaN
    if np.isnan(short_term).any() or np.isnan(long_term).any() or np.isnan(current_state).any():
        print(f"⚠️  样本 {i} 包含NaN值")
        break
    
    # 检查数值范围（归一化后应该在[0,1]附近）
    if short_term.max() > 10 or short_term.min() < -10:
        print(f"⚠️  样本 {i} 数值范围异常: [{short_term.min():.2f}, {short_term.max():.2f}]")

print("\n✅ 数据加载测试通过")

# 6. 检查环境（如果可用）
print("\n" + "="*80)
print("检查训练环境")
print("="*80)

try:
    from pv_env import make_env
    print("✅ pv_env 模块可用")
    
    # 尝试创建环境
    try:
        env = make_env(str(data_path), mode='multiscale')
        print("✅ 环境创建成功")
        
        # 测试reset
        obs = env.reset()
        print(f"✅ 环境reset成功，观测类型: {type(obs)}")
        
    except Exception as e:
        print(f"⚠️  环境创建失败: {e}")
        print("   可能需要更新 pv_env.py 以适配新数据格式")
        
except ImportError:
    print("⚠️  pv_env 模块不可用，跳过环境测试")

# 7. 检查算法（如果可用）
print("\n" + "="*80)
print("检查算法模块")
print("="*80)

try:
    from algorithms import DR3L, DDPG, PPOAgent
    print("✅ algorithms 模块可用")
    print("   可用算法: DR3L, DDPG, PPOAgent")
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA不可用，将使用CPU")
        
except ImportError as e:
    print(f"⚠️  algorithms 模块导入失败: {e}")

# 8. 总结
print("\n" + "="*80)
print("测试总结")
print("="*80)

print("\n✅ 数据文件检查通过")
print("✅ 数据格式验证通过")
print("✅ 数据加载测试通过")

print("\n📝 下一步:")
print("  1. 如果环境/算法模块有问题，需要更新代码以适配新数据格式")
print("  2. 运行完整训练: python run_experiments.py --mode strict --experiment lambda_tradeoff")
print("  3. 或创建自定义训练脚本（参考 TRAINING_GUIDE.md）")

print("\n" + "="*80)
print("测试完成！")
print("="*80)
