#!/usr/bin/env python3
"""
只生成Yulara的RL样本文件
从现有的test.pkl.gz生成test_rl.pkl.gz
"""

import sys
import gzip
import pickle
from pathlib import Path
import pandas as pd

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from preprocess_dkasc_data_v2 import DKASCPreprocessor

def generate_yulara_rl_samples(mode='strict'):
    """
    为Yulara生成RL样本文件
    
    Args:
        mode: 数据模式 ('strict', 'light', 'raw')
    """
    print("=" * 80)
    print(f"生成Yulara RL样本 - {mode.upper()}模式")
    print("=" * 80)
    
    # 检查输入文件
    test_file = Path(f"processed_data/{mode}/yulara/test.pkl.gz")
    output_file = Path(f"processed_data/{mode}/yulara/test_rl.pkl.gz")
    
    if not test_file.exists():
        print(f"❌ 错误: 输入文件不存在: {test_file}")
        print(f"   请先运行预处理: python preprocess_dkasc_data_v2.py {mode}")
        return False
    
    if output_file.exists():
        print(f"⚠️  输出文件已存在: {output_file}")
        response = input("是否覆盖? (y/n): ")
        if response.lower() != 'y':
            print("已取消")
            return False
    
    # 加载test数据
    print(f"\n加载数据: {test_file}")
    with gzip.open(test_file, 'rb') as f:
        df = pickle.load(f)
    
    print(f"  数据形状: {df.shape}")
    print(f"  列: {list(df.columns)[:10]}...")
    
    # 检查归一化参数
    norm_params_file = Path(f"processed_data/strict/alice/norm_params.pkl")
    if not norm_params_file.exists():
        print(f"❌ 错误: 归一化参数文件不存在: {norm_params_file}")
        print(f"   请先运行STRICT模式预处理生成归一化参数")
        return False
    
    # 创建预处理器实例
    preprocessor = DKASCPreprocessor()
    
    # 检查数据是否已经归一化（如果列值在[0,1]范围内，可能已经归一化）
    # 这里假设数据已经经过完整预处理，包括归一化
    # 如果数据未归一化，需要先归一化
    
    # 检查是否需要归一化
    numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
    exclude_cols = ['hour', 'day_of_year', 'hour_sin', 'hour_cos', 
                   'day_sin', 'day_cos', 'wind_dir_sin', 'wind_dir_cos']
    cols_to_check = [c for c in numeric_cols if c not in exclude_cols]
    
    if len(cols_to_check) > 0:
        sample_col = cols_to_check[0]
        max_val = df[sample_col].max()
        min_val = df[sample_col].min()
        
        # 如果值不在[0,1]范围内，可能需要归一化
        if max_val > 1.1 or min_val < -0.1:
            print(f"\n⚠️  数据可能未归一化 (范围: [{min_val:.2f}, {max_val:.2f}])")
            print("   尝试加载归一化参数并归一化...")
            
            with open(norm_params_file, 'rb') as f:
                norm_params = pickle.load(f)
            
            # 归一化数据
            df, _ = preprocessor.normalize(df, norm_params, fit=False)
            print("   ✅ 数据已归一化")
    
    # 生成RL样本
    print(f"\n生成RL样本...")
    print(f"  窗口大小: {preprocessor.window_size}")
    print(f"  预测范围: {preprocessor.forecast_horizon}")
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 生成RL样本
    num_samples = preprocessor.build_rl_samples_chunked(
        df, 
        output_file, 
        chunk_size=5000,
        streaming=False
    )
    
    if num_samples > 0:
        file_size = output_file.stat().st_size / 1024 / 1024
        print(f"\n✅ 成功生成RL样本!")
        print(f"   文件: {output_file}")
        print(f"   样本数: {num_samples:,}")
        print(f"   文件大小: {file_size:.1f} MB")
        return True
    else:
        print(f"\n❌ 生成RL样本失败")
        return False

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else 'strict'
    
    if mode not in ['strict', 'light', 'raw']:
        print(f"❌ 错误: 无效的模式 '{mode}'")
        print("   使用: python generate_yulara_rl_samples.py [strict|light|raw]")
        sys.exit(1)
    
    success = generate_yulara_rl_samples(mode)
    sys.exit(0 if success else 1)
