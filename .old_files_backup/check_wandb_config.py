#!/usr/bin/env python3
"""
检查WandB配置，帮助用户找到正确的entity名称
"""
import subprocess
import sys

def check_wandb_config():
    print("=" * 60)
    print("WandB 配置检查工具")
    print("=" * 60)
    
    # 1. 检查是否登录
    print("\n1. 检查WandB登录状态...")
    try:
        result = subprocess.run(['wandb', 'whoami'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"   ✅ {result.stdout.strip()}")
            username = result.stdout.strip().split()[-1] if result.stdout.strip() else None
        else:
            print(f"   ❌ 未登录或wandb命令不可用")
            print(f"   错误: {result.stderr}")
            username = None
    except FileNotFoundError:
        print("   ❌ wandb命令未找到，请先安装: pip install wandb")
        username = None
    except Exception as e:
        print(f"   ❌ 检查失败: {e}")
        username = None
    
    # 2. 检查配置文件
    print("\n2. 检查WandB配置文件...")
    import os
    from pathlib import Path
    
    netrc_path = Path.home() / '.netrc'
    if netrc_path.exists():
        print(f"   ✅ 找到配置文件: {netrc_path}")
        try:
            with open(netrc_path, 'r') as f:
                content = f.read()
                if 'api.wandb.ai' in content:
                    print("   ✅ WandB API key已配置")
                else:
                    print("   ⚠️  未找到WandB API key")
        except Exception as e:
            print(f"   ⚠️  无法读取配置文件: {e}")
    else:
        print(f"   ⚠️  配置文件不存在: {netrc_path}")
        print("   请运行: wandb login")
    
    # 3. 建议
    print("\n" + "=" * 60)
    print("建议的修复方法:")
    print("=" * 60)
    
    if username:
        print(f"\n✅ 使用你的个人账户（推荐）:")
        print(f"   python run_experiments.py \\")
        print(f"       --data_mode strict \\")
        print(f"       --use_wandb \\")
        print(f"       --wandb_project PV-RL-Experiments")
        print(f"       # 不指定 --wandb_entity，使用默认账户: {username}")
    else:
        print(f"\n✅ 不指定entity（使用默认账户）:")
        print(f"   python run_experiments.py \\")
        print(f"       --data_mode strict \\")
        print(f"       --use_wandb \\")
        print(f"       --wandb_project PV-RL-Experiments")
        print(f"       # 不添加 --wandb_entity 参数")
    
    print(f"\n❌ 不要使用:")
    print(f"   --wandb_entity PV_train  # 这个entity不存在")
    
    print(f"\n💡 如果必须使用团队entity:")
    print(f"   1. 登录 https://wandb.ai")
    print(f"   2. 创建或加入团队")
    print(f"   3. 使用团队名称作为entity")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_wandb_config()
