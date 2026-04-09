"""
Quick test script to verify installation and basic functionality
"""

import torch
import numpy as np
from pathlib import Path

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    try:
        import torch
        import numpy
        import pandas
        import matplotlib
        import gym
        import tqdm
        print("✓ All packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  PyTorch version: {torch.__version__}")
        return True
    else:
        print("✗ CUDA not available (will use CPU)")
        return False

def test_models():
    """Test model instantiation"""
    print("\nTesting models...")
    try:
        from models import MultiScaleFusionNet, QuantileCritic, GaussianActor
        
        # Test feature net
        feature_net = MultiScaleFusionNet()
        short_term = torch.randn(2, 6, 12)
        long_term = torch.randn(2, 288, 2)
        features, _ = feature_net(short_term, long_term)
        assert features.shape == (2, 256), f"Expected (2, 256), got {features.shape}"
        print("✓ MultiScaleFusionNet works")
        
        # Test critic
        critic = QuantileCritic()
        quantiles = critic(features)
        assert quantiles.shape == (2, 51), f"Expected (2, 51), got {quantiles.shape}"
        print("✓ QuantileCritic works")
        
        # Test actor
        actor = GaussianActor()
        mean, std = actor(features)
        assert mean.shape == (2, 1), f"Expected (2, 1), got {mean.shape}"
        print("✓ GaussianActor works")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_algorithms():
    """Test algorithm instantiation"""
    print("\nTesting algorithms...")
    try:
        from algorithms import DDPG, PPOAgent, DR3L
        
        # Test DDPG
        ddpg = DDPG(state_dim=10, device='cpu')
        state = np.random.randn(10)
        action = ddpg.select_action(state, noise=0.1)
        assert action.shape == (1,), f"Expected (1,), got {action.shape}"
        print("✓ DDPG works")
        
        # Test PPO
        ppo = PPOAgent(state_dim=10, device='cpu')
        action, log_prob = ppo.select_action(state)
        assert action.shape == (1,), f"Expected (1,), got {action.shape}"
        print("✓ PPOAgent works")
        
        # Test DR3L
        dr3l = DR3L(state_dim=10, device='cpu')
        short_term = np.random.randn(6, 12)
        long_term = np.random.randn(288, 2)
        action, log_prob = dr3l.select_action(short_term, long_term)
        assert action.shape == (1,), f"Expected (1,), got {action.shape}"
        print("✓ DR3L works")
        
        return True
    except Exception as e:
        print(f"✗ Algorithm test failed: {e}")
        return False

def test_data_structure():
    """Test data directory structure"""
    print("\nTesting data structure...")
    
    data_root = Path("Dataset/DKASC")
    if not data_root.exists():
        print(f"✗ Data directory not found: {data_root}")
        print("  Please ensure DKASC data is in Dataset/DKASC/")
        return False
    
    alice_path = data_root / "Alice Springs"
    if not alice_path.exists():
        print(f"✗ Alice Springs data not found: {alice_path}")
        return False
    
    weather_file = alice_path / "101-Site_DKA-WeatherStation.csv"
    if not weather_file.exists():
        print(f"✗ Weather station file not found: {weather_file}")
        return False
    
    print(f"✓ Data structure looks good")
    print(f"  Found: {len(list(alice_path.glob('*.csv')))} CSV files in Alice Springs")
    
    return True

def test_preprocessing():
    """Test preprocessing script"""
    print("\nTesting preprocessing...")
    try:
        from preprocess_dkasc_data import DKASCPreprocessor
        
        preprocessor = DKASCPreprocessor()
        print("✓ Preprocessor instantiated")
        print("  Note: Run 'python preprocess_dkasc_data.py strict' to preprocess data")
        
        return True
    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")
        return False

def main():
    print("="*80)
    print("DR3L Quick Test")
    print("="*80)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("CUDA", test_cuda()))
    results.append(("Models", test_models()))
    results.append(("Algorithms", test_algorithms()))
    results.append(("Data Structure", test_data_structure()))
    results.append(("Preprocessing", test_preprocessing()))
    
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("All tests passed! You're ready to run experiments.")
        print("\nNext steps:")
        print("1. Preprocess data: python preprocess_dkasc_data.py strict")
        print("2. Run experiments: python run_experiments.py --experiment all")
    else:
        print("Some tests failed. Please check the errors above.")
        print("\nCommon issues:")
        print("- Missing packages: pip install -r requirements.txt")
        print("- Missing data: Ensure DKASC data is in Dataset/DKASC/")
        print("- CUDA issues: Check NVIDIA driver and CUDA installation")
    print("="*80)

if __name__ == "__main__":
    main()
