"""
Verification script for DKASC preprocessing
Checks data integrity, statistics, and format
"""

import pickle
import gzip
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_compressed_pickle(file_path: Path):
    """Load gzipped pickle file"""
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)


def verify_dataframe(df: pd.DataFrame, split_name: str, mode: str):
    """Verify DataFrame integrity"""
    print(f"\n  {split_name.upper()}:")
    print(f"    Samples: {len(df):,}")
    print(f"    Features: {len(df.columns)}")
    print(f"    Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    # Check for NaN
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"    ⚠️  WARNING: {nan_count} NaN values found!")
    else:
        print(f"    ✅ No NaN values")
    
    # Check timestamp
    if 'timestamp' in df.columns:
        print(f"    Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Check for duplicates
        dup_count = df['timestamp'].duplicated().sum()
        if dup_count > 0:
            print(f"    ⚠️  WARNING: {dup_count} duplicate timestamps!")
        else:
            print(f"    ✅ No duplicate timestamps")
    
    # Check physical constraints
    if 'Global_Horizontal_Radiation' in df.columns:
        neg_ghi = (df['Global_Horizontal_Radiation'] < 0).sum()
        if neg_ghi > 0:
            print(f"    ⚠️  WARNING: {neg_ghi} negative GHI values!")
        else:
            print(f"    ✅ GHI >= 0")
    
    if 'Wind_Speed' in df.columns:
        neg_wind = (df['Wind_Speed'] < 0).sum()
        if neg_wind > 0:
            print(f"    ⚠️  WARNING: {neg_wind} negative Wind Speed values!")
        else:
            print(f"    ✅ Wind Speed >= 0")
    
    if 'pv_power_total' in df.columns:
        neg_pv = (df['pv_power_total'] < 0).sum()
        if neg_pv > 0:
            print(f"    ⚠️  WARNING: {neg_pv} negative PV power values!")
        else:
            print(f"    ✅ PV Power >= 0")
        
        print(f"    PV power range: [{df['pv_power_total'].min():.2f}, {df['pv_power_total'].max():.2f}]")


def verify_rl_samples(samples: List[Dict], split_name: str):
    """Verify RL samples format"""
    print(f"\n  {split_name.upper()} RL:")
    print(f"    Samples: {len(samples):,}")
    
    if len(samples) > 0:
        sample = samples[0]
        
        # Check keys
        required_keys = ['short_term', 'long_term', 'current_state', 'pv_actual', 'pv_forecast', 'timestamp']
        missing_keys = [k for k in required_keys if k not in sample]
        if missing_keys:
            print(f"    ⚠️  WARNING: Missing keys: {missing_keys}")
        else:
            print(f"    ✅ All required keys present")
        
        # Check shapes
        print(f"    short_term shape: {sample['short_term'].shape} (expected: (12, 6))")
        print(f"    long_term shape: {sample['long_term'].shape} (expected: (288, 2))")
        print(f"    current_state shape: {sample['current_state'].shape} (expected: (6,))")
        
        # Check for NaN
        short_nan = np.isnan(sample['short_term']).sum()
        long_nan = np.isnan(sample['long_term']).sum()
        curr_nan = np.isnan(sample['current_state']).sum()
        
        if short_nan > 0 or long_nan > 0 or curr_nan > 0:
            print(f"    ⚠️  WARNING: NaN in samples (short={short_nan}, long={long_nan}, curr={curr_nan})")
        else:
            print(f"    ✅ No NaN in samples")


def verify_normalization(norm_params: Dict):
    """Verify normalization parameters"""
    print(f"\n  Normalization Parameters:")
    print(f"    Features: {len(norm_params)}")
    
    # Check for invalid ranges
    invalid = []
    for col, params in norm_params.items():
        if params['max'] <= params['min']:
            invalid.append(col)
    
    if invalid:
        print(f"    ⚠️  WARNING: Invalid ranges for: {invalid}")
    else:
        print(f"    ✅ All ranges valid (max > min)")
    
    # Show sample parameters
    print(f"\n    Sample parameters:")
    for col in list(norm_params.keys())[:5]:
        params = norm_params[col]
        print(f"      {col:40s}: [{params['min']:.4f}, {params['max']:.4f}]")


def verify_mode(mode: str, site: str):
    """Verify all files for a given mode and site"""
    print(f"\n{'='*80}")
    print(f"VERIFYING: {mode.upper()} - {site.upper()}")
    print(f"{'='*80}")
    
    base_dir = Path(f"processed_data/{mode}/{site}")
    
    if not base_dir.exists():
        print(f"❌ Directory not found: {base_dir}")
        return False
    
    # Check normalization parameters (only for strict/alice)
    if mode == "strict" and site == "alice":
        norm_file = base_dir / "norm_params.pkl"
        if norm_file.exists():
            print(f"\n✅ Found: {norm_file}")
            norm_params = load_compressed_pickle(norm_file) if norm_file.suffix == '.gz' else pickle.load(open(norm_file, 'rb'))
            verify_normalization(norm_params)
        else:
            print(f"❌ Missing: {norm_file}")
    
    # Check DataFrames
    print(f"\nDataFrames:")
    for split in ['train', 'val', 'test']:
        df_file = base_dir / f"{split}.pkl.gz"
        if df_file.exists():
            print(f"\n✅ Found: {df_file} ({df_file.stat().st_size / 1024 / 1024:.1f} MB)")
            df = load_compressed_pickle(df_file)
            verify_dataframe(df, split, mode)
        else:
            if site == "alice" or split == "test":
                print(f"⚠️  Not found: {df_file}")
    
    # Check RL samples
    print(f"\nRL Samples:")
    for split in ['train', 'val', 'test']:
        rl_file = base_dir / f"{split}_rl.pkl.gz"
        if rl_file.exists():
            print(f"\n✅ Found: {rl_file} ({rl_file.stat().st_size / 1024 / 1024:.1f} MB)")
            samples = load_compressed_pickle(rl_file)
            verify_rl_samples(samples, split)
        else:
            if site == "alice" or split == "test":
                print(f"⚠️  Not found: {rl_file}")
    
    # Check stats
    stats_file = base_dir / "stats.txt"
    if stats_file.exists():
        print(f"\n✅ Found: {stats_file}")
        print(f"\n{'='*80}")
        print(f"STATISTICS FILE CONTENT:")
        print(f"{'='*80}")
        with open(stats_file, 'r') as f:
            print(f.read())
    else:
        print(f"❌ Missing: {stats_file}")
    
    return True


def main():
    """Main verification function"""
    print(f"\n{'='*80}")
    print(f"DKASC PREPROCESSING VERIFICATION")
    print(f"{'='*80}")
    
    modes = ['strict', 'light', 'raw']
    sites = ['alice', 'yulara']
    
    results = {}
    
    for mode in modes:
        for site in sites:
            key = f"{mode}_{site}"
            results[key] = verify_mode(mode, site)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*80}\n")
    
    for mode in modes:
        print(f"{mode.upper()}:")
        for site in sites:
            key = f"{mode}_{site}"
            status = "✅ PASS" if results.get(key, False) else "❌ FAIL"
            print(f"  {site:10s}: {status}")
        print()
    
    # Overall status
    all_passed = all(results.values())
    if all_passed:
        print("✅ ALL VERIFICATIONS PASSED")
    else:
        print("⚠️  SOME VERIFICATIONS FAILED")
        print("\nMissing datasets:")
        for key, passed in results.items():
            if not passed:
                print(f"  - {key}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
