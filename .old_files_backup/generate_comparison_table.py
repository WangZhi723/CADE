"""
Generate comparison table across all three modes
For inclusion in IEEE TSG paper
"""

import pickle
import gzip
from pathlib import Path
from typing import Dict


def load_stats_from_file(stats_file: Path) -> Dict:
    """Parse statistics from stats.txt file"""
    stats = {}
    
    with open(stats_file, 'r') as f:
        content = f.read()
        
        # Extract key metrics
        for line in content.split('\n'):
            if 'Number of PV systems' in line:
                stats['pv_systems'] = int(line.split(':')[1].strip())
            elif 'Peak PV capacity:' in line:
                stats['pv_capacity'] = float(line.split(':')[1].strip().replace('kW', '').strip())
            elif 'Initial samples (after merge):' in line:
                stats['initial_samples'] = int(line.split(':')[1].strip().replace(',', ''))
            elif 'Final samples:' in line:
                stats['final_samples'] = int(line.split(':')[1].strip().replace(',', ''))
            elif 'Retention rate:' in line:
                stats['retention_rate'] = float(line.split(':')[1].strip().replace('%', ''))
            elif 'Days removed:' in line and 'Completeness' in content:
                stats['days_removed'] = int(line.split(':')[1].strip())
            elif 'Negative GHI fixed:' in line:
                stats['ghi_fixed'] = int(line.split(':')[1].strip().replace(',', ''))
            elif 'Values clipped (3-sigma):' in line:
                stats['outliers_clipped'] = int(line.split(':')[1].strip().replace(',', ''))
    
    return stats


def generate_comparison_table():
    """Generate comparison table for all modes"""
    
    modes = ['strict', 'light', 'raw']
    site = 'alice'
    
    print("\n" + "="*80)
    print("DKASC PREPROCESSING - MODE COMPARISON TABLE")
    print("="*80 + "\n")
    
    # Collect statistics
    all_stats = {}
    for mode in modes:
        stats_file = Path(f"processed_data/{mode}/{site}/stats.txt")
        if stats_file.exists():
            all_stats[mode] = load_stats_from_file(stats_file)
        else:
            print(f"⚠️  Warning: {stats_file} not found. Run preprocessing first.")
            return
    
    # Table 1: Overall Statistics
    print("Table 1: Dataset Statistics by Mode")
    print("-" * 80)
    print(f"{'Metric':<40s} {'STRICT':>12s} {'LIGHT':>12s} {'RAW':>12s}")
    print("-" * 80)
    
    # PV systems (should be same for all)
    pv_sys = all_stats['strict'].get('pv_systems', 0)
    pv_cap = all_stats['strict'].get('pv_capacity', 0)
    print(f"{'PV Systems Aggregated':<40s} {pv_sys:>12d} {pv_sys:>12d} {pv_sys:>12d}")
    print(f"{'Peak PV Capacity (kW)':<40s} {pv_cap:>12.1f} {pv_cap:>12.1f} {pv_cap:>12.1f}")
    print()
    
    # Initial samples (should be same)
    init_s = all_stats['strict'].get('initial_samples', 0)
    init_l = all_stats['light'].get('initial_samples', 0)
    init_r = all_stats['raw'].get('initial_samples', 0)
    print(f"{'Initial Samples (after merge)':<40s} {init_s:>12,d} {init_l:>12,d} {init_r:>12,d}")
    
    # Final samples
    final_s = all_stats['strict'].get('final_samples', 0)
    final_l = all_stats['light'].get('final_samples', 0)
    final_r = all_stats['raw'].get('final_samples', 0)
    print(f"{'Final Samples':<40s} {final_s:>12,d} {final_l:>12,d} {final_r:>12,d}")
    
    # Retention rate
    ret_s = all_stats['strict'].get('retention_rate', 0)
    ret_l = all_stats['light'].get('retention_rate', 0)
    ret_r = all_stats['raw'].get('retention_rate', 0)
    print(f"{'Retention Rate (%)':<40s} {ret_s:>12.2f} {ret_l:>12.2f} {ret_r:>12.2f}")
    print()
    
    # Data removed
    removed_s = init_s - final_s
    removed_l = init_l - final_l
    removed_r = init_r - final_r
    print(f"{'Samples Removed':<40s} {removed_s:>12,d} {removed_l:>12,d} {removed_r:>12,d}")
    
    # Days removed
    days_s = all_stats['strict'].get('days_removed', 0)
    days_l = all_stats['light'].get('days_removed', 0)
    days_r = all_stats['raw'].get('days_removed', 0)
    print(f"{'Days Removed (completeness)':<40s} {days_s:>12d} {days_l:>12d} {days_r:>12d}")
    print("-" * 80 + "\n")
    
    # Table 2: Cleaning Operations
    print("Table 2: Cleaning Operations Applied")
    print("-" * 80)
    print(f"{'Operation':<40s} {'STRICT':>12s} {'LIGHT':>12s} {'RAW':>12s}")
    print("-" * 80)
    
    ghi_s = all_stats['strict'].get('ghi_fixed', 0)
    ghi_l = all_stats['light'].get('ghi_fixed', 0)
    ghi_r = all_stats['raw'].get('ghi_fixed', 0)
    print(f"{'Negative GHI Fixed':<40s} {ghi_s:>12,d} {ghi_l:>12,d} {ghi_r:>12,d}")
    
    outlier_s = all_stats['strict'].get('outliers_clipped', 0)
    outlier_l = all_stats['light'].get('outliers_clipped', 0)
    outlier_r = all_stats['raw'].get('outliers_clipped', 0)
    print(f"{'Outliers Clipped (3-sigma)':<40s} {outlier_s:>12,d} {outlier_l:>12,d} {outlier_r:>12,d}")
    
    print(f"{'Completeness Threshold':<40s} {'≥ 0.8':>12s} {'≥ 0.6':>12s} {'None':>12s}")
    print(f"{'Missing Value Imputation':<40s} {'Yes':>12s} {'Yes':>12s} {'No':>12s}")
    print("-" * 80 + "\n")
    
    # Load split statistics
    print("Table 3: Train/Val/Test Split Statistics")
    print("-" * 80)
    print(f"{'Split':<15s} {'Mode':<10s} {'Samples':>15s} {'RL Samples':>15s}")
    print("-" * 80)
    
    for mode in modes:
        for split in ['train', 'val', 'test']:
            # Load DataFrame
            df_file = Path(f"processed_data/{mode}/{site}/{split}.pkl.gz")
            rl_file = Path(f"processed_data/{mode}/{site}/{split}_rl.pkl.gz")
            
            df_samples = 0
            rl_samples = 0
            
            if df_file.exists():
                with gzip.open(df_file, 'rb') as f:
                    df = pickle.load(f)
                    df_samples = len(df)
            
            if rl_file.exists():
                with gzip.open(rl_file, 'rb') as f:
                    rl = pickle.load(f)
                    rl_samples = len(rl)
            
            print(f"{split.upper():<15s} {mode.upper():<10s} {df_samples:>15,d} {rl_samples:>15,d}")
        
        if mode != 'raw':
            print("-" * 80)
    
    print("-" * 80 + "\n")
    
    # Summary for paper
    print("="*80)
    print("SUMMARY FOR PAPER (IEEE TSG)")
    print("="*80 + "\n")
    
    print(f"Dataset: DKASC Alice Springs (2010-2020)")
    print(f"PV Systems: {pv_sys} aggregated systems")
    print(f"Peak Capacity: {pv_cap:.1f} kW")
    print(f"Time Resolution: 5 minutes")
    print(f"\nThree preprocessing modes:")
    print(f"  - STRICT: {final_s:,} samples ({ret_s:.1f}% retention)")
    print(f"  - LIGHT:  {final_l:,} samples ({ret_l:.1f}% retention)")
    print(f"  - RAW:    {final_r:,} samples ({ret_r:.1f}% retention)")
    print(f"\nNormalization: Min-Max scaling using STRICT train set parameters")
    print(f"Split: Train (2010-2017), Val (2018), Test (2019-2020)")
    print(f"\nKey difference from STRICT to RAW:")
    print(f"  - Samples retained: +{final_r - final_s:,} ({(final_r/final_s - 1)*100:.1f}% more)")
    print(f"  - This enables robustness testing under noisy conditions")
    
    print("\n" + "="*80 + "\n")
    
    # Save to file
    output_file = Path("processed_data/MODE_COMPARISON_TABLE.txt")
    with open(output_file, 'w') as f:
        f.write("DKASC PREPROCESSING - MODE COMPARISON TABLE\n")
        f.write("="*80 + "\n\n")
        f.write("For IEEE TSG Paper: Distributionally Robust RL for PV-BESS\n\n")
        
        f.write("Table 1: Dataset Statistics by Mode\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Metric':<40s} {'STRICT':>12s} {'LIGHT':>12s} {'RAW':>12s}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'PV Systems':<40s} {pv_sys:>12d} {pv_sys:>12d} {pv_sys:>12d}\n")
        f.write(f"{'Peak Capacity (kW)':<40s} {pv_cap:>12.1f} {pv_cap:>12.1f} {pv_cap:>12.1f}\n")
        f.write(f"{'Final Samples':<40s} {final_s:>12,d} {final_l:>12,d} {final_r:>12,d}\n")
        f.write(f"{'Retention Rate (%)':<40s} {ret_s:>12.2f} {ret_l:>12.2f} {ret_r:>12.2f}\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("Normalization: Min-Max scaling fitted on STRICT train set\n")
        f.write("Note: LIGHT and RAW use STRICT normalization parameters\n")
        f.write("      Values may exceed [0,1] if distribution differs\n")
    
    print(f"Comparison table saved to: {output_file}")


if __name__ == "__main__":
    generate_comparison_table()
