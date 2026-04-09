"""
Generate cross-mode retention summary for IEEE TSG paper
Reads statistics from all three modes and creates a comparison table
"""

import re
from pathlib import Path


def parse_stats_file(stats_file: Path) -> dict:
    """Parse key statistics from stats.txt file"""
    stats = {
        'pv_systems': 0,
        'pv_capacity': 0.0,
        'initial': 0,
        'final': 0,
        'retention': 0.0,
        'days_removed': 0,
        'time_gaps': 0,
        'ghi_fixed': 0,
        'wind_fixed': 0,
        'pv_fixed': 0,
        'outliers_clipped': 0,
        'missing_filled': 0
    }
    
    if not stats_file.exists():
        return stats
    
    with open(stats_file, 'r') as f:
        content = f.read()
        
        # Extract values
        patterns = {
            'pv_systems': r'Number of PV systems aggregated:\s*(\d+)',
            'pv_capacity': r'Peak PV capacity:\s*([\d.]+)\s*kW',
            'initial': r'Initial samples \(after merge\):\s*([\d,]+)',
            'final': r'Final samples:\s*([\d,]+)',
            'retention': r'Retention rate:\s*([\d.]+)%',
            'days_removed': r'Days removed:\s*(\d+)',
            'time_gaps': r'Time gap occurrences > 10 min:\s*(\d+)',
            'ghi_fixed': r'Negative GHI fixed:\s*([\d,]+)',
            'wind_fixed': r'Negative Wind Speed fixed:\s*([\d,]+)',
            'pv_fixed': r'Negative PV Power fixed:\s*([\d,]+)',
            'outliers_clipped': r'Values clipped \(3-sigma\):\s*([\d,]+)',
            'missing_filled': r'Values filled:\s*([\d,]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                value = match.group(1).replace(',', '')
                if key in ['pv_capacity', 'retention']:
                    stats[key] = float(value)
                else:
                    stats[key] = int(value)
    
    return stats


def generate_summary_comparison():
    """Generate summary comparison across all modes"""
    
    modes = ['strict', 'light', 'raw']
    site = 'alice'
    
    print("\n" + "="*80)
    print("GENERATING CROSS-MODE RETENTION SUMMARY")
    print("="*80 + "\n")
    
    # Collect statistics from all modes
    all_stats = {}
    for mode in modes:
        stats_file = Path(f"processed_data/{mode}/{site}/stats.txt")
        if stats_file.exists():
            all_stats[mode] = parse_stats_file(stats_file)
            print(f"✅ Loaded {mode.upper():6s}: {stats_file}")
        else:
            print(f"❌ Missing {mode.upper():6s}: {stats_file}")
            print(f"   Run: python preprocess_dkasc_data_v2.py {mode}")
            return
    
    print()
    
    # Generate summary comparison file
    output_file = Path("processed_data/SUMMARY_COMPARISON.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DKASC PREPROCESSING - CROSS-MODE RETENTION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write("For IEEE TSG Paper: Distributionally Robust RL for PV-BESS Systems\n")
        f.write("Dataset: Alice Springs, Australia (2010-2020)\n\n")
        
        # System information (same across modes)
        pv_sys = all_stats['strict']['pv_systems']
        pv_cap = all_stats['strict']['pv_capacity']
        f.write(f"PV Systems Aggregated: {pv_sys}\n")
        f.write(f"Peak PV Capacity: {pv_cap:.1f} kW\n")
        f.write(f"Time Resolution: 5 minutes\n\n")
        
        # Main comparison table
        f.write("="*80 + "\n")
        f.write("TABLE 1: DATA RETENTION BY MODE\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Mode':<10s} {'Initial':>15s} {'Final':>15s} {'Retention':>12s} {'Days Removed':>15s}\n")
        f.write("-"*80 + "\n")
        
        for mode in modes:
            stats = all_stats[mode]
            f.write(f"{mode.upper():<10s} "
                   f"{stats['initial']:>15,d} "
                   f"{stats['final']:>15,d} "
                   f"{stats['retention']:>11.2f}% "
                   f"{stats['days_removed']:>15d}\n")
        
        f.write("-"*80 + "\n\n")
        
        # Incremental retention
        strict_final = all_stats['strict']['final']
        light_final = all_stats['light']['final']
        raw_final = all_stats['raw']['final']
        
        f.write("INCREMENTAL DATA GAIN:\n")
        f.write(f"  LIGHT vs STRICT: +{light_final - strict_final:,d} samples "
               f"({(light_final/strict_final - 1)*100:+.2f}%)\n")
        f.write(f"  RAW vs STRICT:   +{raw_final - strict_final:,d} samples "
               f"({(raw_final/strict_final - 1)*100:+.2f}%)\n")
        f.write(f"  RAW vs LIGHT:    +{raw_final - light_final:,d} samples "
               f"({(raw_final/light_final - 1)*100:+.2f}%)\n\n")
        
        # Cleaning operations table
        f.write("="*80 + "\n")
        f.write("TABLE 2: CLEANING OPERATIONS APPLIED\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Operation':<35s} {'STRICT':>12s} {'LIGHT':>12s} {'RAW':>12s}\n")
        f.write("-"*80 + "\n")
        
        f.write(f"{'Negative GHI Fixed':<35s} "
               f"{all_stats['strict']['ghi_fixed']:>12,d} "
               f"{all_stats['light']['ghi_fixed']:>12,d} "
               f"{all_stats['raw']['ghi_fixed']:>12,d}\n")
        
        f.write(f"{'Negative Wind Fixed':<35s} "
               f"{all_stats['strict']['wind_fixed']:>12,d} "
               f"{all_stats['light']['wind_fixed']:>12,d} "
               f"{all_stats['raw']['wind_fixed']:>12,d}\n")
        
        f.write(f"{'Negative PV Fixed':<35s} "
               f"{all_stats['strict']['pv_fixed']:>12,d} "
               f"{all_stats['light']['pv_fixed']:>12,d} "
               f"{all_stats['raw']['pv_fixed']:>12,d}\n")
        
        f.write(f"{'Outliers Clipped (3-sigma)':<35s} "
               f"{all_stats['strict']['outliers_clipped']:>12,d} "
               f"{all_stats['light']['outliers_clipped']:>12,d} "
               f"{all_stats['raw']['outliers_clipped']:>12,d}\n")
        
        f.write(f"{'Missing Values Filled':<35s} "
               f"{all_stats['strict']['missing_filled']:>12,d} "
               f"{all_stats['light']['missing_filled']:>12,d} "
               f"{all_stats['raw']['missing_filled']:>12,d}\n")
        
        f.write("-"*80 + "\n\n")
        
        # Temporal continuity
        f.write("="*80 + "\n")
        f.write("TABLE 3: TEMPORAL CONTINUITY\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Mode':<10s} {'Time Gaps > 10min':>20s} {'Status':>25s}\n")
        f.write("-"*80 + "\n")
        
        for mode in modes:
            gaps = all_stats[mode]['time_gaps']
            status = "✅ Continuous" if gaps == 0 else f"⚠️  {gaps} gaps"
            f.write(f"{mode.upper():<10s} {gaps:>20d} {status:>25s}\n")
        
        f.write("-"*80 + "\n\n")
        
        # Key takeaways for paper
        f.write("="*80 + "\n")
        f.write("KEY TAKEAWAYS FOR PAPER\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. DATASET SCALE:\n")
        f.write(f"   - Three preprocessing modes with different quality-quantity tradeoffs\n")
        f.write(f"   - STRICT mode retains {all_stats['strict']['retention']:.1f}% (highest quality)\n")
        f.write(f"   - RAW mode retains {all_stats['raw']['retention']:.1f}% (maximum coverage)\n")
        f.write(f"   - This enables robustness evaluation across data quality levels\n\n")
        
        f.write("2. TEMPORAL CONTINUITY:\n")
        strict_gaps = all_stats['strict']['time_gaps']
        light_gaps = all_stats['light']['time_gaps']
        raw_gaps = all_stats['raw']['time_gaps']
        
        if strict_gaps == 0 and light_gaps == 0 and raw_gaps == 0:
            f.write(f"   - No temporal discontinuity observed in any mode\n")
            f.write(f"   - All datasets maintain continuous 5-minute resolution\n")
            f.write(f"   - This ensures valid sequential modeling for RL\n\n")
        else:
            f.write(f"   - STRICT: {strict_gaps} gaps (from completeness filtering)\n")
            f.write(f"   - LIGHT: {light_gaps} gaps\n")
            f.write(f"   - RAW: {raw_gaps} gaps\n")
            f.write(f"   - Gaps indicate removed incomplete days\n\n")
        
        f.write("3. NORMALIZATION:\n")
        f.write(f"   - All modes use min-max scaling fitted on STRICT train set\n")
        f.write(f"   - Values outside [0,1] are clipped for numerical stability\n")
        f.write(f"   - This ensures fair comparison across modes\n\n")
        
        f.write("4. RECOMMENDED CITATION:\n")
        f.write(f"   \"We preprocessed the DKASC Alice Springs dataset (2010-2020)\n")
        f.write(f"    with three quality levels: STRICT ({all_stats['strict']['retention']:.1f}% retention),\n")
        f.write(f"    LIGHT ({all_stats['light']['retention']:.1f}%), and RAW ({all_stats['raw']['retention']:.1f}%).\n")
        f.write(f"    All modes aggregated {pv_sys} PV systems with peak capacity of {pv_cap:.1f} kW.\n")
        f.write(f"    The data was split into train (2010-2017), validation (2018),\n")
        f.write(f"    and test (2019-2020) sets with consistent normalization parameters.\"\n\n")
        
        # LaTeX table for paper
        f.write("="*80 + "\n")
        f.write("LATEX TABLE FOR PAPER\n")
        f.write("="*80 + "\n\n")
        
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{DKASC Dataset Statistics by Preprocessing Mode}\n")
        f.write("\\label{tab:dataset_stats}\n")
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("\\hline\n")
        f.write("Mode & Final Samples & Retention (\\%) & Days Removed \\\\\n")
        f.write("\\hline\n")
        
        for mode in modes:
            stats = all_stats[mode]
            f.write(f"{mode.upper():6s} & {stats['final']:>10,d} & "
                   f"{stats['retention']:>6.2f} & {stats['days_removed']:>4d} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
    print(f"✅ Summary comparison saved to: {output_file}\n")
    
    # Print preview to console
    print("="*80)
    print("PREVIEW: DATA RETENTION BY MODE")
    print("="*80 + "\n")
    
    print(f"{'Mode':<10s} {'Initial':>15s} {'Final':>15s} {'Retention':>12s} {'Days Removed':>15s}")
    print("-"*80)
    
    for mode in modes:
        stats = all_stats[mode]
        print(f"{mode.upper():<10s} "
              f"{stats['initial']:>15,d} "
              f"{stats['final']:>15,d} "
              f"{stats['retention']:>11.2f}% "
              f"{stats['days_removed']:>15d}")
    
    print("-"*80)
    print(f"\nINCREMENTAL GAIN:")
    print(f"  LIGHT vs STRICT: +{light_final - strict_final:,d} samples ({(light_final/strict_final - 1)*100:+.2f}%)")
    print(f"  RAW vs STRICT:   +{raw_final - strict_final:,d} samples ({(raw_final/strict_final - 1)*100:+.2f}%)")
    
    print("\n" + "="*80)
    print(f"✅ Summary saved to: {output_file}")
    print("="*80 + "\n")


if __name__ == "__main__":
    generate_summary_comparison()
