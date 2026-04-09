"""
DKASC Data Preprocessing for Distributionally Robust RL (IEEE TSG Level)
Complete and reproducible preprocessing pipeline with memory-efficient chunked processing

Three dataset modes: STRICT, LIGHT, RAW
- All share identical train/val/test split
- Normalization based on STRICT train set
"""

import pandas as pd
import numpy as np
import pickle
import gzip
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from datetime import datetime
import warnings
import gc
warnings.filterwarnings('ignore')


class DataCleaningStats:
    """Track data cleaning statistics"""
    
    def __init__(self):
        self.stats = {
            'initial_samples': 0,
            'after_merge': 0,
            'num_pv_systems': 0,
            'pv_capacity_kw': 0,
            'negative_ghi_fixed': 0,
            'negative_wind_fixed': 0,
            'negative_pv_fixed': 0,
            'days_removed_completeness': 0,
            'samples_removed_completeness': 0,
            'outliers_clipped': 0,
            'missing_filled': 0,
            'final_samples': 0,
            'days_kept': 0,
            'date_range': None,
            'time_resolution_check': None
        }
    
    def print_summary(self, mode: str):
        """Print detailed cleaning summary"""
        print(f"\n{'='*80}")
        print(f"DATA CLEANING SUMMARY - {mode.upper()} MODE")
        print(f"{'='*80}")
        print(f"PV systems aggregated:                {self.stats['num_pv_systems']}")
        print(f"PV capacity (peak):                   {self.stats['pv_capacity_kw']:.2f} kW")
        print(f"Initial samples (after merge):        {self.stats['after_merge']:,}")
        
        if self.stats['time_resolution_check']:
            print(f"\nTime resolution check:")
            print(f"  {self.stats['time_resolution_check']}")
        
        print(f"\nPhysical constraint fixes:")
        print(f"  - Negative GHI fixed:               {self.stats['negative_ghi_fixed']:,}")
        print(f"  - Negative Wind Speed fixed:        {self.stats['negative_wind_fixed']:,}")
        print(f"  - Negative PV Power fixed:          {self.stats['negative_pv_fixed']:,}")
        
        if self.stats['days_removed_completeness'] > 0:
            print(f"\nCompleteness filtering (key variables only):")
            print(f"  - Days removed:                     {self.stats['days_removed_completeness']}")
            print(f"  - Samples removed:                  {self.stats['samples_removed_completeness']:,}")
            print(f"  - Days kept:                        {self.stats['days_kept']}")
        
        if self.stats['outliers_clipped'] > 0:
            print(f"\nOutlier handling:")
            print(f"  - Values clipped (3-sigma):         {self.stats['outliers_clipped']:,}")
        
        if self.stats['missing_filled'] > 0:
            print(f"\nMissing value imputation:")
            print(f"  - Values filled:                    {self.stats['missing_filled']:,}")
        
        print(f"\nFinal samples:                        {self.stats['final_samples']:,}")
        print(f"Data retention rate:                  {self.stats['final_samples']/max(self.stats['after_merge'],1)*100:.2f}%")
        
        if self.stats['date_range']:
            print(f"Date range:                           {self.stats['date_range'][0]} to {self.stats['date_range'][1]}")
        print(f"{'='*80}\n")


class DKASCPreprocessor:
    """Preprocess DKASC data with three quality levels and memory-efficient processing"""
    
    def __init__(self, data_root: str = "Dataset/DKASC"):
        self.data_root = Path(data_root)
        self.alice_path = self.data_root / "Alice Springs"
        self.yulara_path = self.data_root / "Yulara"
        
        # Shared parameters across all modes
        self.window_size = 288  # 24 hours at 5-min resolution
        self.forecast_horizon = 6  # 30 minutes ahead
        
    def load_raw_data(self, site: str = "alice") -> pd.DataFrame:
        """
        Load raw CSV files with ALL PV systems
        - Use engine="python" and on_bad_lines="skip"
        - Merge all PV active_power columns by timestamp
        - Aggregate as pv_power_total
        - Inner join with weather data
        - 5-min alignment
        - Sort by timestamp
        """
        print(f"\n{'='*80}")
        print(f"LOADING RAW DATA - {site.upper()}")
        print(f"{'='*80}\n")
        
        if site == "alice":
            weather_file = self.alice_path / "101-Site_DKA-WeatherStation.csv"
            pv_files = sorted(list(self.alice_path.glob("*-Site_DKA-M*.csv")))
        else:
            weather_file = self.yulara_path / "3052-Site_Environment-DG_Weather_Station.csv"
            pv_files = sorted(list(self.yulara_path.glob("*-Site_*.csv")))
            pv_files = [f for f in pv_files if "Weather" not in f.name and "Environment" not in f.name]
        
        print(f"Found {len(pv_files)} PV system files")
        
        # Load weather data
        print(f"\nLoading weather data: {weather_file.name}")
        weather_df = pd.read_csv(
            weather_file, 
            engine='python',
            on_bad_lines='skip',
            quotechar='"'
        )
        # Parse timestamp after loading (to handle quote issues)
        weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'], errors='coerce')
        weather_df = weather_df.dropna(subset=['timestamp'])
        print(f"  Weather samples: {len(weather_df):,}")
        
        # Load and aggregate PV data from ALL systems
        print(f"\nLoading PV systems (ALL {len(pv_files)} files):")
        pv_dfs = []
        
        for idx, pv_file in enumerate(pv_files, 1):
            try:
                df = pd.read_csv(
                    pv_file, 
                    engine='python',
                    on_bad_lines='skip',
                    quotechar='"'
                )
                
                # Check for Active_Power or active_power
                power_col = None
                if 'Active_Power' in df.columns:
                    power_col = 'Active_Power'
                elif 'active_power' in df.columns:
                    power_col = 'active_power'
                
                if power_col:
                    pv_data = df[['timestamp', power_col]].copy()
                    pv_data.rename(columns={power_col: f'pv_{idx}'}, inplace=True)
                    pv_dfs.append(pv_data)
                    print(f"  [{idx:2d}/{len(pv_files)}] {pv_file.name:50s} - {len(df):,} samples")
                else:
                    print(f"  [{idx:2d}/{len(pv_files)}] {pv_file.name:50s} - SKIPPED (no power column)")
                    
            except Exception as e:
                print(f"  [{idx:2d}/{len(pv_files)}] {pv_file.name:50s} - ERROR: {e}")
                continue
        
        if not pv_dfs:
            raise ValueError(f"No valid PV data found for {site}")
        
        print(f"\nSuccessfully loaded {len(pv_dfs)} PV systems")
        
        # Drop duplicate timestamps in each PV dataframe before merging
        # Also ensure timestamp is datetime type
        print("\nCleaning PV data before merge...")
        for i, df in enumerate(pv_dfs):
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                # Clean timestamp: remove extra quotes and whitespace
                if df['timestamp'].dtype == 'object':
                    df['timestamp'] = df['timestamp'].astype(str).str.strip()
                    # Remove quotes (both single and double, from start and end)
                    df['timestamp'] = df['timestamp'].str.replace(r'^["\']+|["\']+$', '', regex=True)
                
                # Parse with flexible format handling
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
                
                # Drop rows where timestamp couldn't be parsed
                invalid = df['timestamp'].isna().sum()
                if invalid > 0:
                    print(f"  PV {i+1}: Dropped {invalid} rows with invalid timestamps")
                    df = df.dropna(subset=['timestamp'])
            
            before = len(df)
            df = df.drop_duplicates('timestamp', keep='first')
            after = len(df)
            if before > after:
                print(f"  PV {i+1}: Removed {before - after} duplicate timestamps")
            pv_dfs[i] = df
        
        # Merge PV data progressively to save memory
        print("\nMerging PV systems (outer join to preserve all timestamps)...")
        pv_df = pv_dfs[0].copy()
        
        for idx, df in enumerate(pv_dfs[1:], start=2):
            pv_df = pv_df.merge(df, on='timestamp', how='outer')
            if idx % 5 == 0:
                print(f"  Merged {idx}/{len(pv_dfs)} systems...")
        
        print(f"  Merged all {len(pv_dfs)} systems")
        
        # Check time resolution after merge
        print("\nChecking time resolution...")
        time_diff = pv_df['timestamp'].sort_values().diff().dropna()
        resolution_counts = time_diff.value_counts().head(5)
        print(f"  Time step distribution:")
        for delta, count in resolution_counts.items():
            print(f"    {delta}: {count:,} occurrences")
        
        # Verify 5-minute resolution dominates
        expected_5min = pd.Timedelta(minutes=5)
        if resolution_counts.index[0] != expected_5min:
            print(f"  ⚠️  WARNING: Expected 5-min resolution, but most common is {resolution_counts.index[0]}")
        else:
            print(f"  ✅ Confirmed 5-minute resolution")
        
        # Sum all PV power columns to get total
        power_cols = [c for c in pv_df.columns if c.startswith('pv_')]
        print(f"\nAggregating {len(power_cols)} PV power columns...")
        pv_df['pv_power_total'] = pv_df[power_cols].sum(axis=1)
        pv_df = pv_df[['timestamp', 'pv_power_total']]
        
        # Store metadata
        num_pv_systems = len(power_cols)
        pv_capacity_kw = pv_df['pv_power_total'].max()
        
        print(f"  PV total samples: {len(pv_df):,}")
        print(f"  PV power range: [{pv_df['pv_power_total'].min():.2f}, {pv_df['pv_power_total'].max():.2f}] kW")
        print(f"  Peak capacity: {pv_capacity_kw:.2f} kW")
        
        # Merge weather and PV with inner join (5-min alignment)
        print("\nMerging weather and PV data (inner join for time alignment)...")
        df = weather_df.merge(pv_df, on='timestamp', how='inner')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"  Merged samples: {len(df):,}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Columns: {list(df.columns)}")
        
        # Final time resolution check
        print("\nFinal time resolution check...")
        time_diff = df['timestamp'].diff().dropna()
        resolution_counts = time_diff.value_counts().head(3)
        time_resolution_str = ", ".join([f"{delta}: {count:,}" for delta, count in resolution_counts.items()])
        print(f"  {time_resolution_str}")
        
        # Store metadata for stats
        self.last_load_metadata = {
            'num_pv_systems': num_pv_systems,
            'pv_capacity_kw': pv_capacity_kw,
            'time_resolution_check': time_resolution_str
        }
        
        # Clean up
        del weather_df, pv_df, pv_dfs
        gc.collect()
        
        return df
    
    def clean_data(self, df: pd.DataFrame, mode: str, metadata: Dict = None) -> Tuple[pd.DataFrame, DataCleaningStats]:
        """
        Clean data based on mode with detailed statistics
        
        STRICT mode:
            - Physical constraints: GHI >= 0, Wind_Speed >= 0, pv_power_total >= 0
            - Remove days with completeness < 0.8
            - 3-sigma outlier clipping
            - Forward/backward fill
            
        LIGHT mode:
            - Physical constraints only
            - Remove days with completeness < 0.6
            - Simple fill
            
        RAW mode:
            - NO physical constraints applied
            - NO completeness filtering applied
            - NO outlier clipping applied
            - Minimal imputation: forward fill + backward fill ONLY
            - Drop rows only if timestamp missing or critical columns (pv_power_total, GHI) missing
            - Preserve raw data characteristics while ensuring RL window construction is possible
        """
        print(f"\n{'='*80}")
        print(f"CLEANING DATA - {mode.upper()} MODE")
        print(f"{'='*80}\n")
        
        stats = DataCleaningStats()
        df = df.copy()
        
        stats.stats['after_merge'] = len(df)
        
        # Store metadata from loading
        if metadata:
            stats.stats['num_pv_systems'] = metadata.get('num_pv_systems', 0)
            stats.stats['pv_capacity_kw'] = metadata.get('pv_capacity_kw', 0)
            stats.stats['time_resolution_check'] = metadata.get('time_resolution_check', 'N/A')
        
        # Replace error codes with NaN (all modes)
        print("Replacing error codes with NaN...")
        df = df.replace(-99999, np.nan)
        df = df.replace(-999, np.nan)
        
        # Physical constraint fixes (only for STRICT and LIGHT modes)
        if mode in ["strict", "light"]:
            print("\nApplying physical constraints...")
            
            # GHI >= 0
            if 'Global_Horizontal_Radiation' in df.columns:
                neg_ghi = (df['Global_Horizontal_Radiation'] < 0).sum()
                df.loc[df['Global_Horizontal_Radiation'] < 0, 'Global_Horizontal_Radiation'] = 0
                stats.stats['negative_ghi_fixed'] = neg_ghi
                print(f"  Fixed {neg_ghi:,} negative GHI values")
            
            # Wind_Speed >= 0
            if 'Wind_Speed' in df.columns:
                neg_wind = (df['Wind_Speed'] < 0).sum()
                df.loc[df['Wind_Speed'] < 0, 'Wind_Speed'] = 0
                stats.stats['negative_wind_fixed'] = neg_wind
                print(f"  Fixed {neg_wind:,} negative Wind Speed values")
            
            # pv_power_total >= 0
            neg_pv = (df['pv_power_total'] < 0).sum()
            df.loc[df['pv_power_total'] < 0, 'pv_power_total'] = 0
            stats.stats['negative_pv_fixed'] = neg_pv
            print(f"  Fixed {neg_pv:,} negative PV power values")
        else:
            # RAW mode: only count but don't fix
            print("\nRAW mode: Physical constraints NOT applied (preserving raw data)")
            if 'Global_Horizontal_Radiation' in df.columns:
                neg_ghi = (df['Global_Horizontal_Radiation'] < 0).sum()
                stats.stats['negative_ghi_fixed'] = 0
                print(f"  Negative GHI values: {neg_ghi:,} (kept as-is)")
            
            if 'Wind_Speed' in df.columns:
                neg_wind = (df['Wind_Speed'] < 0).sum()
                stats.stats['negative_wind_fixed'] = 0
                print(f"  Negative Wind Speed values: {neg_wind:,} (kept as-is)")
            
            neg_pv = (df['pv_power_total'] < 0).sum()
            stats.stats['negative_pv_fixed'] = 0
            print(f"  Negative PV power values: {neg_pv:,} (kept as-is)")
        
        if mode == "strict":
            # Remove days with completeness < 0.8 (based on KEY variables only)
            print("\nFiltering days by completeness (threshold: 0.8, key variables only)...")
            df['date'] = df['timestamp'].dt.date
            
            # Only check completeness for key variables used in RL
            key_vars = ['Global_Horizontal_Radiation', 'pv_power_total']
            key_vars_present = [v for v in key_vars if v in df.columns]
            
            print(f"  Key variables checked: {key_vars_present}")
            
            initial_days = df['date'].nunique()
            daily_completeness = df.groupby('date').apply(
                lambda x: x[key_vars_present].notna().mean().mean()
            )
            
            valid_dates = daily_completeness[daily_completeness >= 0.8].index
            removed_dates = daily_completeness[daily_completeness < 0.8].index
            
            samples_before = len(df)
            df = df[df['date'].isin(valid_dates)]
            samples_after = len(df)
            
            stats.stats['days_removed_completeness'] = len(removed_dates)
            stats.stats['samples_removed_completeness'] = samples_before - samples_after
            stats.stats['days_kept'] = len(valid_dates)
            
            print(f"  Days removed: {len(removed_dates)} / {initial_days}")
            print(f"  Days kept: {len(valid_dates)}")
            print(f"  Samples removed: {samples_before - samples_after:,}")
            
            if len(removed_dates) > 0 and len(removed_dates) <= 20:
                print(f"  Removed dates: {sorted(removed_dates)}")
            
            # Smooth outliers (3-sigma rule)
            print("\nClipping outliers (3-sigma rule)...")
            outlier_cols = ['Global_Horizontal_Radiation', 'Weather_Temperature_Celsius', 
                           'Wind_Speed', 'pv_power_total']
            
            total_clipped = 0
            for col in outlier_cols:
                if col in df.columns:
                    mean = df[col].mean()
                    std = df[col].std()
                    lower = mean - 3*std
                    upper = mean + 3*std
                    
                    clipped = ((df[col] > upper) | (df[col] < lower)).sum()
                    df.loc[df[col] > upper, col] = upper
                    df.loc[df[col] < lower, col] = lower
                    
                    if clipped > 0:
                        print(f"  {col}: {clipped:,} values clipped to [{lower:.2f}, {upper:.2f}]")
                        total_clipped += clipped
            
            stats.stats['outliers_clipped'] = total_clipped
            
            # Forward fill then backward fill
            print("\nFilling missing values (forward then backward)...")
            missing_before = df.isna().sum().sum()
            df = df.ffill().bfill()
            missing_after = df.isna().sum().sum()
            stats.stats['missing_filled'] = missing_before - missing_after
            print(f"  Filled {missing_before - missing_after:,} missing values")
            
        elif mode == "light":
            # Remove days with completeness < 0.6 (based on KEY variables only)
            print("\nFiltering days by completeness (threshold: 0.6, key variables only)...")
            df['date'] = df['timestamp'].dt.date
            
            # Only check completeness for key variables used in RL
            key_vars = ['Global_Horizontal_Radiation', 'pv_power_total']
            key_vars_present = [v for v in key_vars if v in df.columns]
            
            print(f"  Key variables checked: {key_vars_present}")
            
            initial_days = df['date'].nunique()
            daily_completeness = df.groupby('date').apply(
                lambda x: x[key_vars_present].notna().mean().mean()
            )
            
            valid_dates = daily_completeness[daily_completeness >= 0.6].index
            removed_dates = daily_completeness[daily_completeness < 0.6].index
            
            samples_before = len(df)
            df = df[df['date'].isin(valid_dates)]
            samples_after = len(df)
            
            stats.stats['days_removed_completeness'] = len(removed_dates)
            stats.stats['samples_removed_completeness'] = samples_before - samples_after
            stats.stats['days_kept'] = len(valid_dates)
             
            print(f"  Days removed: {len(removed_dates)} / {initial_days}")
            print(f"  Days kept: {len(valid_dates)}")
            print(f"  Samples removed: {samples_before - samples_after:,}")
            
            # Simple fill
            print("\nFilling missing values (forward then backward)...")
            missing_before = df.isna().sum().sum()
            df = df.ffill().bfill()
            missing_after = df.isna().sum().sum()
            stats.stats['missing_filled'] = missing_before - missing_after
            print(f"  Filled {missing_before - missing_after:,} missing values")
            
        else:  # raw
            # RAW mode: Minimal imputation ONLY for temporal continuity
            # Goal: Preserve raw data characteristics while ensuring RL window construction is possible
            print("\nRAW mode: Minimal imputation (forward fill + backward fill only)")
            print("  - No physical constraints applied")
            print("  - No completeness filtering applied")
            print("  - No outlier clipping applied")
            print("  - Only minimal fill to preserve temporal continuity")
            
            # Step 1: Drop rows with missing timestamp (essential for time series)
            if 'timestamp' in df.columns:
                before_drop = len(df)
                df = df.dropna(subset=['timestamp'])
                after_drop = len(df)
                if before_drop != after_drop:
                    print(f"  Dropped {before_drop - after_drop:,} rows with missing timestamp")
            
            # Step 2: Sort by timestamp to ensure proper fill order
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Step 3: Minimal forward fill then backward fill
            # This preserves temporal continuity without aggressive imputation
            print("\n  Applying minimal fill (forward fill + backward fill)...")
            missing_before = df.isna().sum().sum()
            
            # Forward fill (carry last valid value forward)
            df = df.ffill()
            
            # Backward fill (fill remaining leading NaNs)
            df = df.bfill()
            
            missing_after = df.isna().sum().sum()
            filled_count = missing_before - missing_after
            
            stats.stats['missing_filled'] = filled_count
            print(f"  Missing values before fill: {missing_before:,}")
            print(f"  Missing values after fill: {missing_after:,}")
            print(f"  Values filled: {filled_count:,}")
            
            # Step 4: Only drop rows that still have NaN in critical columns for RL
            # Critical columns needed for RL window construction
            critical_cols = ['pv_power_total', 'Global_Horizontal_Radiation']
            critical_cols_present = [col for col in critical_cols if col in df.columns]
            
            if critical_cols_present:
                before_drop = len(df)
                df = df.dropna(subset=critical_cols_present)
                after_drop = len(df)
                if before_drop != after_drop:
                    print(f"  Dropped {before_drop - after_drop:,} rows with NaN in critical columns: {critical_cols_present}")
                    print(f"  (These rows cannot be used for RL window construction)")
            
            # Note: Other columns may still have NaN, which is acceptable for RAW mode
            remaining_nan = df.isna().sum().sum()
            if remaining_nan > 0:
                print(f"  Note: {remaining_nan:,} NaN values remain in non-critical columns (preserved as raw data)")
        
        # Remove date column if exists
        df = df.drop(columns=['date'], errors='ignore')
        
        # For RAW mode, we already handled NaN dropping above
        # For other modes, drop remaining NaN rows
        if mode != 'raw':
            before_dropna = len(df)
            df = df.dropna()
            after_dropna = len(df)
            
            if before_dropna > after_dropna:
                print(f"\nDropped {before_dropna - after_dropna:,} rows with remaining NaN values")
        
        # Check for time discontinuities AFTER cleaning (before feature engineering)
        # This is the right place because gaps are introduced by completeness filtering
        print("\nChecking temporal continuity after cleaning...")
        df = df.sort_values('timestamp').reset_index(drop=True)
        time_diff = df['timestamp'].diff()
        expected_5min = pd.Timedelta(minutes=5)
        gaps = time_diff[time_diff > expected_5min * 2]  # Gaps > 10 min
        
        num_gaps = len(gaps)
        stats.stats['time_gaps'] = num_gaps
        
        if num_gaps > 0:
            print(f"  ⚠️  Found {num_gaps} time gaps > 10 minutes")
            print(f"      These gaps were introduced by completeness filtering")
            # Show largest gaps
            if num_gaps <= 5:
                print(f"      Gap sizes: {sorted(gaps.values, reverse=True)[:5]}")
        else:
            print(f"  ✅ No temporal discontinuity detected (all gaps ≤ 10 min)")
            print(f"      Dataset maintains continuous 5-minute resolution")
        
        stats.stats['final_samples'] = len(df)
        stats.stats['date_range'] = (
            df['timestamp'].min().strftime('%Y-%m-%d'),
            df['timestamp'].max().strftime('%Y-%m-%d')
        )
        
        # Print summary
        stats.print_summary(mode)
        
        return df, stats
    
    def add_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Add engineered features (same for all modes)"""
        print("\nAdding engineered features...")
        df = df.copy()
        
        # Time features (no NaN introduced)
        df['hour'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60.0
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Wind direction encoding (may introduce NaN if Wind_Direction has NaN)
        if 'Wind_Direction' in df.columns:
            df['wind_dir_sin'] = np.sin(np.deg2rad(df['Wind_Direction']))
            df['wind_dir_cos'] = np.cos(np.deg2rad(df['Wind_Direction']))
        
        # Rolling statistics (short-term)
        # Use min_periods=2 for std to avoid NaN from single sample
        for col in ['Global_Horizontal_Radiation', 'pv_power_total']:
            if col in df.columns:
                df[f'{col}_roll_mean_1h'] = df[col].rolling(12, min_periods=1).mean()
                # std needs at least 2 samples, fill first values with 0
                df[f'{col}_roll_std_1h'] = df[col].rolling(12, min_periods=2).std().fillna(0)
        
        # PV forecast (simple persistence model) - ONLY drops last 36 rows
        df['pv_forecast_1h'] = df['pv_power_total'].shift(-12)
        df['pv_forecast_2h'] = df['pv_power_total'].shift(-24)
        df['pv_forecast_3h'] = df['pv_power_total'].shift(-36)
        
        # Only drop rows with NaN from shift operations (last 36 rows)
        # This preserves time continuity in the main data
        before = len(df)
        df = df.dropna(subset=['pv_forecast_1h', 'pv_forecast_2h', 'pv_forecast_3h'])
        after = len(df)
        
        print(f"\n  Added time, wind, rolling, and forecast features")
        print(f"  Dropped {before - after:,} rows from forecast shifts (last 36 rows)")
        print(f"  Final feature count: {len(df.columns)}")
        print(f"  Final samples: {len(df):,}")
        
        # Return 0 for gap count since it's now computed in clean_data
        return df, 0
    
    def normalize(self, df: pd.DataFrame, norm_params: Dict = None, 
                  fit: bool = False) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize features using STRICT train set parameters
        
        Args:
            df: DataFrame to normalize
            norm_params: Normalization parameters (if fit=False)
            fit: Whether to fit new parameters (only for STRICT train)
        
        Returns:
            Normalized DataFrame and normalization parameters
        """
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['hour', 'day_of_year', 'hour_sin', 'hour_cos', 
                       'day_sin', 'day_cos', 'wind_dir_sin', 'wind_dir_cos']
        cols_to_norm = [c for c in numeric_cols if c not in exclude_cols]
        
        if fit:
            print("\nFitting normalization parameters (min-max scaling)...")
            norm_params = {}
            for col in cols_to_norm:
                # Check for NaN before computing min/max
                if df[col].isna().any():
                    print(f"  ⚠️  Warning: {col} contains NaN values, skipping")
                    continue
                
                norm_params[col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
            print(f"  Computed normalization for {len(norm_params)} features")
        else:
            print("\nApplying normalization parameters...")
        
        # Track clipping statistics
        clipped_count = 0
        
        for col in cols_to_norm:
            if col in norm_params:
                min_val = norm_params[col]['min']
                max_val = norm_params[col]['max']
                if max_val > min_val:
                    # Normalize
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                    
                    # Clip to [0, 1] for non-fit mode (LIGHT/RAW using STRICT params)
                    # This prevents extreme values from breaking the network
                    if not fit:
                        before_clip = ((df[col] < 0) | (df[col] > 1)).sum()
                        df[col] = np.clip(df[col], 0, 1)
                        if before_clip > 0:
                            clipped_count += before_clip
        
        if not fit and clipped_count > 0:
            print(f"  Clipped {clipped_count:,} values outside [0,1] range")
            print(f"  (Values outside training distribution, preserving numerical stability)")
        
        return df, norm_params
    
    def split_data(self, df: pd.DataFrame, site: str = "alice") -> Dict[str, pd.DataFrame]:
        """
        Split into train/val/test (same split for all modes)
        
        Alice Springs:
            - 2010-2017: train
            - 2018: val
            - 2019-2020: test
            
        Yulara:
            - All as test set (distribution shift evaluation)
        """
        print("\nSplitting data...")
        
        if site == "alice":
            train_df = df[df['timestamp'].dt.year <= 2017].copy()
            val_df = df[df['timestamp'].dt.year == 2018].copy()
            test_df = df[df['timestamp'].dt.year >= 2019].copy()
            
            print(f"  Train (2010-2017): {len(train_df):,} samples")
            print(f"  Val   (2018):      {len(val_df):,} samples")
            print(f"  Test  (2019-2020): {len(test_df):,} samples")
        else:  # yulara
            train_df = pd.DataFrame()
            val_df = pd.DataFrame()
            test_df = df.copy()
            
            print(f"  Test (all):        {len(test_df):,} samples")
        
        return {'train': train_df, 'val': val_df, 'test': test_df}
    
    def build_rl_samples_chunked(self, df: pd.DataFrame, output_file: Path, 
                                 chunk_size: int = 5000, streaming: bool = False) -> int:
        """
        Build RL samples with sliding window - chunked to save memory
        
        Args:
            df: Input DataFrame
            output_file: Output pickle file path
            chunk_size: Number of samples per chunk
            streaming: If True, use streaming save (lower memory, slower I/O)
                      If False, accumulate all samples (faster, more memory)
        
        Note: For datasets > 1M RL samples, consider streaming=True
        """
        print(f"\n  Building RL samples (chunk size: {chunk_size}, streaming: {streaming})...")
        
        if len(df) <= self.window_size + self.forecast_horizon:
            print(f"  Skipped: insufficient data (need > {self.window_size + self.forecast_horizon})")
            return 0
        
        # Check and handle missing columns (e.g., Yulara may lack some weather columns)
        required_short_term = ['Global_Horizontal_Radiation', 'Weather_Temperature_Celsius',
                               'Weather_Relative_Humidity', 'Wind_Speed', 
                               'wind_dir_sin', 'wind_dir_cos']
        required_long_term = ['pv_power_total', 'Global_Horizontal_Radiation']
        required_current = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                           'Global_Horizontal_Radiation', 'pv_power_total']
        
        # Fill missing columns with default values
        missing_short = [col for col in required_short_term if col not in df.columns]
        missing_long = [col for col in required_long_term if col not in df.columns]
        missing_current = [col for col in required_current if col not in df.columns]
        
        if missing_short or missing_long or missing_current:
            print(f"  ⚠️  Missing columns detected. Filling with defaults:")
            if missing_short:
                print(f"     Short-term: {missing_short}")
                for col in missing_short:
                    if 'Humidity' in col:
                        df[col] = 50.0  # Default humidity 50%
                    elif 'Temperature' in col:
                        df[col] = 25.0  # Default temperature 25°C
                    elif 'Wind' in col:
                        df[col] = 0.0 if 'Speed' in col else 0.0  # Default wind
                    else:
                        df[col] = 0.0
            if missing_long:
                print(f"     Long-term: {missing_long}")
                for col in missing_long:
                    df[col] = 0.0
            if missing_current:
                print(f"     Current state: {missing_current}")
                for col in missing_current:
                    if 'sin' in col or 'cos' in col:
                        df[col] = 0.0  # Default time features
                    else:
                        df[col] = 0.0
        
        total_samples = len(df) - self.window_size - self.forecast_horizon
        num_chunks = (total_samples + chunk_size - 1) // chunk_size
        
        # Process in chunks
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if streaming:
            # Streaming mode: save each chunk independently (lowest memory)
            # NOTE: This creates a list of chunks, not a single list
            # Loading requires special handling
            samples_written = 0
            
            with gzip.open(output_file, 'wb') as f:
                for chunk_idx in range(num_chunks):
                    start_idx = self.window_size + chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, len(df) - self.forecast_horizon)
                    
                    chunk_samples = []
                    for i in range(start_idx, end_idx):
                        # Use only existing columns (missing ones already filled above)
                        short_cols = [col for col in required_short_term if col in df.columns]
                        long_cols = [col for col in required_long_term if col in df.columns]
                        current_cols = [col for col in required_current if col in df.columns]
                        
                        sample = {
                            'short_term': df.iloc[i-12:i][short_cols].values,
                            'long_term': df.iloc[i-self.window_size:i][long_cols].values,
                            'current_state': df.iloc[i][current_cols].values,
                            'pv_actual': df.iloc[i]['pv_power_total'],
                            'pv_forecast': df.iloc[i]['pv_forecast_1h'],
                            'timestamp': df.iloc[i]['timestamp']
                        }
                        chunk_samples.append(sample)
                    
                    # Save chunk incrementally
                    if chunk_idx == 0:
                        # First chunk: save as list
                        pickle.dump(chunk_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        # Subsequent chunks: append (requires special loading)
                        # For simplicity, we accumulate in first pass
                        pass
                    
                    samples_written += len(chunk_samples)
                    
                    # Print progress every 20%
                    if (chunk_idx + 1) % max(1, num_chunks // 5) == 0:
                        progress = (chunk_idx + 1) / num_chunks * 100
                        print(f"    Progress: {progress:.1f}% ({samples_written:,}/{total_samples:,} samples)")
                    
                    del chunk_samples
                    gc.collect()
            
            # Note: Streaming save is complex with pickle. Fall back to accumulate mode.
            print(f"  ⚠️  Streaming mode not fully implemented. Using accumulate mode.")
            streaming = False
        
        if not streaming:
            # Accumulate mode: collect all samples then save (current approach)
            all_samples = []
            samples_written = 0
            
            for chunk_idx in range(num_chunks):
                start_idx = self.window_size + chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(df) - self.forecast_horizon)
                
                chunk_samples = []
                for i in range(start_idx, end_idx):
                    # Use only existing columns (missing ones already filled above)
                    short_cols = [col for col in required_short_term if col in df.columns]
                    long_cols = [col for col in required_long_term if col in df.columns]
                    current_cols = [col for col in required_current if col in df.columns]
                    
                    sample = {
                        'short_term': df.iloc[i-12:i][short_cols].values,
                        'long_term': df.iloc[i-self.window_size:i][long_cols].values,
                        'current_state': df.iloc[i][current_cols].values,
                        'pv_actual': df.iloc[i]['pv_power_total'],
                        'pv_forecast': df.iloc[i]['pv_forecast_1h'],
                        'timestamp': df.iloc[i]['timestamp']
                    }
                    chunk_samples.append(sample)
                
                all_samples.extend(chunk_samples)
                samples_written += len(chunk_samples)
                
                # Print progress every 20%
                if (chunk_idx + 1) % max(1, num_chunks // 5) == 0:
                    progress = (chunk_idx + 1) / num_chunks * 100
                    print(f"    Progress: {progress:.1f}% ({samples_written:,}/{total_samples:,} samples)")
                
                # Clear chunk to free memory
                del chunk_samples
                gc.collect()
            
            # Save all samples at once (compressed)
            print(f"  Saving {len(all_samples):,} samples to {output_file.name}...")
            with gzip.open(output_file, 'wb') as f:
                pickle.dump(all_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"  Saved: {output_file} ({output_file.stat().st_size / 1024 / 1024:.1f} MB)")
            
            total = len(all_samples)
            del all_samples
            gc.collect()
            
            return total
        
        return samples_written
    
    def preprocess(self, mode: str = "strict", site: str = "alice"):
        """
        Main preprocessing pipeline
        
        Args:
            mode: 'strict', 'light', or 'raw'
            site: 'alice' or 'yulara'
        """
        print(f"\n{'='*80}")
        print(f"DKASC PREPROCESSING PIPELINE")
        print(f"Site: {site.upper()} | Mode: {mode.upper()}")
        print(f"{'='*80}")
        
        # Load raw data (ALL PV systems)
        df = self.load_raw_data(site)
        
        # Clean data with detailed statistics (pass metadata)
        # Time gap detection happens here after completeness filtering
        df, clean_stats = self.clean_data(df, mode, metadata=self.last_load_metadata)
        
        # Add features (time gap already recorded in clean_stats)
        df, _ = self.add_features(df)
        
        # Split data
        splits = self.split_data(df, site)
        
        # Setup output directory
        output_dir = Path(f"processed_data/{mode}/{site}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Normalize
        if mode == "strict" and site == "alice":
            # Fit normalization on STRICT train set
            splits['train'], norm_params = self.normalize(splits['train'], fit=True)
            splits['val'], _ = self.normalize(splits['val'], norm_params)
            splits['test'], _ = self.normalize(splits['test'], norm_params)
            
            # Save normalization parameters
            norm_file = output_dir / "norm_params.pkl"
            with open(norm_file, 'wb') as f:
                pickle.dump(norm_params, f)
            print(f"\n  Saved normalization parameters: {norm_file}")
        else:
            # Load STRICT normalization parameters
            strict_norm_path = Path(f"processed_data/strict/alice/norm_params.pkl")
            if not strict_norm_path.exists():
                raise FileNotFoundError(
                    "Must run STRICT mode for Alice Springs first to generate normalization parameters"
                )
            with open(strict_norm_path, 'rb') as f:
                norm_params = pickle.load(f)
            
            if len(splits['train']) > 0:
                splits['train'], _ = self.normalize(splits['train'], norm_params)
            if len(splits['val']) > 0:
                splits['val'], _ = self.normalize(splits['val'], norm_params)
            if len(splits['test']) > 0:
                splits['test'], _ = self.normalize(splits['test'], norm_params)
            
            print(f"\n  Using STRICT normalization parameters from: {strict_norm_path}")
        
        # Save splits (chunked for memory efficiency)
        print("\nSaving processed data...")
        for split_name, split_df in splits.items():
            if len(split_df) > 0:
                output_file = output_dir / f"{split_name}.pkl.gz"
                with gzip.open(output_file, 'wb') as f:
                    pickle.dump(split_df, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                file_size = output_file.stat().st_size / 1024 / 1024
                print(f"  Saved {split_name:5s}: {len(split_df):,} samples ({file_size:.1f} MB)")
        
        # Build RL samples (chunked to save memory)
        print("\nBuilding RL samples...")
        for split_name, split_df in splits.items():
            if len(split_df) > self.window_size:
                output_file = output_dir / f"{split_name}_rl.pkl.gz"
                num_samples = self.build_rl_samples_chunked(split_df, output_file, chunk_size=5000)
                
                if num_samples > 0:
                    file_size = output_file.stat().st_size / 1024 / 1024
                    print(f"  Saved {split_name:5s} RL: {num_samples:,} samples ({file_size:.1f} MB)")
        
        # Clear memory
        del df, splits
        gc.collect()
        
        # Save statistics
        print("\nSaving statistics...")
        stats_file = output_dir / "stats.txt"
        with open(stats_file, 'w') as f:
            f.write(f"DKASC Preprocessing Statistics\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Site: {site.upper()}\n")
            f.write(f"Mode: {mode.upper()}\n")
            f.write(f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # PV System Information
            f.write(f"PV SYSTEM INFORMATION:\n")
            f.write(f"  Number of PV systems aggregated: {clean_stats.stats['num_pv_systems']}\n")
            f.write(f"  Peak PV capacity: {clean_stats.stats['pv_capacity_kw']:.2f} kW\n")
            f.write(f"  Time resolution check: {clean_stats.stats['time_resolution_check']}\n")
            
            # Temporal continuity report
            time_gaps = clean_stats.stats.get('time_gaps', 0)
            f.write(f"\nTEMPORAL CONTINUITY:\n")
            f.write(f"  Time gap occurrences > 10 min: {time_gaps}\n")
            if time_gaps == 0:
                f.write(f"  Status: ✅ No temporal discontinuity observed after preprocessing\n")
            else:
                f.write(f"  Status: ⚠️  Data contains temporal gaps (likely from completeness filtering)\n")
            f.write("\n")
            
            # Data splits
            split_stats = {}
            for split_name in ['train', 'val', 'test']:
                split_file = output_dir / f"{split_name}.pkl.gz"
                if split_file.exists():
                    with gzip.open(split_file, 'rb') as sf:
                        split_df = pickle.load(sf)
                    
                    split_stats[split_name] = {
                        'samples': len(split_df),
                        'time_range': f"{split_df['timestamp'].min()} to {split_df['timestamp'].max()}" if len(split_df) > 0 else "N/A",
                        'features': len(split_df.columns)
                    }
                    
                    f.write(f"{split_name.upper()}:\n")
                    f.write(f"  Samples: {len(split_df):,}\n")
                    if len(split_df) > 0:
                        f.write(f"  Time range: {split_df['timestamp'].min()} to {split_df['timestamp'].max()}\n")
                        f.write(f"  Features: {len(split_df.columns)}\n")
                    f.write("\n")
                    
                    del split_df
                    gc.collect()
            
            # Cleaning statistics
            f.write(f"\nCLEANING STATISTICS:\n")
            f.write(f"  Initial samples (after merge): {clean_stats.stats['after_merge']:,}\n")
            f.write(f"  Final samples: {clean_stats.stats['final_samples']:,}\n")
            f.write(f"  Retention rate: {clean_stats.stats['final_samples']/max(clean_stats.stats['after_merge'],1)*100:.2f}%\n")
            f.write(f"  Date range: {clean_stats.stats['date_range'][0]} to {clean_stats.stats['date_range'][1]}\n\n")
            
            f.write(f"  Physical constraint fixes:\n")
            f.write(f"    Negative GHI fixed: {clean_stats.stats['negative_ghi_fixed']:,}\n")
            f.write(f"    Negative Wind Speed fixed: {clean_stats.stats['negative_wind_fixed']:,}\n")
            f.write(f"    Negative PV Power fixed: {clean_stats.stats['negative_pv_fixed']:,}\n\n")
            
            if clean_stats.stats['days_removed_completeness'] > 0:
                f.write(f"  Completeness filtering (key variables only):\n")
                f.write(f"    Days removed: {clean_stats.stats['days_removed_completeness']}\n")
                f.write(f"    Days kept: {clean_stats.stats['days_kept']}\n")
                f.write(f"    Samples removed: {clean_stats.stats['samples_removed_completeness']:,}\n\n")
            
            if clean_stats.stats['outliers_clipped'] > 0:
                f.write(f"  Outlier handling:\n")
                f.write(f"    Values clipped (3-sigma): {clean_stats.stats['outliers_clipped']:,}\n\n")
            
            if clean_stats.stats['missing_filled'] > 0:
                f.write(f"  Missing value imputation:\n")
                f.write(f"    Values filled: {clean_stats.stats['missing_filled']:,}\n\n")
            
            # Normalization note
            f.write(f"\nNORMALIZATION:\n")
            if mode == "strict" and site == "alice":
                f.write(f"  Normalization parameters fitted on STRICT train set\n")
                f.write(f"  Method: Min-Max scaling\n")
            else:
                f.write(f"  Normalization parameters from STRICT train set (Alice Springs)\n")
                f.write(f"  Note: Values may exceed [0,1] range if data differs from STRICT train distribution\n")
        
        print(f"  Saved statistics: {stats_file}")
        
        # Save retention rate comparison table (only for Alice)
        if site == "alice":
            comparison_file = output_dir / "retention_rate.txt"
            with open(comparison_file, 'w') as f:
                f.write(f"Data Retention Rate - {mode.upper()} Mode\n")
                f.write(f"{'='*60}\n\n")
                f.write(f"{'Split':<10s} {'Samples':<15s} {'Percentage':<12s}\n")
                f.write(f"{'-'*60}\n")
                
                total = clean_stats.stats['after_merge']
                for split_name in ['train', 'val', 'test']:
                    if split_name in split_stats:
                        samples = split_stats[split_name]['samples']
                        pct = samples / total * 100 if total > 0 else 0
                        f.write(f"{split_name.upper():<10s} {samples:>12,d}    {pct:>6.2f}%\n")
                
                f.write(f"{'-'*60}\n")
                final = clean_stats.stats['final_samples']
                final_pct = final / total * 100 if total > 0 else 0
                f.write(f"{'TOTAL':<10s} {final:>12,d}    {final_pct:>6.2f}%\n")
            
            print(f"  Saved retention rate: {comparison_file}")
        
        print(f"\n{'='*80}")
        print(f"PREPROCESSING COMPLETE")
        print(f"Output directory: {output_dir.absolute()}")
        print(f"{'='*80}\n")


def main():
    """
    Main execution function
    
    Usage:
        python preprocess_dkasc_data_v2.py [strict|light|raw]
    
    Process order:
        1. STRICT mode for Alice Springs (generates normalization params)
        2. LIGHT and RAW modes for Alice Springs (use STRICT params)
        3. All modes for Yulara (use Alice STRICT params)
    """
    import sys
    
    preprocessor = DKASCPreprocessor()
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "strict"
    
    if mode not in ["strict", "light", "raw"]:
        print("Usage: python preprocess_dkasc_data_v2.py [strict|light|raw]")
        print("\nModes:")
        print("  strict - Strictest cleaning (completeness >= 0.8, 3-sigma clipping)")
        print("  light  - Light cleaning (completeness >= 0.6)")
        print("  raw    - Minimal cleaning (time alignment only)")
        sys.exit(1)
    
    # Process Alice Springs
    print("\n" + "="*80)
    print("PROCESSING ALICE SPRINGS")
    print("="*80)
    preprocessor.preprocess(mode=mode, site="alice")
    
    # Process Yulara (for distribution shift test)
    print("\n" + "="*80)
    print("PROCESSING YULARA")
    print("="*80)
    try:
        preprocessor.preprocess(mode=mode, site="yulara")
    except Exception as e:
        print(f"\nWarning: Could not process Yulara: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("ALL PROCESSING COMPLETE")
    print("="*80)
    print(f"\nProcessed data saved in: processed_data/{mode}/")
    print("\nNext steps:")
    if mode == "strict":
        print("  1. Run LIGHT mode: python preprocess_dkasc_data_v2.py light")
        print("  2. Run RAW mode: python preprocess_dkasc_data_v2.py raw")
    print("  3. Use the processed data for RL training")


if __name__ == "__main__":
    main()
