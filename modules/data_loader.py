"""
Data Loader for DKASC Dataset
DKASC 数据集加载器：处理光伏和气象数据
"""
import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('/home/zhi/Risk')
from config import Config


class DKASCDataLoader:
    """
    DKASC 数据加载器
    
    功能:
        1. 读取 CSV 文件（5分钟间隔）
        2. 数据清洗：处理缺失值、异常值、传感器故障标记
        3. 特征工程：时间特征、滚动统计、预测特征
        4. 归一化：Min-Max 归一化到 [0, 1]
        5. 滑动窗口：构建训练序列
    """
    
    def __init__(self, 
                 data_path: str = Config.DATA_PATH,
                 master_meter_file: str = Config.MASTER_METER_FILE,
                 window_size: int = Config.WINDOW_SIZE,
                 train_ratio: float = Config.TRAIN_RATIO):
        """
        初始化数据加载器
        
        Args:
            data_path: 数据目录路径
            master_meter_file: 主表文件名
            window_size: 滑动窗口大小（时间步数）
            train_ratio: 训练集比例
        """
        self.data_path = data_path
        self.master_meter_file = master_meter_file
        self.window_size = window_size
        self.train_ratio = train_ratio
        
        self.data = None
        self.norm_params = {}
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def load_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        加载原始数据
        
        Args:
            use_cache: 是否使用缓存（如果存在）
        
        Returns:
            原始数据 DataFrame
        """
        file_path = os.path.join(self.data_path, self.master_meter_file)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        print(f"正在加载数据: {file_path}")
        print("这可能需要几分钟...")
        
        # 读取 CSV（大文件，可能较慢）
        # 只读取需要的列以节省内存
        usecols = [
            'timestamp',
            'Active_Power',
            'Active_Energy_Delivered_Received',
            'Current_Phase_Average',
            'Weather_Temperature_Celsius',
            'Weather_Relative_Humidity',
            'Global_Horizontal_Radiation',
            'Diffuse_Horizontal_Radiation',
            'Wind_Speed',
            'Wind_Direction'
        ]
        
        # 分块读取以节省内存
        chunk_size = 100000
        chunks = []
        
        for chunk in pd.read_csv(file_path, usecols=usecols, chunksize=chunk_size):
            chunks.append(chunk)
            if len(chunks) * chunk_size > 500000:  # 限制读取约50万行（约1个月数据）
                print(f"已读取 {len(chunks) * chunk_size} 行，停止读取以节省内存")
                break
        
        df = pd.concat(chunks, ignore_index=True)
        
        print(f"数据加载完成: {len(df)} 行")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        
        清洗步骤:
            1. 处理时间戳
            2. 处理缺失值
            3. 处理异常值（风速负值、传感器故障标记 -99999）
            4. 删除无效行
        
        Args:
            df: 原始数据
        
        Returns:
            清洗后的数据
        """
        print("\n开始数据清洗...")
        
        df = df.copy()
        
        # 1. 处理时间戳
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 2. 处理传感器故障标记 (-99999)
        for col in df.columns:
            if col != 'timestamp':
                df[col] = df[col].replace(-99999, np.nan)
                df[col] = df[col].replace(-99999.91, np.nan)
        
        # 3. 处理风速负值
        if 'Wind_Speed' in df.columns:
            df.loc[df['Wind_Speed'] < 0, 'Wind_Speed'] = np.nan
        
        # 4. 处理风向（转换为 sin/cos）
        if 'Wind_Direction' in df.columns:
            wind_dir_rad = np.deg2rad(df['Wind_Direction'].fillna(0))
            df['Wind_Direction_Sin'] = np.sin(wind_dir_rad)
            df['Wind_Direction_Cos'] = np.cos(wind_dir_rad)
            df = df.drop('Wind_Direction', axis=1)
        
        # 5. 填充缺失值
        # 对于气象数据：使用前向填充 + 线性插值
        weather_cols = [
            'Weather_Temperature_Celsius',
            'Weather_Relative_Humidity',
            'Global_Horizontal_Radiation',
            'Diffuse_Horizontal_Radiation',
            'Wind_Speed'
        ]
        
        for col in weather_cols:
            if col in df.columns:
                # 先前向填充（pandas 2.0+ 兼容）
                df[col] = df[col].ffill(limit=12)  # 最多填充1小时
                # 再线性插值
                df[col] = df[col].interpolate(method='linear', limit=12)
                # 最后用0填充（夜间辐射）
                if 'Radiation' in col:
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(df[col].mean())
        
        # 对于电力数据：使用前向填充
        power_cols = [
            'Active_Power',
            'Active_Energy_Delivered_Received',
            'Current_Phase_Average'
        ]
        
        for col in power_cols:
            if col in df.columns:
                df[col] = df[col].ffill(limit=12)  # pandas 2.0+ 兼容
                df[col] = df[col].fillna(0)  # 剩余的用0填充
        
        # 6. 删除仍有缺失值的行（极少数）
        initial_len = len(df)
        df = df.dropna()
        final_len = len(df)
        
        print(f"  删除了 {initial_len - final_len} 行无效数据")
        print(f"  剩余 {final_len} 行有效数据")
        
        return df
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特征工程
        
        新增特征:
            1. 时间特征：小时、星期、月份的 sin/cos 编码
            2. 滚动统计：光伏输出的滚动均值、标准差
            3. 预测特征：简单的持续性预测
        
        Args:
            df: 清洗后的数据
        
        Returns:
            添加特征后的数据
        """
        print("\n开始特征工程...")
        
        df = df.copy()
        
        # 1. 时间特征
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # 周期性编码（sin/cos）
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # 2. 光伏输出（从 Active_Power 转换）
        # 注意：Active_Power 可能是负值（向电网送电）
        if 'Active_Power' in df.columns:
            df['PV_Power'] = -df['Active_Power']  # 转换符号
            df['PV_Power'] = df['PV_Power'].clip(lower=0)  # 光伏输出非负
        
        # 3. 滚动统计（24小时窗口）
        window = 288  # 24小时
        if 'PV_Power' in df.columns:
            df['PV_Power_MA_24h'] = df['PV_Power'].rolling(window, min_periods=1).mean()
            df['PV_Power_Std_24h'] = df['PV_Power'].rolling(window, min_periods=1).std().fillna(0)
        
        if 'Global_Horizontal_Radiation' in df.columns:
            df['GHI_MA_1h'] = df['Global_Horizontal_Radiation'].rolling(12, min_periods=1).mean()
        
        # 4. 简单持续性预测（作为基准预测）
        if 'PV_Power' in df.columns:
            # 预测未来 30 分钟的光伏输出（持续性模型）
            for i in range(1, 7):  # 预测未来6步（30分钟）
                df[f'PV_Forecast_{i*5}min'] = df['PV_Power'].shift(i).bfill()  # pandas 2.0+ 兼容
        
        print(f"  添加了 {len(df.columns)} 个特征")
        
        return df
    
    def normalize_data(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        数据归一化 (Min-Max Scaling)
        
        X_norm = (X - X_min) / (X_max - X_min)
        
        Args:
            df: 特征数据
            fit: 是否拟合归一化参数（仅在训练集上为 True）
        
        Returns:
            归一化后的数据
        """
        print("\n开始数据归一化...")
        
        df = df.copy()
        
        # 需要归一化的列
        norm_cols = [
            'Active_Power', 'Active_Energy_Delivered_Received', 'Current_Phase_Average',
            'Weather_Temperature_Celsius', 'Weather_Relative_Humidity',
            'Global_Horizontal_Radiation', 'Diffuse_Horizontal_Radiation',
            'Wind_Speed', 'PV_Power', 'PV_Power_MA_24h', 'PV_Power_Std_24h', 'GHI_MA_1h'
        ] + [f'PV_Forecast_{i*5}min' for i in range(1, 7)]
        
        norm_cols = [col for col in norm_cols if col in df.columns]
        
        for col in norm_cols:
            if fit:
                # 计算并保存归一化参数
                self.norm_params[col] = {
                    'min': df[col].min(),
                    'max': df[col].max()
                }
            
            # 应用归一化
            if col in self.norm_params:
                min_val = self.norm_params[col]['min']
                max_val = self.norm_params[col]['max']
                
                if max_val - min_val > 1e-6:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[col] = 0.0
        
        print(f"  归一化了 {len(norm_cols)} 个特征")
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        划分训练集、验证集、测试集
        
        按时间顺序划分（避免数据泄露）
        
        Args:
            df: 处理后的数据
        
        Returns:
            (train_df, val_df, test_df)
        """
        print("\n开始数据划分...")
        
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + Config.VAL_RATIO))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        print(f"  训练集: {len(train_df)} 行")
        print(f"  验证集: {len(val_df)} 行")
        print(f"  测试集: {len(test_df)} 行")
        
        return train_df, val_df, test_df
    
    def prepare_data(self) -> Dict[str, pd.DataFrame]:
        """
        数据预处理主流程
        
        Returns:
            {'train': train_df, 'val': val_df, 'test': test_df}
        """
        print("=" * 80)
        print("开始数据预处理")
        print("=" * 80)
        
        # 1. 加载数据
        df = self.load_data()
        
        # 2. 清洗数据
        df = self.clean_data(df)
        
        # 3. 特征工程
        df = self.add_features(df)
        
        # 4. 划分数据集
        train_df, val_df, test_df = self.split_data(df)
        
        # 5. 归一化（仅在训练集上拟合）
        train_df = self.normalize_data(train_df, fit=True)
        val_df = self.normalize_data(val_df, fit=False)
        test_df = self.normalize_data(test_df, fit=False)
        
        # 保存
        self.train_data = train_df
        self.val_data = val_df
        self.test_data = test_df
        
        print("\n" + "=" * 80)
        print("数据预处理完成！")
        print("=" * 80)
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    def get_episode_data(self, split: str = 'train', start_idx: int = None) -> pd.DataFrame:
        """
        获取一个 episode 的数据（24小时）
        
        Args:
            split: 'train', 'val', 或 'test'
            start_idx: 起始索引（如果为 None，则随机选择）
        
        Returns:
            episode 数据（288 行）
        """
        if split == 'train':
            data = self.train_data
        elif split == 'val':
            data = self.val_data
        else:
            data = self.test_data
        
        if data is None:
            raise ValueError("数据未准备好，请先调用 prepare_data()")
        
        # 随机或指定起始位置
        max_start = len(data) - self.window_size
        if start_idx is None:
            start_idx = np.random.randint(0, max_start)
        else:
            start_idx = min(start_idx, max_start)
        
        episode_data = data.iloc[start_idx:start_idx + self.window_size].copy()
        
        return episode_data
    
    def denormalize(self, value: float, column: str) -> float:
        """
        反归一化
        
        X = X_norm * (X_max - X_min) + X_min
        
        Args:
            value: 归一化后的值
            column: 列名
        
        Returns:
            原始值
        """
        if column in self.norm_params:
            min_val = self.norm_params[column]['min']
            max_val = self.norm_params[column]['max']
            return value * (max_val - min_val) + min_val
        else:
            return value


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("=" * 80)
    print("测试 DKASC 数据加载器")
    print("=" * 80)
    
    # 创建数据加载器
    loader = DKASCDataLoader()
    
    # 预处理数据
    datasets = loader.prepare_data()
    
    # 打印数据集信息
    print("\n数据集信息:")
    for split, df in datasets.items():
        print(f"\n{split.upper()} 数据集:")
        print(f"  形状: {df.shape}")
        print(f"  时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
        print(f"  列: {list(df.columns)}")
    
    # 获取一个 episode
    print("\n" + "=" * 80)
    print("测试获取 Episode 数据")
    episode = loader.get_episode_data(split='train')
    print(f"Episode 形状: {episode.shape}")
    print(f"Episode 时间范围: {episode['timestamp'].min()} 至 {episode['timestamp'].max()}")
    
    # 打印前几行
    print("\nEpisode 前5行:")
    print(episode.head())
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
