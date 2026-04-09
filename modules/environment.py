"""
Resilient PV-BESS Environment
弹性光伏储能环境：模拟 DKASC 站点的光伏输出和储能调度
"""
import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import sys
sys.path.append('/home/zhi/Risk')
from config import Config
from modules.data_loader import DKASCDataLoader
from modules.reward import RiskAwareReward


class PVBESSEnv(gym.Env):
    """
    光伏储能系统环境（Gym 接口）
    
    状态空间 (23维):
        - 时间特征 (4): hour_sin, hour_cos, day_sin, day_cos
        - 气象特征 (6): GHI, temp, humidity, wind_speed, wind_dir_sin, wind_dir_cos
        - 系统状态 (4): SoC, P_battery_prev, P_pv_actual, load_demand
        - 预测特征 (6): P_pv_forecast_1h, ..., P_pv_forecast_6h
        - 风险指标 (3): forecast_uncertainty, recent_energy_gap, cvar_rolling
    
    动作空间:
        - 连续动作: P_battery ∈ [-1, 1]，映射到 [-P_rated, P_rated] MW
    
    奖励函数:
        - 风险感知奖励（包含 CVaR 惩罚）
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 data_loader: Optional[DKASCDataLoader] = None,
                 split: str = 'train',
                 max_steps: int = 288):
        """
        初始化环境
        
        Args:
            data_loader: 数据加载器（如果为 None，则创建新的）
            split: 'train', 'val', 或 'test'
            max_steps: 每个 episode 的最大步数（默认288步=24小时）
        """
        super(PVBESSEnv, self).__init__()
        
        self.split = split
        self.max_steps = max_steps
        
        # 数据加载器
        if data_loader is None:
            self.data_loader = DKASCDataLoader()
            self.data_loader.prepare_data()
        else:
            self.data_loader = data_loader
        
        # 奖励函数
        self.reward_func = RiskAwareReward()
        
        # 定义动作空间：储能功率（连续）
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # 定义观测空间：23维状态向量
        obs_dim = 23
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # 环境状态
        self.current_step = 0
        self.episode_data = None
        self.soc = Config.BESS_SOC_INIT
        self.p_battery_prev = 0.0
        self.energy_gap_history = []
        
        # 统计信息
        self.episode_stats = {
            'total_reward': 0.0,
            'total_cost': 0.0,
            'total_pv_generation': 0.0,
            'total_load': 0.0,
            'soc_violations': 0,
            'ramp_violations': 0
        }
        
    def reset(self) -> np.ndarray:
        """
        重置环境，开始新的 episode
        
        Returns:
            初始观测
        """
        # 获取新的 episode 数据（24小时）
        self.episode_data = self.data_loader.get_episode_data(split=self.split)
        
        # 重置状态
        self.current_step = 0
        self.soc = Config.BESS_SOC_INIT
        self.p_battery_prev = 0.0
        self.energy_gap_history = []
        
        # 重置奖励函数
        self.reward_func.reset()
        
        # 重置统计
        for key in self.episode_stats:
            self.episode_stats[key] = 0.0 if 'total' in key else 0
        
        # 获取初始观测
        obs = self._get_observation()
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步动作
        
        Args:
            action: 储能功率（归一化到 [-1, 1]）
        
        Returns:
            (observation, reward, done, info)
        """
        # 1. 解析动作
        p_battery_normalized = np.clip(action[0], -1.0, 1.0)
        p_battery = p_battery_normalized * Config.BESS_POWER_RATED  # MW
        
        # 2. 获取当前时刻的数据
        current_data = self.episode_data.iloc[self.current_step]
        
        # 光伏输出
        p_pv_actual = current_data.get('PV_Power', 0.0)
        p_pv_actual = self.data_loader.denormalize(p_pv_actual, 'PV_Power')
        
        # 预测输出（用于计算极端偏差）
        p_pv_forecast = current_data.get('PV_Forecast_5min', p_pv_actual)
        if 'PV_Forecast_5min' in self.data_loader.norm_params:
            p_pv_forecast = self.data_loader.denormalize(p_pv_forecast, 'PV_Forecast_5min')
        
        # 负荷（模拟：基于时间的正弦模式）
        hour = current_data.get('hour', 12)
        p_load = Config.LOAD_BASE + (Config.LOAD_PEAK - Config.LOAD_BASE) * \
                 (0.5 + 0.5 * np.sin(2 * np.pi * (hour - 6) / 24))
        
        # 3. 储能物理模型
        # 分解为充电和放电
        if p_battery > 0:
            # 放电
            p_discharge = p_battery
            p_charge = 0.0
            energy_change = -p_discharge / Config.BESS_ETA_DISCHARGE
        else:
            # 充电
            p_charge = -p_battery
            p_discharge = 0.0
            energy_change = p_charge * Config.BESS_ETA_CHARGE
        
        # 更新 SoC（时间步长 = 5分钟 = 1/12 小时）
        dt = 5.0 / 60.0  # 小时
        delta_soc = energy_change * dt / Config.BESS_CAPACITY
        new_soc = self.soc + delta_soc
        
        # 检查 SoC 约束
        soc_violated = False
        if new_soc > Config.BESS_SOC_MAX:
            new_soc = Config.BESS_SOC_MAX
            soc_violated = True
            self.episode_stats['soc_violations'] += 1
        elif new_soc < Config.BESS_SOC_MIN:
            new_soc = Config.BESS_SOC_MIN
            soc_violated = True
            self.episode_stats['soc_violations'] += 1
        
        # 检查爬坡约束
        ramp_rate = abs(p_battery - self.p_battery_prev) / Config.BESS_POWER_RATED
        ramp_violated = ramp_rate > Config.BESS_RAMP_RATE
        if ramp_violated:
            self.episode_stats['ramp_violations'] += 1
        
        constraint_penalty = float(1.0 * int(soc_violated) + 0.5 * int(ramp_violated))
        p_grid = float(p_load - p_pv_actual - p_battery)

        reward, info = self.reward_func.compute_total_reward(
            p_grid_mw=p_grid,
            p_battery_mw=p_battery,
            p_battery_prev_mw=self.p_battery_prev,
            hour=float(hour),
            constraint_penalty=constraint_penalty,
        )
        
        # 5. 更新状态
        self.soc = new_soc
        self.p_battery_prev = p_battery
        self.current_step += 1
        
        # 记录能量缺口
        energy_gap = max(0, p_load - p_pv_actual - p_battery)
        self.energy_gap_history.append(energy_gap)
        
        # 更新统计
        self.episode_stats['total_reward'] += reward
        self.episode_stats['total_pv_generation'] += p_pv_actual * dt
        self.episode_stats['total_load'] += p_load * dt
        
        # 6. 判断是否结束
        done = self.current_step >= self.max_steps
        
        # 7. 获取新观测
        obs = self._get_observation()
        
        # 8. 额外信息
        info.update({
            'step': self.current_step,
            'soc': new_soc,
            'p_battery': p_battery,
            'p_pv': p_pv_actual,
            'p_load': p_load,
            'energy_gap': energy_gap,
            'soc_violated': soc_violated,
            'ramp_violated': ramp_violated
        })
        
        return obs, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """
        构建观测向量
        
        Returns:
            23维状态向量
        """
        if self.current_step >= len(self.episode_data):
            # Episode 结束，返回零向量
            return np.zeros(23, dtype=np.float32)
        
        current_data = self.episode_data.iloc[self.current_step]
        
        # 1. 时间特征 (4)
        hour_sin = current_data.get('hour_sin', 0.0)
        hour_cos = current_data.get('hour_cos', 0.0)
        day_sin = current_data.get('day_sin', 0.0)
        day_cos = current_data.get('day_cos', 0.0)
        
        # 2. 气象特征 (6) - 已归一化
        ghi = current_data.get('Global_Horizontal_Radiation', 0.0)
        temp = current_data.get('Weather_Temperature_Celsius', 0.5)
        humidity = current_data.get('Weather_Relative_Humidity', 0.5)
        wind_speed = current_data.get('Wind_Speed', 0.0)
        wind_dir_sin = current_data.get('Wind_Direction_Sin', 0.0)
        wind_dir_cos = current_data.get('Wind_Direction_Cos', 1.0)
        
        # 3. 系统状态 (4)
        soc = self.soc
        p_battery_prev = self.p_battery_prev / Config.BESS_POWER_RATED  # 归一化
        p_pv = current_data.get('PV_Power', 0.0)
        
        # 负荷（归一化）
        hour = current_data.get('hour', 12)
        load_demand = Config.LOAD_BASE + (Config.LOAD_PEAK - Config.LOAD_BASE) * \
                     (0.5 + 0.5 * np.sin(2 * np.pi * (hour - 6) / 24))
        load_demand = load_demand / Config.LOAD_PEAK
        
        # 4. 预测特征 (6) - 已归一化
        pv_forecast = [
            current_data.get(f'PV_Forecast_{i*5}min', p_pv)
            for i in range(1, 7)
        ]
        
        # 5. 风险指标 (3)
        # 预测不确定性（标准差）
        forecast_uncertainty = np.std(pv_forecast) if len(pv_forecast) > 0 else 0.0
        
        # 最近能量缺口
        recent_energy_gap = np.mean(self.energy_gap_history[-12:]) if len(self.energy_gap_history) > 0 else 0.0
        recent_energy_gap = np.clip(recent_energy_gap / Config.LOAD_PEAK, 0, 1)
        
        # CVaR（滚动）
        risk_metrics = self.reward_func.get_risk_metrics()
        cvar_rolling = risk_metrics.get('cvar_005', 0.0)
        cvar_rolling = np.clip(cvar_rolling / Config.LOAD_PEAK, 0, 1)
        
        # 组合观测
        obs = np.array([
            # 时间特征
            hour_sin, hour_cos, day_sin, day_cos,
            # 气象特征
            ghi, temp, humidity, wind_speed, wind_dir_sin, wind_dir_cos,
            # 系统状态
            soc, p_battery_prev, p_pv, load_demand,
            # 预测特征
            *pv_forecast,
            # 风险指标
            forecast_uncertainty, recent_energy_gap, cvar_rolling
        ], dtype=np.float32)
        
        return obs
    
    def render(self, mode='human'):
        """
        渲染环境（可视化）
        """
        if mode == 'human':
            print(f"\n步骤 {self.current_step}/{self.max_steps}")
            print(f"  SoC: {self.soc:.2%}")
            print(f"  储能功率: {self.p_battery_prev:.3f} MW")
            print(f"  累计奖励: {self.episode_stats['total_reward']:.2f}")
    
    def seed(self, seed=None):
        """
        设置随机种子（兼容旧版 Gym）
        
        Args:
            seed: 随机种子
        """
        np.random.seed(seed)
        return [seed]
    
    def get_episode_stats(self) -> Dict:
        """
        获取 episode 统计信息
        
        Returns:
            统计字典
        """
        stats = self.episode_stats.copy()
        
        # 添加奖励函数统计
        reward_stats = self.reward_func.get_statistics()
        stats.update(reward_stats)
        
        # 添加风险指标
        risk_metrics = self.reward_func.get_risk_metrics()
        stats.update(risk_metrics)
        
        return stats


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("=" * 80)
    print("测试光伏储能环境")
    print("=" * 80)
    
    # 创建数据加载器
    print("\n准备数据...")
    loader = DKASCDataLoader()
    loader.prepare_data()
    
    # 创建环境
    print("\n创建环境...")
    env = PVBESSEnv(data_loader=loader, split='train')
    
    print(f"  观测空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    
    # 运行一个 episode
    print("\n运行测试 episode...")
    obs = env.reset()
    print(f"  初始观测形状: {obs.shape}")
    
    total_reward = 0
    done = False
    step = 0
    
    while not done:
        # 随机动作（测试）
        action = env.action_space.sample()
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        
        # 每小时打印一次
        if step % 12 == 0:
            env.render()
    
    # 打印最终统计
    print("\n" + "=" * 80)
    print("Episode 统计:")
    stats = env.get_episode_stats()
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key:30s}: {value:12.4f}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
