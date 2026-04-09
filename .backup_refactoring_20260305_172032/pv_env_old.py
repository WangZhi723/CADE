"""
PV-BESS Environment for RL Training
Supports both simple state and multi-scale observations
"""

import numpy as np
import gym
from gym import spaces
from typing import Dict, Tuple, Optional
import pickle
import gzip


class PVBESSEnv(gym.Env):
    """Photovoltaic Battery Energy Storage System Environment"""
    
    def __init__(self, data_path: str, mode: str = 'simple'):
        """
        Args:
            data_path: Path to RL samples (pkl.gz file)
            mode: 'simple' for flat state, 'multiscale' for CNN+LSTM
        """
        super().__init__()
        
        self.mode = mode
        
        # Load data
        with gzip.open(data_path, 'rb') as f:
            self.samples = pickle.load(f)
        
        # BESS parameters
        self.capacity = 1.0  # MWh
        self.power_rated = 0.5  # MW
        self.soc_min = 0.1
        self.soc_max = 0.9
        self.eta_charge = 0.95
        self.eta_discharge = 0.95
        self.dt = 5.0 / 60.0  # 5 minutes in hours
        
        # Action space: battery power [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation space
        if mode == 'simple':
            # Flat state vector
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
            )
        else:
            # Multi-scale: short_term (6, 12) + long_term (288, 2) + current (6)
            self.observation_space = spaces.Dict({
                'short_term': spaces.Box(low=-np.inf, high=np.inf, shape=(6, 12), dtype=np.float32),
                'long_term': spaces.Box(low=-np.inf, high=np.inf, shape=(288, 2), dtype=np.float32),
                'current': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
            })
        
        # Episode state
        self.current_idx = 0
        self.soc = 0.5
        self.episode_length = 288  # 24 hours
        self.step_count = 0
        
        # Metrics tracking
        self.episode_losses = []
        self.episode_rewards = []
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        # Random starting point
        self.current_idx = np.random.randint(0, len(self.samples) - self.episode_length)
        self.soc = 0.5
        self.step_count = 0
        self.episode_losses = []
        self.episode_rewards = []
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        sample = self.samples[self.current_idx + self.step_count]
        
        if self.mode == 'simple':
            # Flatten to simple state
            obs = np.concatenate([
                sample['current_state'],  # 6 dims
                [self.soc],
                [sample['pv_actual']],
                [sample['pv_forecast']],
                [0.0]  # previous action (placeholder)
            ]).astype(np.float32)
            return obs
        else:
            # Multi-scale observation
            return {
                'short_term': sample['short_term'].T.astype(np.float32),  # (6, 12)
                'long_term': sample['long_term'].astype(np.float32),  # (288, 2)
                'current': np.concatenate([
                    sample['current_state'],
                    [self.soc]
                ]).astype(np.float32)
            }
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step"""
        # Parse action
        p_battery = np.clip(action[0], -1, 1) * self.power_rated
        
        # Get current sample
        sample = self.samples[self.current_idx + self.step_count]
        pv_actual = sample['pv_actual'] * 2.0  # Denormalize (assuming 2MW capacity)
        
        # Simulate load (simple sinusoidal pattern)
        hour = (self.step_count * 5 / 60.0) % 24
        load = 0.5 + 0.5 * np.sin(2 * np.pi * (hour - 6) / 24)  # MW
        
        # Update SoC
        if p_battery > 0:  # Discharge
            energy_change = -p_battery / self.eta_discharge
        else:  # Charge
            energy_change = -p_battery * self.eta_charge
        
        delta_soc = energy_change * self.dt / self.capacity
        new_soc = np.clip(self.soc + delta_soc, self.soc_min, self.soc_max)
        
        # Compute energy balance
        p_available = pv_actual + p_battery
        energy_gap = max(0, load - p_available)
        
        # Compute reward
        reward = -energy_gap * 100  # Penalty for energy deficit
        
        # SoC constraint penalty
        if new_soc >= self.soc_max or new_soc <= self.soc_min:
            reward -= 50
        
        # Track metrics
        self.episode_losses.append(energy_gap)
        self.episode_rewards.append(reward)
        
        # Update state
        self.soc = new_soc
        self.step_count += 1
        
        # Check if done
        done = self.step_count >= self.episode_length
        
        # Info
        info = {
            'soc': self.soc,
            'energy_gap': energy_gap,
            'pv_actual': pv_actual,
            'load': load,
            'p_battery': p_battery
        }
        
        if done:
            # Compute episode metrics
            losses = np.array(self.episode_losses)
            info['episode_return'] = sum(self.episode_rewards)
            info['mean_loss'] = losses.mean()
            info['max_loss'] = losses.max()
            
            # CVaR computation
            if len(losses) > 0:
                sorted_losses = np.sort(losses)[::-1]
                n_tail = max(1, int(0.1 * len(sorted_losses)))
                info['cvar_0.1'] = sorted_losses[:n_tail].mean()
        
        obs = self._get_observation()
        
        return obs, reward, done, info
    
    def render(self, mode='human'):
        """Render environment"""
        pass


def make_env(data_path: str, mode: str = 'simple'):
    """Factory function to create environment"""
    return PVBESSEnv(data_path, mode)
