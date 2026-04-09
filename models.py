"""
Neural Network Models for DR3L
Includes: MultiScaleFusionNet, QuantileCritic, GaussianActor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MultiScaleFusionNet(nn.Module):
    """Multi-scale feature fusion network (CNN + LSTM + Attention)"""
    
    def __init__(self, 
                 cnn_input_channels: int = 6,
                 lstm_input_dim: int = 2,
                 fusion_dim: int = 256,
                 short_window: int = 12,
                 long_window: int = 288):
        super().__init__()
        
        self.short_window = short_window
        self.long_window = long_window
        
        # Short-term CNN (1D causal convolution)
        self.cnn = nn.Sequential(
            nn.Conv1d(cnn_input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Long-term LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, fusion_dim)
        )
    
    def forward(self, short_term: torch.Tensor, long_term: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            short_term: (batch, channels, short_window)
            long_term: (batch, long_window, lstm_features)
        Returns:
            features: (batch, fusion_dim)
            attention_weights: dict
        """
        # CNN features
        cnn_out = self.cnn(short_term)  # (batch, 128, 1)
        cnn_features = cnn_out.squeeze(-1)  # (batch, 128)
        
        # LSTM features
        lstm_out, _ = self.lstm(long_term)  # (batch, long_window, 128)
        lstm_features = lstm_out[:, -1, :]  # (batch, 128) - last timestep
        
        # Fusion
        combined = torch.cat([cnn_features, lstm_features], dim=1)  # (batch, 256)
        features = self.fusion(combined)  # (batch, fusion_dim)
        
        return features, {}


class QuantileCritic(nn.Module):
    """Quantile regression critic for distributional RL with CVaR support
    
    Estimates Q(s,a) distribution using quantile regression
    """
    
    def __init__(self, feature_dim: int = 256, action_dim: int = 1, num_quantiles: int = 51):
        super().__init__()
        
        self.num_quantiles = num_quantiles
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        
        # Q(s,a) network: takes concatenated [features, actions]
        self.net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_quantiles)
        )
    
    def forward(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, feature_dim)
            actions: (batch, action_dim)
        Returns:
            quantiles: (batch, num_quantiles)
        """
        # Concatenate features and actions
        x = torch.cat([features, actions], dim=-1)
        return self.net(x)
    
    def compute_cvar(self, quantiles: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
        """
        Compute CVaR (Conditional Value at Risk) from quantiles
        
        CVaR_α = E[Z | Z ≤ VaR_α] ≈ (1/⌊αN⌋) Σ_{i=1}^{⌊αN⌋} θ_i
        
        Args:
            quantiles: (batch, num_quantiles) - quantile estimates
            alpha: confidence level (e.g., 0.1 for worst 10%)
        
        Returns:
            cvar: (batch,) - CVaR estimate (mean of worst α quantiles)
        """
        # Sort quantiles in ascending order (worst to best for returns)
        sorted_q, _ = torch.sort(quantiles, dim=-1)
        
        # Take worst α% quantiles
        k = max(1, int(self.num_quantiles * alpha))
        worst_quantiles = sorted_q[:, :k]
        
        # CVaR = mean of worst quantiles
        cvar = worst_quantiles.mean(dim=-1)
        
        return cvar
    
    def compute_var(self, quantiles: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
        """
        Compute VaR (Value at Risk) from quantiles
        
        Args:
            quantiles: (batch, num_quantiles)
            alpha: confidence level
        
        Returns:
            var: (batch,) - VaR estimate (α-quantile)
        """
        sorted_q, _ = torch.sort(quantiles, dim=-1)
        k = max(0, int(self.num_quantiles * alpha) - 1)
        return sorted_q[:, k]


class GaussianActor(nn.Module):
    """Gaussian policy for continuous actions"""
    
    def __init__(self, feature_dim: int = 256, action_dim: int = 1):
        super().__init__()
        
        self.mean_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (batch, feature_dim)
        Returns:
            mean: (batch, action_dim)
            std: (batch, action_dim)
        """
        mean = self.mean_net(features)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std
    
    def sample(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability"""
        mean, std = self.forward(features)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
    
    def get_distribution(self, features: torch.Tensor):
        """Get distribution for PPO"""
        mean, std = self.forward(features)
        return torch.distributions.Normal(mean, std)


class DDPGActor(nn.Module):
    """Deterministic actor for DDPG"""
    
    def __init__(self, state_dim: int, action_dim: int = 1, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DDPGCritic(nn.Module):
    """Q-function critic for DDPG"""
    
    def __init__(self, state_dim: int, action_dim: int = 1, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
