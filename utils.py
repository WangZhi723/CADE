"""
Utility functions for DR3L experiments
"""

import numpy as np
import torch
from typing import Dict, List


def compute_cvar(losses: np.ndarray, alpha: float = 0.1) -> float:
    """
    Compute CVaR (Conditional Value at Risk)
    
    Args:
        losses: Array of loss values
        alpha: Confidence level (e.g., 0.1 for worst 10%)
    
    Returns:
        CVaR value
    """
    if len(losses) == 0:
        return 0.0
    
    sorted_losses = np.sort(losses)[::-1]  # Sort descending
    n_tail = max(1, int(alpha * len(sorted_losses)))
    cvar = sorted_losses[:n_tail].mean()
    
    return float(cvar)


def compute_var(losses: np.ndarray, alpha: float = 0.1) -> float:
    """
    Compute VaR (Value at Risk)
    
    Args:
        losses: Array of loss values
        alpha: Confidence level
    
    Returns:
        VaR value
    """
    if len(losses) == 0:
        return 0.0
    
    return float(np.quantile(losses, 1 - alpha))


def compute_constraint_violation(soc: float, soc_min: float, soc_max: float) -> float:
    """Compute SoC constraint violation"""
    if soc > soc_max:
        return soc - soc_max
    elif soc < soc_min:
        return soc_min - soc
    else:
        return 0.0


def compute_ramp_violation(p_current: float, p_prev: float, 
                          p_rated: float, ramp_max: float) -> float:
    """Compute ramp rate violation"""
    ramp_rate = abs(p_current - p_prev) / p_rated
    if ramp_rate > ramp_max:
        return ramp_rate - ramp_max
    else:
        return 0.0


def compute_dri_metrics(recovery_times: List[float], 
                       loss_of_load: List[float]) -> Dict[str, float]:
    """
    Compute Dynamic Resilience Index (DRI) metrics
    
    Args:
        recovery_times: List of recovery times (steps)
        loss_of_load: List of loss of load values (MWh)
    
    Returns:
        Dictionary of DRI metrics
    """
    metrics = {}
    
    if len(recovery_times) > 0:
        metrics['mean_recovery_time'] = np.mean(recovery_times)
        metrics['max_recovery_time'] = np.max(recovery_times)
        metrics['recovery_time_std'] = np.std(recovery_times)
    else:
        metrics['mean_recovery_time'] = 0.0
        metrics['max_recovery_time'] = 0.0
        metrics['recovery_time_std'] = 0.0
    
    if len(loss_of_load) > 0:
        metrics['total_loss_of_load'] = np.sum(loss_of_load)
        metrics['mean_loss_of_load'] = np.mean(loss_of_load)
        metrics['max_loss_of_load'] = np.max(loss_of_load)
    else:
        metrics['total_loss_of_load'] = 0.0
        metrics['mean_loss_of_load'] = 0.0
        metrics['max_loss_of_load'] = 0.0
    
    return metrics


def moving_average(data: np.ndarray, window: int = 100) -> np.ndarray:
    """Compute moving average"""
    if len(data) < window:
        return data
    
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
