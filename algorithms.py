"""
Algorithms for IEEE TSG Paper
- DDPG
- PPO
- CVaR-RL
- DR3L (Distributionally Robust Risk-Resilient RL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque
import copy

from models import (MultiScaleFusionNet, QuantileCritic, GaussianActor,
                    DDPGActor, DDPGCritic)


class OrnsteinUhlenbeckNoise:
    """DDPG 探索：OU 过程相关噪声，比纯高斯更利于物理控制任务持续探索。"""

    def __init__(
        self,
        size: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.25,
    ):
        self.mu = mu * np.ones(size, dtype=np.float64)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.state = self.mu.copy()

    def reset(self) -> None:
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state.copy()


class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones))
        )
    
    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """环形存储的比例优先经验回放（PER），与 DR3L 的 8 元组 transition 兼容。"""

    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-6):
        self.capacity = max(1, int(capacity))
        self.alpha = float(alpha)
        self.eps = float(eps)
        self._data: list = [None] * self.capacity
        self._priorities = np.zeros(self.capacity, dtype=np.float64)
        self._pos = 0
        self._size = 0
        self._max_p = 1.0

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        self._data = [None] * self.capacity
        self._priorities.fill(0.0)
        self._pos = 0
        self._size = 0
        self._max_p = 1.0

    def append(self, transition: tuple) -> None:
        p = max(self._max_p, self.eps)
        self._data[self._pos] = transition
        self._priorities[self._pos] = p
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float) -> Tuple[list, np.ndarray, np.ndarray]:
        n = self._size
        if n < batch_size:
            raise ValueError("batch_size larger than buffer")
        if self._size < self.capacity:
            probs = self._priorities[:n] ** self.alpha
        else:
            probs = self._priorities ** self.alpha
        total = float(probs.sum())
        if total <= 0:
            probs = np.ones(n, dtype=np.float64) / n
        else:
            probs = probs / total
        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)
        batch = [self._data[i] for i in indices]
        w = (n * probs[indices]) ** (-float(beta))
        w = w / (w.max() + 1e-8)
        return batch, indices.astype(np.int64), w.astype(np.float64)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, err in zip(indices, td_errors):
            self._priorities[int(idx)] = float(abs(err)) + self.eps
        if len(td_errors):
            self._max_p = max(self._max_p, float(np.max(np.abs(td_errors))) + self.eps)


class DDPG:
    """Deep Deterministic Policy Gradient"""
    
    def __init__(self, state_dim: int, action_dim: int = 1, 
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3,
                 gamma: float = 0.99, tau: float = 0.005,
                 device: str = 'cuda',
                 ou_theta: float = 0.15,
                 ou_sigma: float = 0.28):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_dim = int(action_dim)
        self.ou_noise = OrnsteinUhlenbeckNoise(
            self.action_dim, mu=0.0, theta=ou_theta, sigma=ou_sigma
        )
        
        # Networks
        self.actor = DDPGActor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = DDPGCritic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.buffer = ReplayBuffer()

    def reset_ou_noise(self) -> None:
        self.ou_noise.reset()
    
    def select_action(self, state: np.ndarray, noise: float = 0.1) -> np.ndarray:
        """Select action; noise>0 时叠加 OU 噪声（noise 为幅度系数）。"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        
        if noise > 0:
            action = action + self.ou_noise.sample() * float(noise)
            action = np.clip(action, -1, 1)
        
        return action
    
    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """Update networks"""
        if len(self.buffer) < batch_size:
            return {}
        
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_q = self.critic(states, self.actor(states))
        actor_loss = -actor_q.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Compute Q value statistics
        q_mean = current_q.mean().item()
        q_std = current_q.std().item()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_value': q_mean,
            'q_mean': q_mean,  # 明确命名
            'q_std': q_std     # Q值标准差
        }


class PPOAgent:
    """Proximal Policy Optimization"""
    
    def __init__(self, state_dim: int, action_dim: int = 1,
                 lr: float = 3e-4, gamma: float = 0.99,
                 clip_range: float = 0.2, device: str = 'cuda',
                 entropy_coef: float = 0.03):
        self.device = device
        self.gamma = gamma
        self.clip_range = clip_range
        self.entropy_coef = float(entropy_coef)
        
        # Simple MLP policy
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        ).to(device)
        
        # 略宽初始方差，配合熵 bonus 增强探索
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5, device=device))
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)
        
        # Collect all parameters (log_std is already a Parameter, so it's a leaf tensor)
        params = list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_std]
        self.optimizer = optim.Adam(params, lr=lr)
        
        self.buffer = []
    
    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Select action and compute log probability"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean = self.actor(state)
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.cpu().numpy()[0], log_prob.cpu().item()
    
    def store_transition(self, state, action, reward, next_state, done, log_prob):
        """Store transition"""
        self.buffer.append((state, action, reward, next_state, done, log_prob))
    
    def update(self, n_epochs: int = 10) -> Dict[str, float]:
        """Update policy"""
        if len(self.buffer) == 0:
            return {}
        
        states, actions, rewards, next_states, dones, old_log_probs = zip(*self.buffer)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
        
        # Compute advantages
        with torch.no_grad():
            values = self.critic(states).squeeze()
            advantages = rewards - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        metrics = {}
        for epoch in range(n_epochs):
            # Compute new log probs
            mean = self.actor(states)
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            
            # PPO loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            values = self.critic(states).squeeze()
            critic_loss = F.mse_loss(values, rewards)

            entropy = dist.entropy().sum(dim=-1).mean()
            
            # Total loss（熵 bonus 鼓励探索）
            loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if epoch == n_epochs - 1:
                metrics = {
                    'actor_loss': actor_loss.item(),
                    'critic_loss': critic_loss.item(),
                    'advantage_mean': advantages.mean().item(),
                    'entropy_mean': entropy.item(),
                }
        
        self.buffer = []
        return metrics


class AdaptiveDR3L:
    """State-Adaptive Distributionally Robust Risk-Resilient RL (Adaptive-DR3L)
    
    Key Innovations:
    1. State-Adaptive Robustness: ρ(s) adapts to distribution stress level
    2. Adaptive Risk-Resilience Tradeoff: λ(s) balances CVaR vs Mean based on stress
    3. Distributional Quantile Critic: Full return distribution estimation
    4. Feature-Space Adversarial Perturbation: Robust to representation uncertainty
    
    Theoretical Framework:
    
    1. State-Adaptive Distributionally Robust Bellman Operator:
       T^{ρ(s)} Z = r + γ[(1-ρ(s))Z(s') + ρ(s)Z(s'+δ)]
       
       where ρ(s) = sigmoid(k * stress(s))
             stress(s) = std(Q(s,a)) - measures distribution uncertainty
       
       Intuition:
       - High stress (heavy tail, high uncertainty) → ρ(s) ↑ → more robust
       - Low stress (narrow distribution) → ρ(s) ↓ → less conservative
       
       This is a STATE-DEPENDENT Wasserstein robust operator, adapting robustness
       to local distribution characteristics rather than using fixed global ρ.
    
    2. Adaptive Risk-Resilience Actor Objective:
       maximize: λ(s) * CVaR_α(Q) + (1-λ(s)) * Mean(Q)
       
       where λ(s) = sigmoid(c * stress(s))
       
       Intuition:
       - High stress → λ(s) ↑ → focus on worst-case (CVaR)
       - Low stress → λ(s) ↓ → focus on average performance (Mean)
       
       This is DYNAMIC RISK-SENSITIVE policy optimization, automatically balancing
       risk aversion and performance based on state characteristics.
       
       Unlike fixed CVaR maximization, this achieves true resilience:
       - Risk-averse under uncertainty
       - Performance-oriented under stability
       - Adaptive to grid stress levels
    
    3. Stress Estimation:
       stress(s) = std(Q(s,a)) across quantile dimension
       
       Captures:
       - Distribution spread (epistemic uncertainty)
       - Tail heaviness (extreme event risk)
       - Value function confidence
       
       Alternative metrics:
       - |Mean(Q) - CVaR(Q)| - tail risk gap
       - Entropy of quantile distribution
       - Quantile range (max - min)
    
    4. Theoretical Advantages over Fixed DR3L:
       - Adapts to heterogeneous disturbances (light/medium/extreme)
       - Avoids over-conservatism in stable states
       - Avoids under-protection in high-risk states
       - Learns state-dependent risk preferences
       - More sample-efficient (less conservative exploration)
    
    Paper Contribution:
    "We propose State-Adaptive Distributionally Robust RL, where both the 
    robustness coefficient ρ(s) and risk-resilience tradeoff λ(s) adapt to 
    local distribution characteristics. This enables dynamic risk management 
    that is conservative under uncertainty but performance-oriented under 
    stability, achieving superior resilience in power grid applications."
    """
    
    def __init__(self, state_dim: int, action_dim: int = 1,
                 lr_actor: float = 3e-4, lr_critic: float = 3e-4,
                 gamma: float = 0.99, tau: float = 0.005,
                 cvar_alpha: float = 0.1,
                 rho_scale: float = 2.0,  # k in ρ(s) = sigmoid(k * stress)
                 lambda_scale: float = 2.0,  # c in λ(s) = sigmoid(c * stress)
                 epsilon: float = 0.01,
                 perturb_state_space: bool = False,
                 device: str = 'cuda',
                 off_policy: bool = True,
                 replay_capacity: int = 200000,
                 replay_learn_every: int = 4,
                 per_alpha: float = 0.6,
                 per_beta_start: float = 0.4,
                 per_beta_frames: int = 100000,
                 # ── Lagrangian 约束优化参数 ────────────────────────────────────────
                 # use_lagrangian_constraint=True 时启用新架构；False 时退回原始 DR3L（完全兼容）
                 use_lagrangian_constraint: bool = False,
                 lambda_init: float = 0.1,      # λ 初始值
                 lambda_lr: float = 1e-3,        # λ 对偶上升学习率
                 target_violation: float = 0.05, # 目标违规率 ε（约束允许上界）
                 ):
        """
        Args:
            cvar_alpha: CVaR confidence level (e.g., 0.1 for worst 10%)
            rho_scale: Scaling factor for adaptive robustness ρ(s) = sigmoid(k * stress)
            lambda_scale: Scaling factor for adaptive risk weight λ(s) = sigmoid(c * stress)
            epsilon: Adversarial perturbation magnitude
            tau: Target network soft update coefficient
            perturb_state_space: If True, perturb raw states; if False, perturb features
            off_policy: True 时用 replay 每步学习；False 时保留原 on-policy（episode 末 update）
            replay_capacity: 经验池容量
            replay_learn_every: off-policy 时每多少步环境步触发 1 次 learn（1=每步都学，极慢）
            per_alpha: PER 采样指数；per_beta_* 为重要性采样系数 beta 的线性退火
            use_lagrangian_constraint: True → Lagrangian 约束架构（reward 已解耦，λ learned）
            lambda_init: Lagrangian 乘子初始值
            lambda_lr: Lagrangian 乘子对偶上升学习率（λ += lr * (avg_cost - ε)）
            target_violation: 约束允许的目标违规率 ε（dual ascent 的参考基准）
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.cvar_alpha = cvar_alpha
        self.rho_scale = rho_scale
        self.lambda_scale = lambda_scale
        self.epsilon = epsilon
        self.perturb_state_space = perturb_state_space
        self.off_policy = bool(off_policy)
        self.replay = PrioritizedReplayBuffer(int(replay_capacity), alpha=per_alpha)
        self.replay_learn_every = max(1, int(replay_learn_every))
        self._global_learn_step = 0
        self._per_learn_count = 0
        self.per_beta_start = float(per_beta_start)
        self.per_beta_frames = max(1, int(per_beta_frames))
        self._last_learn_metrics: Dict[str, float] = {}

        # Feature extractor
        self.feature_net = MultiScaleFusionNet(
            cnn_input_channels=6,
            lstm_input_dim=2,
            fusion_dim=256
        ).to(device)
        
        # Actor
        self.actor = GaussianActor(feature_dim=256, action_dim=action_dim).to(device)
        
        # Quantile critic (online and target)
        self.critic = QuantileCritic(feature_dim=256, action_dim=action_dim, num_quantiles=51).to(device)
        self.critic_target = QuantileCritic(feature_dim=256, action_dim=action_dim, num_quantiles=51).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # feature_net 仅由 critic 优化，避免与 actor 双优化器争梯度；actor 前向用 detach 后的特征
        self.optimizer_actor = optim.Adam(list(self.actor.parameters()), lr=lr_actor)
        self.optimizer_feature_critic = optim.Adam(
            list(self.feature_net.parameters()) + list(self.critic.parameters()),
            lr=lr_critic,
        )
        
        self.buffer = []
        self.quantiles = torch.linspace(0.01, 0.99, 51).to(device)
        
        # State normalization bounds (for clipping adversarial perturbations)
        self.state_min = -3.0  # Assuming normalized states
        self.state_max = 3.0

        # ── Lagrangian 约束优化状态 ────────────────────────────────────────────
        # 优化目标重构：maximize E[return - risk]  s.t. E[violation] ≤ ε
        # Lagrangian：L = E[return - risk] - λ*(E[violation] - ε)
        # False → 原始 DR3L（兼容旧实验）；True → 启用 Lagrangian 约束架构
        self.use_lagrangian_constraint: bool = bool(use_lagrangian_constraint)
        self.lambda_constraint: float = float(lambda_init)   # λ（可学习对偶变量）
        self.lambda_lr: float = float(lambda_lr)              # 对偶上升步长
        self.target_violation: float = float(target_violation)  # ε（约束允许上界）
    
    def select_action(self, short_term: np.ndarray, long_term: np.ndarray) -> Tuple[np.ndarray, float]:
        """Select action"""
        short_term = torch.FloatTensor(short_term).unsqueeze(0).to(self.device)
        long_term = torch.FloatTensor(long_term).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features, _ = self.feature_net(short_term, long_term)
            action, log_prob = self.actor.sample(features)
        
        return action.cpu().numpy()[0], log_prob.cpu().item()
    
    def compute_stress(self, quantiles: torch.Tensor) -> torch.Tensor:
        """
        Compute distribution stress level from quantile estimates
        
        Stress measures the uncertainty/spread of the return distribution:
        - High stress: Heavy tail, high uncertainty, extreme event risk
        - Low stress: Narrow distribution, high confidence, stable conditions
        
        Args:
            quantiles: (batch, num_quantiles) - quantile estimates Q(s,a)
        
        Returns:
            stress: (batch, 1) - stress level per state
        
        Alternative stress metrics:
        1. Standard deviation (current): std(Q)
        2. Tail risk gap: |Mean(Q) - CVaR(Q)|
        3. Quantile range: max(Q) - min(Q)
        4. Distribution entropy: -Σ p_i log p_i
        """
        # Standard deviation across quantile dimension
        stress = quantiles.std(dim=1, keepdim=True)  # (batch, 1)
        return stress
    
    def compute_adaptive_rho(self, stress: torch.Tensor) -> torch.Tensor:
        """
        Compute state-adaptive robustness coefficient
        
        ρ(s) = sigmoid(k * stress(s))
        
        where k is rho_scale (default 2.0)
        
        Behavior:
        - stress → 0: ρ(s) → 0.5 (moderate robustness)
        - stress → ∞: ρ(s) → 1.0 (maximum robustness)
        - stress → -∞: ρ(s) → 0.0 (no robustness)
        
        Intuition:
        - High uncertainty states → more robust Bellman operator
        - Low uncertainty states → less conservative, better performance
        
        Args:
            stress: (batch, 1) - stress level
        
        Returns:
            rho: (batch, 1) - adaptive robustness coefficient
        """
        rho = torch.sigmoid(self.rho_scale * stress)
        return rho
    
    def compute_adaptive_lambda(self, stress: torch.Tensor) -> torch.Tensor:
        """
        Compute state-adaptive risk-resilience tradeoff weight
        
        λ(s) = sigmoid(c * stress(s))
        
        where c is lambda_scale (default 2.0)
        
        Behavior:
        - stress → 0: λ(s) → 0.5 (balanced)
        - stress → ∞: λ(s) → 1.0 (focus on CVaR, risk-averse)
        - stress → -∞: λ(s) → 0.0 (focus on Mean, performance-oriented)
        
        Intuition:
        - High stress → prioritize worst-case (CVaR)
        - Low stress → prioritize average performance (Mean)
        
        Args:
            stress: (batch, 1) - stress level
        
        Returns:
            lambda_s: (batch, 1) - adaptive risk weight
        """
        lambda_s = torch.sigmoid(self.lambda_scale * stress)
        return lambda_s
    
    def compute_adversarial_perturbation(self, features: torch.Tensor, 
                                        next_features: torch.Tensor,
                                        next_actions: torch.Tensor) -> torch.Tensor:
        """
        Compute adversarial perturbation for distributional robustness
        
        Objective: Find worst-case next state that minimizes Q-value
        δ* = argmin_{||δ||≤ε} CVaR(Q(s'+δ, a'))
        
        Approximated by gradient descent:
        δ = -ε * ∇_s CVaR(Q) / ||∇_s CVaR(Q)||
        
        IMPORTANT - Feature Space Perturbation:
        This operates in FEATURE SPACE (after CNN+LSTM encoding), not raw state space.
        
        Theoretical Justification:
        - Approximates Wasserstein ambiguity set in the learned feature manifold
        - Provides robustness to representation uncertainty
        - More efficient than state-space perturbation for high-dim observations
        
        For paper: Must state "We approximate the Wasserstein ambiguity set 
        in the learned feature manifold φ(s), where φ is the CNN+LSTM encoder."
        
        Alternative: Set perturb_state_space=True to perturb raw states (not implemented)
        
        Args:
            features: current state features (batch, feature_dim)
            next_features: next state features (batch, feature_dim) - MUST have gradients
            next_actions: next actions (batch, action_dim)
        
        Returns:
            perturbed_features: adversarially perturbed next state features
        """
        if self.epsilon == 0:
            return next_features.detach()
        
        # next_features must require grad for adversarial computation
        # This is called OUTSIDE torch.no_grad() scope
        assert next_features.requires_grad or next_features.grad_fn is not None, \
            "next_features must have gradients enabled for adversarial perturbation"
        
        # Compute quantiles for next state-action pair
        next_quantiles = self.critic_target(next_features, next_actions)
        
        # Compute CVaR (worst-case value)
        next_cvar = self.critic_target.compute_cvar(next_quantiles, alpha=self.cvar_alpha)
        
        # Compute gradient of CVaR w.r.t. next_features
        grad = torch.autograd.grad(
            outputs=next_cvar.sum(),
            inputs=next_features,
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Adversarial perturbation: move in direction that DECREASES CVaR (worst case)
        # Negative gradient to minimize Q-value
        grad_norm = torch.norm(grad, p=2, dim=-1, keepdim=True)
        delta = -self.epsilon * grad / (grad_norm + 1e-8)
        
        # Apply perturbation
        perturbed_features = next_features + delta
        
        return perturbed_features.detach()
    
    def update_critic(self, short_terms: torch.Tensor, long_terms: torch.Tensor,
                     actions: torch.Tensor, rewards: torch.Tensor,
                     next_short_terms: torch.Tensor, next_long_terms: torch.Tensor,
                     dones: torch.Tensor,
                     importance_weights: Optional[torch.Tensor] = None,
                     return_per_sample: bool = False) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
        """
        Update critic with STATE-ADAPTIVE distributionally robust target
        
        Key Innovation: ρ(s) adapts to distribution stress level
        
        Target: T^{ρ(s)} Z = r + γ[(1-ρ(s))Z(s') + ρ(s)Z(s'+δ)]
        
        where ρ(s) = sigmoid(k * stress(s))
              stress(s) = std(Q(s',a'))
        
        Intuition:
        - High stress (uncertain, heavy tail) → ρ(s) ↑ → more robust
        - Low stress (confident, narrow dist) → ρ(s) ↓ → less conservative
        
        This is a STATE-DEPENDENT Wasserstein robust operator.
        
        Uses full pairwise quantile regression loss for better distribution matching.
        """
        # Extract features
        features, _ = self.feature_net(short_terms, long_terms)
        
        # Compute current quantiles Q(s,a)
        current_quantiles = self.critic(features, actions)  # (batch, num_quantiles)
        
        # Compute target quantiles with STATE-ADAPTIVE adversarial robustness
        # Step 1: Extract next_features and compute next_actions (no grad for target)
        with torch.no_grad():
            next_features_no_grad, _ = self.feature_net(next_short_terms, next_long_terms)
            next_actions, _ = self.actor.sample(next_features_no_grad)
            
            # Compute nominal quantiles for stress estimation
            nominal_quantiles = self.critic_target(next_features_no_grad, next_actions)
            
            # Compute stress level from next state distribution
            stress = self.compute_stress(nominal_quantiles)  # (batch, 1)
            
            # Compute state-adaptive robustness coefficient
            rho_s = self.compute_adaptive_rho(stress)  # (batch, 1)
            # CRITICAL: Detach rho_s to prevent gradient leakage
            rho_s = rho_s.detach()
        
        # Step 2: Compute adversarial perturbation (OUTSIDE no_grad scope)
        # Only if epsilon > 0 (perturbation enabled)
        if self.epsilon > 0:
            # Re-extract next_features WITH gradients for perturbation
            next_features_with_grad, _ = self.feature_net(next_short_terms, next_long_terms)
            perturbed_next_features = self.compute_adversarial_perturbation(
                features.detach(), next_features_with_grad, next_actions
            )
        else:
            perturbed_next_features = None
        
        # Step 3: Compute targets with STATE-ADAPTIVE mixing (no grad)
        with torch.no_grad():
            # Nominal target: r + γZ(s')
            nominal_target = rewards.unsqueeze(1) + \
                           self.gamma * nominal_quantiles * (1 - dones.unsqueeze(1))
            
            # Adversarial target (if perturbation enabled)
            if self.epsilon > 0:
                # Adversarial target: r + γZ(s' + δ)
                # ASSUMPTION: Reward is deterministic and independent of state perturbation
                # i.e., r(s,a,s') ≈ r(s,a,s'+δ) for small δ in feature space
                adversarial_quantiles = self.critic_target(perturbed_next_features, next_actions)
                adversarial_target = rewards.unsqueeze(1) + \
                                   self.gamma * adversarial_quantiles * (1 - dones.unsqueeze(1))
                
                # STATE-ADAPTIVE convex robust mixing:
                # T^{ρ(s)} Z = (1-ρ(s))[r+γZ(s')] + ρ(s)[r+γZ(s'+δ)]
                # Broadcasting: rho_s is (batch, 1), targets are (batch, num_quantiles)
                target_quantiles = (1 - rho_s) * nominal_target + rho_s * adversarial_target
            else:
                target_quantiles = nominal_target
        
        # Full pairwise quantile regression loss
        # Expand dimensions for pairwise computation
        # current: (B, N) -> (B, N, 1)
        # target: (B, N) -> (B, 1, N)
        current_expanded = current_quantiles.unsqueeze(2)  # (B, N, 1)
        target_expanded = target_quantiles.unsqueeze(1)    # (B, 1, N)
        
        # TD errors: (B, N, N) - all pairwise differences
        td_errors = target_expanded - current_expanded
        
        # Quantile Huber loss
        huber_loss = torch.where(
            torch.abs(td_errors) <= 1.0,
            0.5 * td_errors ** 2,
            torch.abs(td_errors) - 0.5
        )
        
        # Quantile regression weights: |τ - I(δ < 0)|
        # quantiles: (N,) -> (1, N, 1)
        quantile_weights = torch.abs(
            self.quantiles.view(1, -1, 1) - (td_errors < 0).float()
        )
        
        # Weighted quantile loss: (B, N, N)
        quantile_loss = quantile_weights * huber_loss
        
        per_sample = quantile_loss.mean(dim=(1, 2))
        if importance_weights is not None:
            critic_loss = (per_sample * importance_weights).mean()
        else:
            critic_loss = per_sample.mean()
        
        self.optimizer_feature_critic.zero_grad()
        critic_loss.backward()
        # 记录裁剪前的梯度L2 Norm
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.feature_net.parameters()) + list(self.critic.parameters()),
            max_norm=1.0,
        )
        self.optimizer_feature_critic.step()
        
        # Soft update target network
        for param, target_param in zip(self.critic.parameters(), 
                                       self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        
        metrics = {
            'critic_loss': critic_loss.item(),
            'quantile_mean': current_quantiles.mean().item(),
            'quantile_std': current_quantiles.std().item(),
            'stress_mean': stress.mean().item(),
            'rho_mean': rho_s.mean().item(),
            'rho_min': rho_s.min().item(),
            'rho_max': rho_s.max().item(),
            'critic_grad_norm': float(critic_grad_norm),  # 记录裁剪前的梯度L2 Norm
        }
        per_np = per_sample.detach().cpu().numpy() if return_per_sample else None
        return metrics, per_np
    
    def update_actor(self, short_terms: torch.Tensor, long_terms: torch.Tensor,
                    actions: torch.Tensor, old_log_probs: torch.Tensor,
                    importance_weights: Optional[torch.Tensor] = None,
                    costs: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Update actor with ADAPTIVE RISK-RESILIENCE objective
        
        Key Innovation: λ(s) adapts to distribution stress level
        
        Objective: maximize λ(s) * CVaR_α(Q) + (1-λ(s)) * Mean(Q)
        
        where λ(s) = sigmoid(c * stress(s))
              stress(s) = std(Q(s,a))
        
        Intuition:
        - High stress (uncertain) → λ(s) ↑ → focus on CVaR (risk-averse)
        - Low stress (confident) → λ(s) ↓ → focus on Mean (performance-oriented)
        
        This is DYNAMIC RISK-SENSITIVE policy optimization, achieving true resilience:
        - Risk-averse under uncertainty
        - Performance-oriented under stability
        - Adaptive to grid stress levels
        
        Unlike fixed CVaR maximization, this balances worst-case and average performance.
        """
        features, _ = self.feature_net(short_terms, long_terms)
        features_for_actor = features.detach()

        # 仅更新 actor：特征已 detach；critic 参数在 actor 反传时关闭 requires_grad，避免
        # policy loss 写入 critic.grad（下一轮 critic 的 zero_grad 虽会清掉，但会浪费计算且易混淆）。
        critic_req: list = [p.requires_grad for p in self.critic.parameters()]
        try:
            for p in self.critic.parameters():
                p.requires_grad_(False)

            dist = self.actor.get_distribution(features_for_actor)
            new_actions = dist.rsample()

            quantiles = self.critic(features_for_actor, new_actions)

            stress = self.compute_stress(quantiles)
            lambda_s = self.compute_adaptive_lambda(stress)

            cvar = self.critic.compute_cvar(quantiles, alpha=self.cvar_alpha)
            mean_q = quantiles.mean(dim=-1)

            lambda_s_squeezed = lambda_s.squeeze(1)
            adaptive_objective = lambda_s_squeezed * cvar + (1 - lambda_s_squeezed) * mean_q

            # ── Lagrangian 约束惩罚项 ──────────────────────────────────────────
            # 当 use_lagrangian_constraint=True 时：actor 目标 = Q(risk-aware) - λ*C
            # 其中 C 是从 replay buffer 中采样的 per-step cost（constraint violation）
            # EMA risk 模块（adaptive_objective）保持不变，λ 仅作用于 cost 维度
            if self.use_lagrangian_constraint and costs is not None:
                lagrangian_penalty = self.lambda_constraint * costs
                adaptive_objective = adaptive_objective - lagrangian_penalty

            if importance_weights is not None:
                actor_loss = -(importance_weights * adaptive_objective).mean()
            else:
                actor_loss = -adaptive_objective.mean()

            entropy = dist.entropy().mean()
            actor_loss = actor_loss - 0.025 * entropy

            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            # 记录裁剪前的梯度L2 Norm
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.optimizer_actor.step()
        finally:
            for p, rg in zip(self.critic.parameters(), critic_req):
                p.requires_grad_(rg)
        
        actor_metrics: Dict[str, float] = {
            'actor_loss': actor_loss.item(),
            'cvar_mean': cvar.mean().item(),
            'mean_q': mean_q.mean().item(),
            'adaptive_objective': adaptive_objective.mean().item(),
            'entropy': entropy.item(),
            'stress_mean': stress.mean().item(),
            'lambda_mean': lambda_s.mean().item(),
            'lambda_min': lambda_s.min().item(),
            'lambda_max': lambda_s.max().item(),
            'actor_grad_norm': float(actor_grad_norm),
        }
        # Lagrangian 诊断指标（use_lagrangian_constraint=True 时记录）
        if self.use_lagrangian_constraint:
            actor_metrics['lagrangian/lambda_constraint'] = float(self.lambda_constraint)
            if costs is not None:
                actor_metrics['lagrangian/avg_cost_batch'] = float(costs.mean().item())
        return actor_metrics
    
    def learn_from_replay(self, batch_size: int = 256) -> Dict[str, float]:
        """PER 采样一批，执行 1 次 critic + 1 次 actor 更新（off-policy，带重要性采样权重）。"""
        if len(self.replay) < batch_size:
            return {}
        beta = min(
            1.0,
            self.per_beta_start
            + (1.0 - self.per_beta_start) * self._per_learn_count / self.per_beta_frames,
        )
        batch, indices, w_np = self.replay.sample(batch_size, beta=beta)
        self._per_learn_count += 1
        weights = torch.FloatTensor(w_np).to(self.device)

        short_terms = torch.FloatTensor(np.stack([b[0] for b in batch])).to(self.device)
        long_terms = torch.FloatTensor(np.stack([b[1] for b in batch])).to(self.device)
        actions = torch.FloatTensor(np.stack([b[2] for b in batch])).to(self.device)
        rewards = torch.FloatTensor([b[3] for b in batch]).to(self.device)
        old_log_probs = torch.FloatTensor([b[4] for b in batch]).to(self.device)
        next_short_terms = torch.FloatTensor(np.stack([b[5] for b in batch])).to(self.device)
        next_long_terms = torch.FloatTensor(np.stack([b[6] for b in batch])).to(self.device)
        dones = torch.FloatTensor([b[7] for b in batch]).to(self.device)
        # cost（索引 8）：当 use_lagrangian_constraint=True 时存储约束违规幅度
        # 向后兼容：旧格式 8 元组（无 cost 字段）时自动填 0
        costs = torch.FloatTensor(
            [b[8] if len(b) > 8 else 0.0 for b in batch]
        ).to(self.device)

        critic_metrics, per_sample = self.update_critic(
            short_terms, long_terms, actions, rewards,
            next_short_terms, next_long_terms, dones,
            importance_weights=weights,
            return_per_sample=True,
        )
        if per_sample is not None:
            self.replay.update_priorities(indices, per_sample)
        actor_metrics = self.update_actor(
            short_terms, long_terms, actions, old_log_probs,
            importance_weights=weights,
            costs=costs,
        )
        self._last_learn_metrics = {**critic_metrics, **actor_metrics}
        return self._last_learn_metrics

    def update(self, n_epochs: int = 10) -> Dict[str, float]:
        """Update networks（on-policy：整局 buffer；off_policy=True 时通常不用）。"""
        if self.off_policy:
            return dict(self._last_learn_metrics)
        if len(self.buffer) == 0:
            return {}
        
        # Unpack buffer
        short_terms, long_terms, actions, rewards, old_log_probs = [], [], [], [], []
        next_short_terms, next_long_terms, dones = [], [], []
        
        for transition in self.buffer:
            short_terms.append(transition[0])
            long_terms.append(transition[1])
            actions.append(transition[2])
            rewards.append(transition[3])
            old_log_probs.append(transition[4])
            next_short_terms.append(transition[5])
            next_long_terms.append(transition[6])
            dones.append(transition[7])
        
        short_terms = torch.FloatTensor(np.array(short_terms)).to(self.device)
        long_terms = torch.FloatTensor(np.array(long_terms)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
        next_short_terms = torch.FloatTensor(np.array(next_short_terms)).to(self.device)
        next_long_terms = torch.FloatTensor(np.array(next_long_terms)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        metrics = {}
        for epoch in range(n_epochs):
            critic_metrics, _ = self.update_critic(
                short_terms, long_terms, actions, rewards,
                next_short_terms, next_long_terms, dones,
            )

            actor_metrics = self.update_actor(
                short_terms, long_terms, actions, old_log_probs,
            )
            
            if epoch == n_epochs - 1:
                metrics.update(critic_metrics)
                metrics.update(actor_metrics)
        
        self.buffer = []
        return metrics
    
    def store_transition(self, short_term, long_term, action, reward, log_prob,
                        next_short_term, next_long_term, done,
                        cost: float = 0.0):
        """写入 replay（off-policy）；on-policy 模式同时写入 episode buffer。

        Args:
            cost: 当步约束违规幅度（来自 info["violation_magnitude"]）。
                  use_lagrangian_constraint=True 时由训练器传入；
                  False 时保持默认 0.0，不影响原有训练流程。
        """
        st = np.asarray(short_term, dtype=np.float32).copy()
        lt = np.asarray(long_term, dtype=np.float32).copy()
        act = np.asarray(action, dtype=np.float32).copy()
        nst = np.asarray(next_short_term, dtype=np.float32).copy()
        nlt = np.asarray(next_long_term, dtype=np.float32).copy()
        # 9 元组：新增 cost 字段（Lagrangian 约束）；旧代码传入 8 元组时通过 len(b)>8 兼容
        self.replay.append(
            (st, lt, act, float(reward), float(log_prob), nst, nlt, float(done), float(cost))
        )
        if self.off_policy:
            self._global_learn_step += 1
        if not self.off_policy:
            self.buffer.append(
                (short_term, long_term, action, reward, log_prob,
                 next_short_term, next_long_term, done)
            )


    def update_lambda(self, avg_cost: float) -> Dict[str, float]:
        """Lagrangian 对偶上升更新（Dual Ascent）。

        每个 episode 结束后调用：
            λ ← max(0, λ + lr * (avg_cost - ε))

        其中：
          avg_cost = E[violation_magnitude]（本 episode 平均约束违规幅度）
          ε = target_violation（允许的约束上界）
          λ ≥ 0（非负性保证 → 约束惩罚方向正确）

        当 use_lagrangian_constraint=False 时直接返回空字典（兼容旧实验）。
        """
        if not self.use_lagrangian_constraint:
            return {}
        old_lambda = self.lambda_constraint
        self.lambda_constraint += self.lambda_lr * (avg_cost - self.target_violation)
        self.lambda_constraint = max(0.0, self.lambda_constraint)
        return {
            'lagrangian/lambda_constraint': float(self.lambda_constraint),
            'lagrangian/lambda_delta': float(self.lambda_constraint - old_lambda),
            'lagrangian/avg_cost': float(avg_cost),
            'lagrangian/target_violation': float(self.target_violation),
            'lagrangian/constraint_gap': float(avg_cost - self.target_violation),
        }


# Backward compatibility: DR3L is now AdaptiveDR3L
# Old code using DR3L will automatically use the new adaptive version
DR3L = AdaptiveDR3L


# ---------------------------------------------------------------------------
# DR3L-Quantile ablation variant
# ---------------------------------------------------------------------------
class DR3LQuantile(AdaptiveDR3L):
    """Ablation: plain distributional critic (N=8 quantiles), no adaptive ρ/λ,
    no adversarial perturbation.  Actor maximises a *fixed* CVaR-mean blend.

    Shares the same MultiScaleFusionNet, GaussianActor, PER infrastructure,
    and train-loop interface as the full DR3L so that the only difference in
    experiment results comes from the critic architecture and objective.
    """

    def __init__(self, state_dim: int, action_dim: int = 1,
                 num_quantiles: int = 8,
                 cvar_alpha: float = 0.1,
                 fixed_lambda: float = 0.5,
                 device: str = 'cuda',
                 **kwargs):
        # Force: no adversarial perturbation, rho/lambda scales irrelevant
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            cvar_alpha=cvar_alpha,
            rho_scale=0.0,
            lambda_scale=0.0,
            epsilon=0.0,            # disable adversarial perturbation
            device=device,
            **kwargs,
        )
        self.fixed_lambda = float(fixed_lambda)
        N = int(num_quantiles)

        # Replace 51-quantile critics with N-quantile versions
        self.critic = QuantileCritic(
            feature_dim=256, action_dim=action_dim, num_quantiles=N,
        ).to(device)
        self.critic_target = QuantileCritic(
            feature_dim=256, action_dim=action_dim, num_quantiles=N,
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.quantiles = torch.linspace(0.01, 0.99, N).to(device)

        # Re-build joint optimizer with the new (smaller) critic
        self.optimizer_feature_critic = optim.Adam(
            list(self.feature_net.parameters()) + list(self.critic.parameters()),
            lr=kwargs.get('lr_critic', 3e-4),
        )

    # -- Override adaptive helpers to be constant ----------------------------
    def compute_adaptive_rho(self, stress: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(stress)       # no robust mixing

    def compute_adaptive_lambda(self, stress: torch.Tensor) -> torch.Tensor:
        return torch.full_like(stress, self.fixed_lambda)  # fixed blend
