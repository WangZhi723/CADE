"""
Test script for DR3L upgrade
Verifies CVaR computation and adversarial perturbation
"""

import torch
import numpy as np
from models import QuantileCritic
from algorithms import DR3L


def test_cvar_computation():
    """Test CVaR computation in QuantileCritic"""
    print("\n" + "="*80)
    print("Test 1: CVaR Computation")
    print("="*80)
    
    critic = QuantileCritic(feature_dim=256, action_dim=1, num_quantiles=51)
    
    # Create test features and actions
    batch_size = 32
    features = torch.randn(batch_size, 256)
    actions = torch.randn(batch_size, 1)
    
    # Compute quantiles Q(s,a)
    quantiles = critic(features, actions)
    print(f"✓ Quantiles shape: {quantiles.shape}")
    assert quantiles.shape == (batch_size, 51), "Quantiles shape mismatch"
    
    # Compute CVaR
    cvar = critic.compute_cvar(quantiles, alpha=0.1)
    print(f"✓ CVaR shape: {cvar.shape}")
    assert cvar.shape == (batch_size,), "CVaR shape mismatch"
    
    # Verify CVaR is mean of worst quantiles
    sorted_q, _ = torch.sort(quantiles, dim=-1)
    k = int(51 * 0.1)
    expected_cvar = sorted_q[:, :k].mean(dim=-1)
    assert torch.allclose(cvar, expected_cvar, atol=1e-6), "CVaR computation error"
    print(f"✓ CVaR computation verified")
    
    # Test VaR
    var = critic.compute_var(quantiles, alpha=0.1)
    print(f"✓ VaR shape: {var.shape}")
    assert var.shape == (batch_size,), "VaR shape mismatch"
    
    print("\n✅ CVaR computation test PASSED")
    return True


def test_adversarial_perturbation():
    """Test adversarial perturbation computation"""
    print("\n" + "="*80)
    print("Test 2: Adversarial Perturbation")
    print("="*80)
    
    # Create DR3L agent (now AdaptiveDR3L with adaptive rho)
    agent = DR3L(
        state_dim=10,
        action_dim=1,
        epsilon=0.01,
        rho_scale=2.0,  # Adaptive rho parameter
        device='cpu'
    )
    
    batch_size = 16
    feature_dim = 256
    action_dim = 1
    
    # Create test features and actions
    features = torch.randn(batch_size, feature_dim)
    # next_features MUST have gradients for adversarial perturbation
    next_features = torch.randn(batch_size, feature_dim, requires_grad=True)
    next_actions = torch.randn(batch_size, action_dim)
    
    # Compute adversarial perturbation
    perturbed_features = agent.compute_adversarial_perturbation(features, next_features, next_actions)
    
    print(f"✓ Perturbed features shape: {perturbed_features.shape}")
    assert perturbed_features.shape == next_features.shape, "Shape mismatch"
    
    # Verify perturbation magnitude
    perturbation = perturbed_features - next_features
    max_perturbation = torch.abs(perturbation).max().item()
    print(f"✓ Max perturbation: {max_perturbation:.6f}")
    
    # Verify perturbation is bounded
    assert max_perturbation <= agent.epsilon * 2, "Perturbation too large"
    
    # Test with epsilon=0 (no perturbation)
    agent_no_perturb = DR3L(state_dim=10, epsilon=0.0, rho_scale=0.0, device='cpu')
    next_features_no_perturb = torch.randn(batch_size, feature_dim)  # No grad needed when epsilon=0
    perturbed_no_change = agent_no_perturb.compute_adversarial_perturbation(
        features, next_features_no_perturb, next_actions
    )
    assert torch.allclose(perturbed_no_change, next_features_no_perturb), "Should have no perturbation"
    print(f"✓ No perturbation when epsilon=0")
    
    print("\n✅ Adversarial perturbation test PASSED")
    return True


def test_dr3l_update():
    """Test DR3L update with new interface"""
    print("\n" + "="*80)
    print("Test 3: DR3L Update (Adaptive)")
    print("="*80)
    
    agent = DR3L(
        state_dim=10,
        action_dim=1,
        cvar_alpha=0.1,
        epsilon=0.01,
        rho_scale=2.0,  # Adaptive rho
        lambda_scale=2.0,  # Adaptive lambda
        device='cpu'
    )
    
    # Create dummy transitions
    n_transitions = 64
    for i in range(n_transitions):
        short_term = np.random.randn(6, 12)
        long_term = np.random.randn(288, 2)
        action = np.random.randn(1)
        reward = np.random.randn()
        log_prob = np.random.randn()
        next_short_term = np.random.randn(6, 12)
        next_long_term = np.random.randn(288, 2)
        done = 0.0
        
        agent.store_transition(
            short_term, long_term, action, reward, log_prob,
            next_short_term, next_long_term, done
        )
    
    print(f"✓ Stored {len(agent.buffer)} transitions")
    assert len(agent.buffer) == n_transitions, "Buffer size mismatch"
    
    # Perform update
    metrics = agent.update(n_epochs=2)
    
    print(f"✓ Update completed")
    print(f"  Actor loss: {metrics.get('actor_loss', 0):.4f}")
    print(f"  Critic loss: {metrics.get('critic_loss', 0):.4f}")
    print(f"  CVaR mean: {metrics.get('cvar_mean', 0):.4f}")
    
    # Verify buffer is cleared
    assert len(agent.buffer) == 0, "Buffer should be cleared after update"
    print(f"✓ Buffer cleared after update")
    
    print("\n✅ DR3L update test PASSED")
    return True


def test_target_network_update():
    """Test target network soft update"""
    print("\n" + "="*80)
    print("Test 4: Target Network Update")
    print("="*80)
    
    agent = DR3L(state_dim=10, tau=0.005, device='cpu')
    
    # Get initial parameters
    critic_params = list(agent.critic.parameters())
    target_params_initial = [p.clone() for p in agent.critic_target.parameters()]
    
    # Verify initial sync
    for p, tp in zip(critic_params, target_params_initial):
        assert torch.allclose(p, tp), "Initial parameters should be synced"
    print(f"✓ Initial parameters synced")
    
    # Perform dummy update with enough samples
    for i in range(64):  # Increased from 10 to 64
        short_term = np.random.randn(6, 12)
        long_term = np.random.randn(288, 2)
        action = np.random.randn(1)
        reward = np.random.randn()
        log_prob = np.random.randn()
        next_short_term = np.random.randn(6, 12)
        next_long_term = np.random.randn(288, 2)
        done = 0.0
        
        agent.store_transition(
            short_term, long_term, action, reward, log_prob,
            next_short_term, next_long_term, done
        )
    
    agent.update(n_epochs=1)
    
    # Verify soft update occurred
    target_params_after = list(agent.critic_target.parameters())
    updated = False
    for tp_init, tp_after in zip(target_params_initial, target_params_after):
        if not torch.allclose(tp_init, tp_after, atol=1e-6):
            updated = True
            break
    
    assert updated, "Target should have updated"
    print(f"✓ Target network updated")
    
    # Verify soft update magnitude
    for p, tp in zip(critic_params, target_params_after):
        diff = torch.abs(p - tp).mean().item()
        # With tau=0.005, difference should be small but non-zero
        assert diff < 1.0, "Target should not differ too much"
    print(f"✓ Soft update magnitude reasonable")
    
    print("\n✅ Target network update test PASSED")
    return True


def test_cvar_vs_mean():
    """Compare CVaR-based vs mean-based objectives"""
    print("\n" + "="*80)
    print("Test 5: CVaR vs Mean Comparison")
    print("="*80)
    
    critic = QuantileCritic(feature_dim=256, action_dim=1, num_quantiles=51)
    
    # Create quantiles with known distribution
    batch_size = 100
    features = torch.randn(batch_size, 256)
    actions = torch.randn(batch_size, 1)
    quantiles = critic(features, actions)
    
    # Compute mean and CVaR
    mean_q = quantiles.mean(dim=-1)
    cvar_01 = critic.compute_cvar(quantiles, alpha=0.1)
    cvar_05 = critic.compute_cvar(quantiles, alpha=0.05)
    
    print(f"Mean Q: {mean_q.mean().item():.4f} ± {mean_q.std().item():.4f}")
    print(f"CVaR (α=0.10): {cvar_01.mean().item():.4f} ± {cvar_01.std().item():.4f}")
    print(f"CVaR (α=0.05): {cvar_05.mean().item():.4f} ± {cvar_05.std().item():.4f}")
    
    # CVaR should be lower than mean (more pessimistic)
    assert cvar_01.mean() < mean_q.mean(), "CVaR should be lower than mean"
    assert cvar_05.mean() < cvar_01.mean(), "Lower α should give lower CVaR"
    print(f"✓ CVaR is more pessimistic than mean")
    
    print("\n✅ CVaR vs Mean comparison test PASSED")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("DR3L UPGRADE TEST SUITE")
    print("="*80)
    
    tests = [
        ("CVaR Computation", test_cvar_computation),
        ("Adversarial Perturbation", test_adversarial_perturbation),
        ("DR3L Update", test_dr3l_update),
        ("Target Network Update", test_target_network_update),
        ("CVaR vs Mean", test_cvar_vs_mean),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:30s}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nDR3L upgrade is ready for experiments.")
        print("\nNext steps:")
        print("1. Run experiments: python run_experiments.py --experiment baselines")
        print("2. Compare CVaR-RL vs DR3L-full")
        print("3. Analyze robustness on distribution shift")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Please fix the issues before running experiments.")
    print("="*80)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
