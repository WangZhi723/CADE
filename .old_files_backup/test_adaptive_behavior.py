"""
Test Adaptive-DR3L Behavior
Verify that ρ(s) and λ(s) adapt to stress levels
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from algorithms import AdaptiveDR3L


def test_adaptive_coefficients():
    """Test that ρ(s) and λ(s) adapt to stress"""
    print("\n" + "="*80)
    print("Testing Adaptive Coefficients")
    print("="*80)
    
    agent = AdaptiveDR3L(
        state_dim=10,
        action_dim=1,
        rho_scale=2.0,
        lambda_scale=2.0,
        device='cpu'
    )
    
    # Create quantiles with different stress levels
    batch_size = 100
    num_quantiles = 51
    
    # Low stress: narrow distribution
    low_stress_quantiles = torch.randn(batch_size, num_quantiles) * 0.1
    
    # Medium stress: moderate distribution
    medium_stress_quantiles = torch.randn(batch_size, num_quantiles) * 0.5
    
    # High stress: wide distribution
    high_stress_quantiles = torch.randn(batch_size, num_quantiles) * 2.0
    
    # Compute stress levels
    stress_low = agent.compute_stress(low_stress_quantiles)
    stress_medium = agent.compute_stress(medium_stress_quantiles)
    stress_high = agent.compute_stress(high_stress_quantiles)
    
    print(f"\nStress Levels:")
    print(f"  Low stress:    {stress_low.mean().item():.4f}")
    print(f"  Medium stress: {stress_medium.mean().item():.4f}")
    print(f"  High stress:   {stress_high.mean().item():.4f}")
    
    # Compute adaptive rho
    rho_low = agent.compute_adaptive_rho(stress_low)
    rho_medium = agent.compute_adaptive_rho(stress_medium)
    rho_high = agent.compute_adaptive_rho(stress_high)
    
    print(f"\nAdaptive ρ(s):")
    print(f"  Low stress:    {rho_low.mean().item():.4f}")
    print(f"  Medium stress: {rho_medium.mean().item():.4f}")
    print(f"  High stress:   {rho_high.mean().item():.4f}")
    
    # Verify monotonicity: higher stress → higher rho
    assert rho_low.mean() < rho_medium.mean(), "ρ should increase with stress"
    assert rho_medium.mean() < rho_high.mean(), "ρ should increase with stress"
    print("✓ ρ(s) increases with stress (more robust under uncertainty)")
    
    # Compute adaptive lambda
    lambda_low = agent.compute_adaptive_lambda(stress_low)
    lambda_medium = agent.compute_adaptive_lambda(stress_medium)
    lambda_high = agent.compute_adaptive_lambda(stress_high)
    
    print(f"\nAdaptive λ(s):")
    print(f"  Low stress:    {lambda_low.mean().item():.4f}")
    print(f"  Medium stress: {lambda_medium.mean().item():.4f}")
    print(f"  High stress:   {lambda_high.mean().item():.4f}")
    
    # Verify monotonicity: higher stress → higher lambda (more CVaR focus)
    assert lambda_low.mean() < lambda_medium.mean(), "λ should increase with stress"
    assert lambda_medium.mean() < lambda_high.mean(), "λ should increase with stress"
    print("✓ λ(s) increases with stress (more risk-averse under uncertainty)")
    
    print("\n✅ Adaptive coefficient test PASSED")
    
    return {
        'stress': [stress_low.mean().item(), stress_medium.mean().item(), stress_high.mean().item()],
        'rho': [rho_low.mean().item(), rho_medium.mean().item(), rho_high.mean().item()],
        'lambda': [lambda_low.mean().item(), lambda_medium.mean().item(), lambda_high.mean().item()]
    }


def test_adaptive_objective():
    """Test that actor objective adapts to stress"""
    print("\n" + "="*80)
    print("Testing Adaptive Actor Objective")
    print("="*80)
    
    agent = AdaptiveDR3L(
        state_dim=10,
        action_dim=1,
        cvar_alpha=0.1,
        rho_scale=2.0,
        lambda_scale=2.0,
        device='cpu'
    )
    
    batch_size = 32
    num_quantiles = 51
    
    # Create quantiles with different stress levels
    low_stress_q = torch.randn(batch_size, num_quantiles) * 0.1 + 1.0
    high_stress_q = torch.randn(batch_size, num_quantiles) * 2.0 + 1.0
    
    # Compute stress
    stress_low = agent.compute_stress(low_stress_q)
    stress_high = agent.compute_stress(high_stress_q)
    
    # Compute adaptive lambda
    lambda_low = agent.compute_adaptive_lambda(stress_low).squeeze(1)
    lambda_high = agent.compute_adaptive_lambda(stress_high).squeeze(1)
    
    # Compute CVaR and Mean
    cvar_low = agent.critic.compute_cvar(low_stress_q, alpha=0.1)
    mean_low = low_stress_q.mean(dim=-1)
    
    cvar_high = agent.critic.compute_cvar(high_stress_q, alpha=0.1)
    mean_high = high_stress_q.mean(dim=-1)
    
    # Compute adaptive objectives
    obj_low = lambda_low * cvar_low + (1 - lambda_low) * mean_low
    obj_high = lambda_high * cvar_high + (1 - lambda_high) * mean_high
    
    print(f"\nLow Stress State:")
    print(f"  Stress: {stress_low.mean().item():.4f}")
    print(f"  λ(s):   {lambda_low.mean().item():.4f}")
    print(f"  CVaR:   {cvar_low.mean().item():.4f}")
    print(f"  Mean:   {mean_low.mean().item():.4f}")
    print(f"  Objective: {obj_low.mean().item():.4f}")
    print(f"  → More weight on Mean (performance-oriented)")
    
    print(f"\nHigh Stress State:")
    print(f"  Stress: {stress_high.mean().item():.4f}")
    print(f"  λ(s):   {lambda_high.mean().item():.4f}")
    print(f"  CVaR:   {cvar_high.mean().item():.4f}")
    print(f"  Mean:   {mean_high.mean().item():.4f}")
    print(f"  Objective: {obj_high.mean().item():.4f}")
    print(f"  → More weight on CVaR (risk-averse)")
    
    # Verify lambda increases with stress
    assert lambda_high.mean() > lambda_low.mean(), "λ should be higher for high stress"
    print("\n✓ Actor objective adapts to stress level")
    
    print("\n✅ Adaptive objective test PASSED")


def visualize_adaptive_functions():
    """Visualize ρ(s) and λ(s) as functions of stress"""
    print("\n" + "="*80)
    print("Visualizing Adaptive Functions")
    print("="*80)
    
    agent = AdaptiveDR3L(
        state_dim=10,
        action_dim=1,
        rho_scale=2.0,
        lambda_scale=2.0,
        device='cpu'
    )
    
    # Create range of stress values
    stress_values = torch.linspace(0.0, 3.0, 100).unsqueeze(1)
    
    # Compute adaptive coefficients
    rho_values = agent.compute_adaptive_rho(stress_values).squeeze().numpy()
    lambda_values = agent.compute_adaptive_lambda(stress_values).squeeze().numpy()
    stress_np = stress_values.squeeze().numpy()
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot ρ(s)
    ax1.plot(stress_np, rho_values, 'b-', linewidth=2)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='ρ=0.5')
    ax1.set_xlabel('Stress (std of Q)', fontsize=12)
    ax1.set_ylabel('ρ(s)', fontsize=12)
    ax1.set_title('Adaptive Robustness Coefficient', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 1])
    
    # Add annotations
    ax1.annotate('Low stress\n→ Less robust', xy=(0.5, 0.38), fontsize=10, ha='center')
    ax1.annotate('High stress\n→ More robust', xy=(2.5, 0.92), fontsize=10, ha='center')
    
    # Plot λ(s)
    ax2.plot(stress_np, lambda_values, 'r-', linewidth=2)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='λ=0.5')
    ax2.set_xlabel('Stress (std of Q)', fontsize=12)
    ax2.set_ylabel('λ(s)', fontsize=12)
    ax2.set_title('Adaptive Risk Weight', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 1])
    
    # Add annotations
    ax2.annotate('Low stress\n→ Focus on Mean', xy=(0.5, 0.38), fontsize=10, ha='center')
    ax2.annotate('High stress\n→ Focus on CVaR', xy=(2.5, 0.92), fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig('adaptive_functions.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to 'adaptive_functions.png'")
    
    print("\n✅ Visualization complete")


def main():
    """Run all adaptive behavior tests"""
    print("\n" + "="*80)
    print("ADAPTIVE-DR3L BEHAVIOR TEST SUITE")
    print("="*80)
    
    # Test 1: Adaptive coefficients
    results = test_adaptive_coefficients()
    
    # Test 2: Adaptive objective
    test_adaptive_objective()
    
    # Test 3: Visualization
    try:
        visualize_adaptive_functions()
    except Exception as e:
        print(f"⚠️  Visualization skipped: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\n✅ All adaptive behavior tests PASSED!")
    print("\nKey Findings:")
    print(f"  • ρ(s) adapts from {results['rho'][0]:.3f} to {results['rho'][2]:.3f}")
    print(f"  • λ(s) adapts from {results['lambda'][0]:.3f} to {results['lambda'][2]:.3f}")
    print(f"  • Both increase monotonically with stress")
    print("\nAdaptive-DR3L is ready for experiments! 🎉")
    print("="*80)


if __name__ == "__main__":
    main()
