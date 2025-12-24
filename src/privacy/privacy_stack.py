"""
Privacy-Preserving Modules for Campus Anti-Fraud Detection
Wutong Cup AI+Security Competition

This module implements the three-layer privacy stack:
- Layer 1: Differential Privacy for Federated Learning
- Layer 2: DP-Protected Risk Scoring
- Layer 3: Secure MPC with DP Outputs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json

# Try to import diffprivlib
try:
    from diffprivlib.mechanisms import Laplace, Gaussian
    HAS_DIFFPRIVLIB = True
except ImportError:
    HAS_DIFFPRIVLIB = False
    print("Warning: diffprivlib not installed. Using basic DP implementation.")


# =============================================================================
# DIFFERENTIAL PRIVACY MECHANISMS
# =============================================================================

@dataclass
class PrivacyBudget:
    """Privacy budget configuration."""
    epsilon: float = 1.0      # Privacy loss parameter
    delta: float = 1e-5       # Probability of privacy breach
    max_grad_norm: float = 1.0  # Gradient clipping threshold


class BasicLaplace:
    """Basic Laplace mechanism when diffprivlib is not available."""
    
    def __init__(self, epsilon: float, sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        
    def randomise(self, value: float) -> float:
        """Add Laplace noise to value."""
        scale = self.sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise


class DPRiskScorer:
    """
    Apply Differential Privacy to model predictions before sharing.
    Used before MPC layer to protect individual risk scores.
    """
    
    def __init__(self, epsilon: float = 0.5):
        """
        Initialize DP scorer.
        
        Args:
            epsilon: Privacy budget for DP
        """
        self.epsilon = epsilon
        self.sensitivity = 1.0  # Risk scores are normalized to [0, 1]
        
        if HAS_DIFFPRIVLIB:
            self.mechanism = Laplace(epsilon=epsilon, sensitivity=self.sensitivity)
        else:
            self.mechanism = BasicLaplace(epsilon=epsilon, sensitivity=self.sensitivity)
    
    def privatize_risk_score(self, raw_score: float) -> float:
        """
        Add DP noise to individual risk score.
        
        Args:
            raw_score: Original risk score (0-1)
            
        Returns:
            DP-protected risk score
        """
        noisy_score = self.mechanism.randomise(raw_score)
        # Clip to valid range
        return max(0.0, min(1.0, noisy_score))
    
    def privatize_count(self, true_count: int, sensitivity: int = 1) -> int:
        """
        DP-protected count for aggregate reporting.
        
        Args:
            true_count: True count value
            sensitivity: Query sensitivity
            
        Returns:
            DP-protected count
        """
        scale = sensitivity / self.epsilon
        noisy = true_count + np.random.laplace(0, scale)
        return max(0, int(round(noisy)))
    
    def privatize_histogram(self, counts: dict) -> dict:
        """DP-protected histogram of risk categories."""
        return {
            category: self.privatize_count(count)
            for category, count in counts.items()
        }
    
    def privatize_scores_batch(self, scores: np.ndarray) -> np.ndarray:
        """Privatize a batch of risk scores."""
        return np.array([
            self.privatize_risk_score(score) for score in scores
        ])


class DPFederatedFraudModel:
    """
    Federated Learning with Local Differential Privacy.
    Combines Approach A (FL) + Approach B (DP).
    """
    
    def __init__(self, carriers: List[str], privacy: PrivacyBudget):
        """
        Initialize DP-FL model.
        
        Args:
            carriers: List of participating carriers
            privacy: Privacy budget configuration
        """
        self.carriers = carriers
        self.privacy = privacy
        self.global_model = None
        self.total_privacy_spent = 0.0
        self.rounds_completed = 0
        
    def clip_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """Clip gradients to bound sensitivity."""
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > self.privacy.max_grad_norm:
            gradients = gradients * (self.privacy.max_grad_norm / grad_norm)
        return gradients
    
    def add_gaussian_noise(self, gradients: np.ndarray) -> np.ndarray:
        """Add calibrated Gaussian noise for (ε, δ)-DP."""
        sensitivity = self.privacy.max_grad_norm
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.privacy.delta)) / self.privacy.epsilon
        noise = np.random.normal(0, sigma, gradients.shape)
        return gradients + noise
    
    def local_train_with_dp(self, gradients: np.ndarray) -> np.ndarray:
        """
        Apply DP before sharing gradients.
        
        Args:
            gradients: Raw gradients from local training
            
        Returns:
            DP-protected gradients
        """
        # Apply DP: clip then add noise
        clipped = self.clip_gradients(gradients)
        noisy_gradients = self.add_gaussian_noise(clipped)
        
        return noisy_gradients
    
    def federated_round(self, carrier_gradients: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Aggregate DP-protected gradients from all carriers.
        
        Args:
            carrier_gradients: Dictionary of carrier name to gradients
            
        Returns:
            Averaged gradients for global model update
        """
        all_gradients = list(carrier_gradients.values())
        
        # Secure aggregation
        averaged = np.mean(all_gradients, axis=0)
        
        # Track privacy budget
        self.total_privacy_spent += self.privacy.epsilon
        self.rounds_completed += 1
        
        return averaged
    
    def get_privacy_report(self) -> dict:
        """Report total privacy expenditure."""
        remaining_budget = max(0, 10.0 - self.total_privacy_spent)  # Total budget of ε=10
        return {
            'total_epsilon': self.total_privacy_spent,
            'delta': self.privacy.delta,
            'rounds_completed': self.rounds_completed,
            'rounds_remaining': int(remaining_budget / max(self.privacy.epsilon, 0.01)),
            'budget_remaining': remaining_budget
        }
    
    def simulate_federated_training(
        self,
        n_rounds: int = 5,
        n_features: int = 50
    ) -> Dict[str, Any]:
        """
        Simulate a federated training process for demonstration.
        
        Args:
            n_rounds: Number of training rounds
            n_features: Number of model features
            
        Returns:
            Simulation results
        """
        print(f"Simulating {n_rounds} rounds of federated training with {len(self.carriers)} carriers...")
        
        results = []
        for round_num in range(n_rounds):
            # Simulate local gradients from each carrier
            carrier_gradients = {
                carrier: np.random.randn(n_features) 
                for carrier in self.carriers
            }
            
            # Apply DP to each carrier's gradients
            dp_gradients = {
                carrier: self.local_train_with_dp(grads)
                for carrier, grads in carrier_gradients.items()
            }
            
            # Aggregate
            global_update = self.federated_round(dp_gradients)
            
            results.append({
                'round': round_num + 1,
                'gradient_norm': np.linalg.norm(global_update),
                'epsilon_spent': self.total_privacy_spent
            })
            
            print(f"  Round {round_num + 1}: ε spent = {self.total_privacy_spent:.2f}")
        
        return {
            'rounds': results,
            'privacy_report': self.get_privacy_report()
        }


class SecureStudentMatcherWithDP:
    """
    Private Set Intersection with DP-protected outputs.
    Combines Approach C (MPC) + Approach B (DP).
    """
    
    def __init__(self, output_epsilon: float = 0.1):
        """
        Initialize secure matcher.
        
        Args:
            output_epsilon: Privacy budget for output protection
        """
        self.output_epsilon = output_epsilon
        self.dp_scorer = DPRiskScorer(epsilon=output_epsilon)
        
    def private_set_intersection(
        self, 
        cmhk_risky_numbers: List[str], 
        university_student_numbers: List[str]
    ) -> Tuple[int, str]:
        """
        PSI with DP-protected output count.
        
        Note: This is a simulation. Real PSI would use cryptographic protocols.
        
        Args:
            cmhk_risky_numbers: High-risk numbers from carrier
            university_student_numbers: Student phone numbers
            
        Returns:
            Tuple of (DP-noisy count, risk level)
        """
        # Actual PSI computation (in real system, done in secure enclave)
        true_intersection = len(
            set(cmhk_risky_numbers) & set(university_student_numbers)
        )
        
        # Apply DP to output (protects exact count)
        dp_count = self.dp_scorer.privatize_count(
            true_intersection, 
            sensitivity=1
        )
        
        # Categorize risk level
        if dp_count >= 50:
            risk_level = "CRITICAL"
        elif dp_count >= 20:
            risk_level = "HIGH"
        elif dp_count >= 5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
            
        return dp_count, risk_level
    
    def generate_alerts(self, universities: Dict[str, Tuple[int, str]]) -> Dict[str, Dict]:
        """
        Generate DP-protected alerts for each university.
        
        Args:
            universities: Dict of university name to (count, risk_level)
            
        Returns:
            Dictionary of alerts for each university
        """
        alerts = {}
        for uni_name, (dp_count, risk_level) in universities.items():
            alerts[uni_name] = {
                'approximate_at_risk_students': dp_count,
                'risk_level': risk_level,
                'message': self._format_message(dp_count, risk_level),
                'privacy_guarantee': f'ε={self.output_epsilon}'
            }
        return alerts
    
    def _format_message(self, count: int, level: str) -> str:
        """Format alert message based on risk level."""
        if level == "CRITICAL":
            return f"URGENT: Approximately {count} students have been contacted by confirmed fraud numbers this week. Immediate awareness campaign recommended. (紧急：约{count}名学生本周已被确认诈骗号码联系，建议立即开展防诈宣传)"
        elif level == "HIGH":
            return f"WARNING: Approximately {count} students showing elevated fraud exposure. Consider targeted education. (警告：约{count}名学生显示高风险，建议针对性教育)"
        elif level == "MEDIUM":
            return f"NOTICE: Approximately {count} students with some fraud exposure detected. (注意：检测到约{count}名学生有一定诈骗暴露)"
        else:
            return f"INFO: {count} students with minimal fraud exposure. Continue monitoring. (信息：{count}名学生暴露风险较低，继续监控)"


class PrivacyAccountant:
    """
    Track and manage privacy budget across all three layers.
    Ensures total privacy loss stays within acceptable bounds.
    """
    
    def __init__(self, total_epsilon: float = 10.0, total_delta: float = 1e-4):
        """
        Initialize privacy accountant.
        
        Args:
            total_epsilon: Total privacy budget
            total_delta: Total delta budget
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.spent = {
            'federated_learning': 0.0,
            'risk_scoring': 0.0,
            'mpc_output': 0.0
        }
        self.log = []
        
    def allocate(self, layer: str, epsilon: float) -> bool:
        """
        Check if we can spend epsilon on this layer.
        
        Args:
            layer: Name of the layer
            epsilon: Epsilon to spend
            
        Returns:
            True if allocation successful, False if budget exceeded
        """
        current_total = sum(self.spent.values())
        if current_total + epsilon <= self.total_epsilon:
            self.spent[layer] += epsilon
            self.log.append({
                'layer': layer,
                'epsilon': epsilon,
                'cumulative': sum(self.spent.values())
            })
            return True
        return False
    
    def get_remaining(self) -> float:
        """Get remaining privacy budget."""
        return self.total_epsilon - sum(self.spent.values())
    
    def get_report(self) -> dict:
        """Get comprehensive privacy report."""
        return {
            'allocated': self.spent.copy(),
            'total_spent': sum(self.spent.values()),
            'remaining': self.get_remaining(),
            'total_budget': self.total_epsilon,
            'utilization': f"{sum(self.spent.values()) / self.total_epsilon * 100:.1f}%"
        }
    
    def save_log(self, path: str):
        """Save privacy log to file."""
        with open(path, 'w') as f:
            json.dump({
                'budget': self.get_report(),
                'log': self.log
            }, f, indent=2)
        print(f"Privacy log saved to {path}")


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_privacy_stack():
    """Demonstrate the three-layer privacy stack."""
    
    print("=" * 70)
    print("PRIVACY-PRESERVING ARCHITECTURE DEMONSTRATION")
    print("=" * 70)
    
    # Initialize privacy accountant
    accountant = PrivacyAccountant(total_epsilon=10.0)
    
    # Layer 1: Federated Learning with DP
    print("\n=== LAYER 1: FEDERATED LEARNING + DP ===")
    carriers = ['CMHK', 'CSL', '3HK', 'Smartone']
    privacy_config = PrivacyBudget(epsilon=1.0, delta=1e-5)
    
    fl_model = DPFederatedFraudModel(carriers, privacy_config)
    fl_results = fl_model.simulate_federated_training(n_rounds=3)
    
    # Track in accountant
    for round_info in fl_results['rounds']:
        accountant.allocate('federated_learning', privacy_config.epsilon)
    
    print(f"Privacy Report: {fl_model.get_privacy_report()}")
    
    # Layer 2: DP Risk Scoring
    print("\n=== LAYER 2: DP RISK SCORING ===")
    dp_scorer = DPRiskScorer(epsilon=0.5)
    accountant.allocate('risk_scoring', 0.5)
    
    # Simulate some risk scores
    raw_scores = [0.85, 0.32, 0.91, 0.15, 0.67]
    private_scores = [dp_scorer.privatize_risk_score(s) for s in raw_scores]
    
    print("Original scores:  ", [f"{s:.2f}" for s in raw_scores])
    print("Privatized scores:", [f"{s:.2f}" for s in private_scores])
    
    # Layer 3: Secure MPC with DP Output
    print("\n=== LAYER 3: SECURE MPC + DP OUTPUT ===")
    matcher = SecureStudentMatcherWithDP(output_epsilon=0.1)
    accountant.allocate('mpc_output', 0.1)
    
    # Simulate PSI
    risky_numbers = [f"852{i:08d}" for i in range(100)]
    student_numbers = [f"852{i:08d}" for i in range(30, 80)]  # 50 overlap
    
    dp_count, risk_level = matcher.private_set_intersection(
        risky_numbers, student_numbers
    )
    
    print(f"PSI Result: ~{dp_count} students at {risk_level} risk")
    
    # Generate university alert
    alerts = matcher.generate_alerts({
        'HKU': (dp_count, risk_level)
    })
    print(f"Alert: {alerts['HKU']['message'][:100]}...")
    
    # Final privacy report
    print("\n=== PRIVACY BUDGET SUMMARY ===")
    report = accountant.get_report()
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    return accountant


if __name__ == "__main__":
    accountant = demonstrate_privacy_stack()
    
    # Save privacy log
    accountant.save_log('Datasets/Fraud/Results/privacy_log.json')
    
    print("\n✓ Privacy modules demonstration complete!")
