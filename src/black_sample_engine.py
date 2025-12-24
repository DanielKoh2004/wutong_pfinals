"""
Unified Black Sample Rule Engine
Task 4: Black Sample Data Identification and Collection Scheme

This unified module provides:
1. Single rule system for both ML ensemble and Task 4 documentation
2. Data-driven rules with weighted scoring AND binary flagging
3. Automatic rule update mechanism
4. Privacy-preserving collection schemes
5. Comprehensive output (JSON, Markdown, Console)
"""

import os
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path

# =============================================================================
# UNIFIED RULE DATACLASS
# =============================================================================

@dataclass
class UnifiedRule:
    """A unified black sample identification rule."""
    rule_id: str
    name: str
    description: str
    weight: int  # Risk score weight (0-100)
    source: str  # Which task/analysis this rule came from
    confidence: float  # 0-1, based on statistical significance
    
    # Rule evaluation function will be added dynamically
    def __post_init__(self):
        self.triggered_count = 0
        self.true_positives = 0
        self.false_positives = 0


# =============================================================================
# UNIFIED BLACK SAMPLE ENGINE
# =============================================================================

class UnifiedBlackSampleEngine:
    """
    Single unified rule engine for black sample identification.
    
    Provides both:
    - Weighted risk scoring (for Task 4 documentation)
    - Binary flagging (for ML ensemble)
    """
    
    # =========================================================================
    # RULE DEFINITIONS - Single source of truth
    # =========================================================================
    
    RULES = {
        'RULE_01': {
            'name': 'High Volume Caller',
            'description': 'High daily call count with dispersion',
            'weight': 35,  # Increased: SHAP shows call_cnt_day is #4
            'source': 'Task 2: Fraud Portrait + SHAP',
            'confidence': 0.85
        },
        'RULE_02': {
            'name': 'SIM Farm (IMEI Switching)',
            'description': 'Multiple IMEI changes (>= 2 times)',
            'weight': 25,
            'source': 'Task 2: Fraud Portrait',
            'confidence': 0.85
        },
        'RULE_03': {
            'name': 'New Account Rapid Activity',
            'description': 'New account with immediate high activity',
            'weight': 30,  # SHAP: is_new_account is #5
            'source': 'Task 2: Fraud Archetypes + SHAP',
            'confidence': 0.80
        },
        'RULE_04': {
            'name': 'Device Fingerprint',
            'description': 'High calls on old GSM modems (not 5G)',
            'weight': 20,
            'source': 'Task 4: SIM Box Detection',
            'confidence': 0.75
        },
        'RULE_05': {
            'name': 'Student Targeting',
            'description': 'Deliberate targeting of student numbers',
            'weight': 20,
            'source': 'Task 2: Student Reach',
            'confidence': 0.75
        },
        'RULE_06': {
            'name': 'Burst Mode Calling',
            'description': 'High volume (>50 calls) with short duration (<10s avg)',
            'weight': 30,
            'source': 'Task 4: Robocall Pattern',
            'confidence': 0.85
        },
        'RULE_07': {
            'name': 'Anonymous Prepaid Burst',
            'description': 'Prepaid + unknown channel + high volume',
            'weight': 15,  # Decreased: is_prepaid is #10 in SHAP (low impact after split)
            'source': 'Task 2: Fraud Portrait + SHAP',
            'confidence': 0.80
        },
        'RULE_08': {
            'name': 'Multi Device',
            'description': 'Multiple device/IMEI changes',
            'weight': 15,
            'source': 'Task 4: Pattern Analysis',
            'confidence': 0.60
        },
        'RULE_09': {
            'name': 'Smishing Pattern',
            'description': 'High SMS volume or roaming SMS',
            'weight': 15,
            'source': 'Task 4: Pattern Analysis',
            'confidence': 0.65
        },
        'RULE_10': {
            'name': 'Robocall Detection',
            'description': 'Extremely high call volume (>=50/day)',
            'weight': 30,
            'source': 'Task 2: Robocaller Archetype',
            'confidence': 0.85
        },
        # RULE_11 (Corporate Complaint) REMOVED - it's leakage data (filed AFTER fraud confirmed)
    }
    
    # === ε BUDGET CONFIGURATION ===
    EPSILON_BUDGET = {
        'total': 10.0,          # Total privacy budget
        'per_rule': 0.5,        # ε spent per rule evaluation
        'per_collection': 1.0,  # ε spent per collection operation
        'per_output': 0.1,      # ε spent per output generation
    }
    
    def __init__(self, results_dir: str = 'Datasets/Fraud/Results'):
        self.results_dir = Path(results_dir)
        self.collection_stats = {
            'audit_feedback': 0,
            'student_complaints': 0,
            'cross_carrier': 0,
            'model_predictions': 0,
        }
        self.rules = [UnifiedRule(rule_id=k, **v) for k, v in self.RULES.items()]
        
        # === ε BUDGET ACCOUNTING ===
        self.epsilon_spent = {
            'rule_evaluation': 0.0,
            'collection': 0.0,
            'output': 0.0,
        }
        self.epsilon_remaining = self.EPSILON_BUDGET['total']
        self.epsilon_log = []
    
    def _spend_epsilon(self, operation: str, epsilon: float, description: str) -> bool:
        """Track epsilon spend and return True if budget available."""
        if self.epsilon_remaining < epsilon:
            self.epsilon_log.append({
                'operation': operation,
                'epsilon': epsilon,
                'description': description,
                'status': 'DENIED - Budget exhausted'
            })
            return False
        
        self.epsilon_spent[operation] = self.epsilon_spent.get(operation, 0) + epsilon
        self.epsilon_remaining -= epsilon
        self.epsilon_log.append({
            'operation': operation,
            'epsilon': epsilon,
            'description': description,
            'remaining': self.epsilon_remaining,
            'status': 'OK'
        })
        return True
    
    def get_epsilon_report(self) -> dict:
        """Get privacy budget report."""
        return {
            'total_budget': self.EPSILON_BUDGET['total'],
            'spent': sum(self.epsilon_spent.values()),
            'remaining': self.epsilon_remaining,
            'breakdown': self.epsilon_spent,
            'utilization': f"{sum(self.epsilon_spent.values()) / self.EPSILON_BUDGET['total'] * 100:.1f}%",
            'log_entries': len(self.epsilon_log)
        }
    
    # =========================================================================
    # RULE EVALUATION FUNCTIONS
    # =========================================================================
    
    def _rule_01(self, row: pd.Series) -> bool:
        """High Volume Caller"""
        call_cnt = float(row.get('call_cnt_day', 0) or 0)
        dispersion = float(row.get('dispersion_rate', 0) or 0)
        return call_cnt >= 25 and dispersion > 0
    
    def _rule_02(self, row: pd.Series) -> bool:
        """SIM Farm (IMEI Switching) - Any IMEI change is suspicious"""
        imei_changes = int(row.get('change_imei_times', 0) or 0)
        return imei_changes >= 1  # Relaxed: dataset has max 1
    
    def _rule_03(self, row: pd.Series) -> bool:
        """New Account Rapid Activity"""
        is_new = int(row.get('is_new_account', 0) or 0)
        call_cnt = float(row.get('call_cnt_day', 0) or 0)
        return is_new == 1 and call_cnt >= 15
    
    def _rule_04(self, row: pd.Series) -> bool:
        """Device Fingerprint - Detects old GSM modems/simboxes"""
        call_cnt = float(row.get('call_cnt_day', 0) or 0)
        ntwk_type = str(row.get('ntwk_type', '5G')).upper()
        return call_cnt > 30 and ntwk_type != '5G'
    
    def _rule_05(self, row: pd.Series) -> bool:
        """Student Targeting"""
        hit_student = str(row.get('hit_student_model', '')).upper() == 'Y'
        hit_student_alt = int(row.get('hit_student', 0) or 0) == 1
        return hit_student or hit_student_alt
    
    def _rule_06(self, row: pd.Series) -> bool:
        """Burst Mode Calling - High volume, short duration"""
        call_cnt = float(row.get('call_cnt_day', 0) or 0)
        avg_duration = float(row.get('avg_actv_dur', 999) or 999)
        return call_cnt > 50 and avg_duration < 10
    
    def _rule_07(self, row: pd.Series) -> bool:
        """Anonymous Prepaid Burst"""
        is_prepaid = int(row.get('is_prepaid', 0) or 0)
        channel_unknown = int(row.get('channel_unknown', 0) or 0)
        call_cnt = float(row.get('call_cnt_day', 0) or 0)
        return is_prepaid == 1 and channel_unknown == 1 and call_cnt >= 20
    
    def _rule_08(self, row: pd.Series) -> bool:
        """Multi Device"""
        imei_changes = int(row.get('imei_changes', 0) or 0)
        return imei_changes >= 2
    
    def _rule_09(self, row: pd.Series) -> bool:
        """Smishing Pattern"""
        total_sms = float(row.get('total_sms', 0) or 0)
        roam_sms = float(row.get('roam_sms', 0) or 0)
        return total_sms >= 30 or roam_sms >= 10
    
    def _rule_10(self, row: pd.Series) -> bool:
        """Robocall Detection"""
        call_cnt = float(row.get('call_cnt_day', 0) or 0)
        return call_cnt >= 50
    
    # _rule_11 (Corporate Complaint) REMOVED - leakage data
    
    def evaluate_row(self, row: pd.Series) -> Dict[str, Any]:
        """
        Evaluate all rules on a single row.
        
        Returns:
            Dict with:
            - is_flagged: True if ANY rule triggered
            - is_high_confidence: True if 2+ rules triggered
            - risk_score: Weighted sum of triggered rules
            - triggered_rules: List of triggered rule IDs
            - rule_details: Dict of rule_id -> triggered (bool)
        """
        rule_funcs = [
            ('RULE_01', self._rule_01),
            ('RULE_02', self._rule_02),
            ('RULE_03', self._rule_03),
            ('RULE_04', self._rule_04),
            ('RULE_05', self._rule_05),
            ('RULE_06', self._rule_06),
            ('RULE_07', self._rule_07),
            ('RULE_08', self._rule_08),
            ('RULE_09', self._rule_09),
            ('RULE_10', self._rule_10),
            # RULE_11 (Corporate Complaint) REMOVED - leakage data
        ]
        
        triggered = []
        risk_score = 0
        rule_details = {}
        
        for rule_id, func in rule_funcs:
            try:
                result = func(row)
            except:
                result = False
            
            rule_details[rule_id] = result
            if result:
                triggered.append(rule_id)
                risk_score += self.RULES[rule_id]['weight']
        
        return {
            'is_flagged': len(triggered) >= 1,
            'is_high_confidence': len(triggered) >= 2,
            'risk_score': risk_score,
            'rule_count': len(triggered),
            'triggered_rules': triggered,
            'rule_details': rule_details
        }
    
    # =========================================================================
    # DATASET-LEVEL FUNCTIONS
    # =========================================================================
    
    def apply_to_dataset(self, df: pd.DataFrame, dataset_name: str = "", silent: bool = False) -> pd.DataFrame:
        """
        Apply all rules to a dataset.
        
        Returns DataFrame with added columns:
        - is_flagged: Binary flag (for ensemble)
        - is_high_confidence: 2+ rules triggered
        - risk_score: Weighted score (for Task 4)
        - risk_level: LOW/MEDIUM/HIGH/CRITICAL
        - triggered_rules: List of triggered rule IDs
        """
        if not silent:
            label = f" ({dataset_name})" if dataset_name else ""
            print("\n" + "=" * 70)
            print(f"APPLYING UNIFIED BLACK SAMPLE RULES{label}")
            print("=" * 70)
        
        evaluations = df.apply(self.evaluate_row, axis=1)
        
        result = df.copy()
        result['is_flagged'] = evaluations.apply(lambda x: x['is_flagged'])
        result['is_high_confidence'] = evaluations.apply(lambda x: x['is_high_confidence'])
        result['risk_score'] = evaluations.apply(lambda x: x['risk_score'])
        result['rule_count'] = evaluations.apply(lambda x: x['rule_count'])
        result['triggered_rules'] = evaluations.apply(lambda x: x['triggered_rules'])
        
        # Risk levels
        result['risk_level'] = 'LOW'
        result.loc[result['risk_score'] >= 20, 'risk_level'] = 'MEDIUM'
        result.loc[result['risk_score'] >= 40, 'risk_level'] = 'HIGH'
        result.loc[result['risk_score'] >= 60, 'risk_level'] = 'CRITICAL'
        
        # Print stats (unless silent)
        if not silent:
            for rule_id in self.RULES.keys():
                triggered = evaluations.apply(lambda x: x['rule_details'].get(rule_id, False)).sum()
                print(f"  {rule_id}: {triggered:,} triggered (+{self.RULES[rule_id]['weight']} pts)")
            
            print(f"\n  Risk Level Distribution:")
            for level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
                count = (result['risk_level'] == level).sum()
                print(f"    {level}: {count:,} ({count/len(result)*100:.1f}%)")
        
        return result
    
    def validate_on_dataset(self, df: pd.DataFrame, labels: pd.Series) -> Dict:
        """Validate rule performance on labeled dataset."""
        df_scored = self.apply_to_dataset(df)
        df_scored['label'] = labels.values
        
        y_true = df_scored['label']
        y_pred = df_scored['is_flagged'].astype(int)
        
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 0.001)
        
        return {
            'total_records': len(df_scored),
            'total_fraud': int(y_true.sum()),
            'flagged': int(y_pred.sum()),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'recall': f'{recall*100:.1f}%',
            'precision': f'{precision*100:.1f}%',
            'f1_score': f'{f1*100:.1f}%',
        }
    
    # =========================================================================
    # COLLECTION MECHANISMS (Privacy-preserving)
    # =========================================================================
    
    def collect_from_audit(self, df: pd.DataFrame, audit_col: str = 'audit_status') -> int:
        """Collect from audit feedback."""
        if audit_col in df.columns:
            mask = df[audit_col].str.contains('不通過|failed', case=False, na=False)
            count = mask.sum()
            self.collection_stats['audit_feedback'] += count
            return count
        return 0
    
    def collect_from_mpc(self, intersection_count: int, epsilon: float = 0.5) -> int:
        """Collect from cross-carrier MPC with DP noise."""
        noise = np.random.laplace(0, 1/epsilon)
        noisy_count = max(0, int(intersection_count + noise))
        self.collection_stats['cross_carrier'] += noisy_count
        return noisy_count
    
    def collect_from_predictions(self, df: pd.DataFrame, min_score: int = 60) -> int:
        """Collect high-confidence predictions."""
        if 'risk_score' in df.columns:
            count = (df['risk_score'] >= min_score).sum()
            self.collection_stats['model_predictions'] += count
            return count
        return 0
    
    def collect_from_student_complaints(self, df: pd.DataFrame) -> int:
        """Collect from student complaints (hit_student_model column).
        
        Note: hit_student_model is excluded from ML training (leakage),
        but can be used for collection statistics as it indicates
        fraud numbers that were reported via student complaints.
        """
        if 'hit_student_model' in df.columns:
            count = (df['hit_student_model'].str.upper() == 'Y').sum()
            self.collection_stats['student_complaints'] += count
            return count
        return 0
    
    # =========================================================================
    # OUTPUT GENERATION
    # =========================================================================
    
    def save_outputs(self, output_dir: str = None):
        """Save rules as JSON and Markdown."""
        if output_dir is None:
            output_dir = self.results_dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Custom encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        # JSON
        data = {
            'version': '2.0',
            'generated_at': datetime.now().isoformat(),
            'total_rules': len(self.rules),
            'rules': [asdict(r) for r in self.rules],
            'collection_stats': {k: int(v) for k, v in self.collection_stats.items()}
        }
        with open(output_path / 'black_sample_rules.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        print(f"  [JSON] Saved to {output_path}/black_sample_rules.json")
        
        # Markdown
        md = ["# Unified Black Sample Rules", "",
              f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", "",
              "## Rules Summary", "",
              "| ID | Name | Weight | Confidence |",
              "|-----|------|--------|------------|"]
        for r in self.rules:
            md.append(f"| {r.rule_id} | {r.name} | {r.weight} | {r.confidence:.0%} |")
        md.extend(["", "## Rule Details", ""])
        for r in self.rules:
            md.extend([f"### {r.rule_id}: {r.name}", "",
                      f"- **Description:** {r.description}",
                      f"- **Weight:** {r.weight} points",
                      f"- **Source:** {r.source}", ""])
        
        with open(output_path / 'black_sample_rules.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(md))
        print(f"  [Markdown] Saved to {output_path}/black_sample_rules.md")
    
    def print_summary(self):
        """Print console summary."""
        print("\n" + "=" * 70)
        print("UNIFIED BLACK SAMPLE RULES SUMMARY")
        print("=" * 70)
        print(f"\nTotal Rules: {len(self.rules)}")
        print(f"Max Score: {sum(r.weight for r in self.rules)}")
        print("\nRules by Weight:")
        for r in sorted(self.rules, key=lambda x: x.weight, reverse=True)[:5]:
            print(f"  [{r.weight:2d} pts] {r.rule_id}: {r.name}")
        print("\nCollection Stats:")
        for k, v in self.collection_stats.items():
            print(f"  {k}: {v}")
        
        # === ε BUDGET REPORT ===
        report = self.get_epsilon_report()
        print("\n=== PRIVACY BUDGET (ε) REPORT ===")
        print(f"  Total Budget: {report['total_budget']} ε")
        print(f"  Spent: {report['spent']:.2f} ε")
        print(f"  Remaining: {report['remaining']:.2f} ε")
        print(f"  Utilization: {report['utilization']}")
        print(f"  Breakdown:")
        for op, eps in report['breakdown'].items():
            if eps > 0:
                print(f"    {op}: {eps:.2f} ε")


# =============================================================================
# COMPATIBILITY FUNCTIONS (for existing imports)
# =============================================================================

def apply_rules_to_dataset(features_df: pd.DataFrame, dataset_name: str = "", silent: bool = False) -> pd.DataFrame:
    """Apply rules to dataset (compatibility wrapper)."""
    engine = UnifiedBlackSampleEngine()
    return engine.apply_to_dataset(features_df, dataset_name, silent=silent)

def validate_rules_on_dataset(features_df: pd.DataFrame, labels: pd.Series) -> Dict:
    """Validate rules on dataset (compatibility wrapper)."""
    engine = UnifiedBlackSampleEngine()
    return engine.validate_on_dataset(features_df, labels)

def run_black_sample_engine(fraud_df: pd.DataFrame,
                            portrait: Dict,
                            archetypes: List[Dict],
                            student_reach: Dict) -> UnifiedBlackSampleEngine:
    """Run complete black sample pipeline for Task 4."""
    print("=" * 70)
    print("TASK 4: BLACK SAMPLE IDENTIFICATION RULES")
    print("=" * 70)
    
    engine = UnifiedBlackSampleEngine()
    
    # Apply rules (for Task 4 documentation)
    df_scored = engine.apply_to_dataset(fraud_df, "TASK 4 - ALL DATA")
    
    # Collect samples
    print("\n--- Collection Mechanisms ---")
    audit_count = engine.collect_from_audit(fraud_df)
    print(f"  Audit feedback: {audit_count}")
    student_count = engine.collect_from_student_complaints(fraud_df)
    print(f"  Student complaints: {student_count}")
    mpc_count = engine.collect_from_mpc(150, 0.5)
    print(f"  Cross-carrier MPC: {mpc_count}")
    pred_count = engine.collect_from_predictions(df_scored)
    print(f"  Model predictions: {pred_count}")
    
    # Save outputs
    print("\n--- Saving Outputs ---")
    engine.save_outputs()
    
    # Print summary
    engine.print_summary()
    
    print("\n" + "=" * 70)
    print("TASK 4 COMPLETE")
    print("=" * 70)
    
    return engine


if __name__ == "__main__":
    print("Testing Unified Black Sample Engine...")
    
    # Load test data
    fraud_df = pd.read_csv('Datasets/Fraud/Training and Testing Data/fraud_model_2.csv')
    
    # Run engine
    engine = UnifiedBlackSampleEngine()
    df_scored = engine.apply_to_dataset(fraud_df)
    
    print("\n[OK] Unified engine working!")
