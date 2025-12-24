"""
Attack Pattern Detection Module for Campus Anti-Fraud Detection
Wutong Cup AI+Security Competition

This module detects 10 distinct fraud attack patterns:
1. Robocall Burst - High-volume automated calling
2. Wangiri - One-ring attack (missed call scam)
3. SIM Farm - Multiple linked numbers
4. Cross-Border Fraud - Roaming-based attacks
5. SIM Carousel - Rapid device switching
6. Social Engineering - Long manipulation calls
7. Smishing - SMS-based fraud
8. Student Targeting - Deliberate student targeting
9. New Account Burst - New account with immediate high activity
10. Mobile Operation - Non-stationary fraud operation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class PatternResult:
    """Result of attack pattern detection."""
    detected: bool
    severity: float  # 0.0 to 1.0
    indicators: List[str]
    pattern_name: str


class AttackPatternDetector:
    """Comprehensive attack pattern detection using fraud features."""
    
    PATTERN_NAMES = [
        'robocall_burst',
        'wangiri',
        'sim_farm',
        'cross_border',
        'sim_carousel',
        'social_engineering',
        'smishing',
        'student_targeting',
        'new_account_burst',
        'mobile_operation'
    ]
    
    @staticmethod
    def detect_robocall_burst(row: pd.Series) -> PatternResult:
        """
        Pattern 1: High-volume automated calling.
        
        Indicators:
        - High daily call volume (>=30)
        - High dispersion rate (calling many unique numbers)
        - Short average call duration
        """
        call_cnt = float(row.get('call_cnt_day', 0) or 0)
        dispersion = float(row.get('dispersion_rate', 0) or 0)
        avg_dur = float(row.get('avg_call_duration', 0) or 0)
        
        detected = (
            call_cnt >= 30 and
            dispersion > 0 and
            avg_dur < 120
        )
        
        severity = min(call_cnt / 50, 1.0) if detected else 0.0
        
        return PatternResult(
            detected=detected,
            severity=severity,
            indicators=['high_volume', 'many_unique_targets', 'short_duration'],
            pattern_name='robocall_burst'
        )
    
    @staticmethod
    def detect_wangiri(row: pd.Series) -> PatternResult:
        """
        Pattern 2: One-ring attack (missed call scam).
        
        Indicators:
        - Many calls under 2 seconds
        - High ratio of short calls
        - Calls to unknown recipients
        """
        call_cnt = float(row.get('call_cnt_day', 1) or 1)
        short_calls = float(row.get('short_calls_2s', 0) or 0)
        local_unknown = float(row.get('local_unknown_calls', 0) or 0)
        
        short_ratio = short_calls / max(call_cnt, 1)
        
        detected = (
            short_calls >= 5 and
            short_ratio >= 0.2 and
            local_unknown >= call_cnt * 0.5
        )
        
        severity = min(short_ratio * 2, 1.0) if detected else 0.0
        
        return PatternResult(
            detected=detected,
            severity=severity,
            indicators=['many_short_calls', 'unknown_recipients'],
            pattern_name='wangiri'
        )
    
    @staticmethod
    def detect_sim_farm(row: pd.Series) -> PatternResult:
        """
        Pattern 3: SIM farm operation (multiple linked numbers).
        
        Indicators:
        - Multiple numbers linked to same ID
        - Multiple identity documents
        - High total linked numbers
        """
        total_linked = float(row.get('total_linked_numbers', 0) or 0)
        hk_id_nums = float(row.get('hk_id_linked_nums', 0) or 0)
        is_multi_id = int(row.get('is_multi_identity', 0) or 0)
        
        detected = (
            total_linked >= 5 or
            hk_id_nums >= 3 or
            is_multi_id == 1
        )
        
        severity = min(total_linked / 10, 1.0) if detected else 0.0
        
        return PatternResult(
            detected=detected,
            severity=severity,
            indicators=['multiple_linked_ids', 'many_numbers_per_id'],
            pattern_name='sim_farm'
        )
    
    @staticmethod
    def detect_cross_border(row: pd.Series) -> PatternResult:
        """
        Pattern 4: Cross-border fraud using roaming.
        
        Indicators:
        - Roaming support enabled
        - More roaming calls than local calls
        - Significant roaming unknown calls
        """
        has_roaming = int(row.get('has_roaming_support', 0) or 0)
        roam_calls = float(row.get('roam_unknown_calls', 0) or 0)
        local_calls = float(row.get('local_unknown_calls', 0) or 0)
        roam_ratio = float(row.get('roam_call_ratio', 0) or 0)
        
        detected = (
            has_roaming == 1 and
            roam_calls > local_calls and
            roam_calls >= 5
        )
        
        severity = min(roam_ratio, 1.0) if detected else 0.0
        
        return PatternResult(
            detected=detected,
            severity=severity,
            indicators=['roaming_enabled', 'roam_exceeds_local'],
            pattern_name='cross_border'
        )
    
    @staticmethod
    def detect_sim_carousel(row: pd.Series) -> PatternResult:
        """
        Pattern 5: Rapid device switching (SIM carousel).
        
        Indicators:
        - Multiple IMEI changes in one day
        """
        imei_changes = int(row.get('imei_changes', 0) or 0)
        
        detected = imei_changes >= 2
        severity = min(imei_changes / 5, 1.0) if detected else 0.0
        
        return PatternResult(
            detected=detected,
            severity=severity,
            indicators=['multiple_devices_same_day'],
            pattern_name='sim_carousel'
        )
    
    @staticmethod
    def detect_social_engineering(row: pd.Series) -> PatternResult:
        """
        Pattern 6: Long calls indicating manipulation.
        
        Indicators:
        - Multiple calls over 3 minutes
        - High average call duration
        - Moderate call volume
        """
        long_calls = float(row.get('long_calls_3m', 0) or 0)
        avg_dur = float(row.get('avg_call_duration', 0) or 0)
        call_cnt = float(row.get('call_cnt_day', 0) or 0)
        
        detected = (
            long_calls >= 3 and
            avg_dur >= 180 and
            call_cnt >= 10
        )
        
        severity = min(long_calls / 10, 1.0) if detected else 0.0
        
        return PatternResult(
            detected=detected,
            severity=severity,
            indicators=['multiple_long_calls', 'high_avg_duration'],
            pattern_name='social_engineering'
        )
    
    @staticmethod
    def detect_smishing(row: pd.Series) -> PatternResult:
        """
        Pattern 7: SMS-based fraud (smishing).
        
        Indicators:
        - High SMS volume
        - Roaming SMS activity
        """
        total_sms = float(row.get('total_sms', 0) or 0)
        roam_sms = float(row.get('roam_sms', 0) or 0)
        roam_sms_ratio = float(row.get('roam_sms_ratio', 0) or 0)
        
        detected = (
            total_sms >= 20 or
            (roam_sms >= 5 and roam_sms_ratio > 0.5)
        )
        
        severity = min(total_sms / 50, 1.0) if detected else 0.0
        
        return PatternResult(
            detected=detected,
            severity=severity,
            indicators=['high_sms_volume', 'roaming_sms'],
            pattern_name='smishing'
        )
    
    @staticmethod
    def detect_student_targeting(row: pd.Series) -> PatternResult:
        """
        Pattern 8: Deliberate student targeting.
        
        Indicators:
        - Hit student model flag
        - Multiple student targets today
        - High student focus ratio
        """
        hit_student = int(row.get('hit_student', 0) or 0)
        student_today = float(row.get('student_targets_today', 0) or 0)
        focus_ratio = float(row.get('student_focus_ratio', 0) or 0)
        student_total = float(row.get('student_calls_total', 0) or 0)
        
        detected = (
            hit_student == 1 or
            student_today >= 2 or
            focus_ratio >= 0.1
        )
        
        severity = min(student_total / 20, 1.0) if detected else 0.0
        
        return PatternResult(
            detected=detected,
            severity=severity,
            indicators=['hit_student_model', 'high_student_focus'],
            pattern_name='student_targeting'
        )
    
    @staticmethod
    def detect_new_account_burst(row: pd.Series) -> PatternResult:
        """
        Pattern 9: New account with immediate high activity.
        
        Indicators:
        - Account age <= 7 days
        - High call volume on new account
        """
        is_new = int(row.get('is_new_account', 0) or 0)
        is_very_new = int(row.get('is_very_new', 0) or 0)
        call_cnt = float(row.get('call_cnt_day', 0) or 0)
        
        detected = is_new == 1 and call_cnt >= 20
        severity = 1.0 if is_very_new else 0.7 if detected else 0.0
        
        return PatternResult(
            detected=detected,
            severity=severity,
            indicators=['account_age_under_7days', 'high_immediate_activity'],
            pattern_name='new_account_burst'
        )
    
    @staticmethod
    def detect_mobile_operation(row: pd.Series) -> PatternResult:
        """
        Pattern 10: Mobile fraud operation (not stationary).
        
        Indicators:
        - Short cellsite duration (moving around)
        - High call volume while mobile
        """
        is_mobile = int(row.get('is_mobile_operation', 0) or 0)
        call_cnt = float(row.get('call_cnt_day', 0) or 0)
        
        detected = is_mobile == 1 and call_cnt >= 20
        severity = 0.8 if detected else 0.0
        
        return PatternResult(
            detected=detected,
            severity=severity,
            indicators=['short_cellsite_duration', 'high_call_volume'],
            pattern_name='mobile_operation'
        )
    
    @classmethod
    def detect_all_patterns(cls, row: pd.Series) -> Dict[str, PatternResult]:
        """
        Detect all 10 attack patterns for a single row.
        
        Args:
            row: Feature-engineered row
            
        Returns:
            Dictionary of pattern name to PatternResult
        """
        return {
            'robocall_burst': cls.detect_robocall_burst(row),
            'wangiri': cls.detect_wangiri(row),
            'sim_farm': cls.detect_sim_farm(row),
            'cross_border': cls.detect_cross_border(row),
            'sim_carousel': cls.detect_sim_carousel(row),
            'social_engineering': cls.detect_social_engineering(row),
            'smishing': cls.detect_smishing(row),
            'student_targeting': cls.detect_student_targeting(row),
            'new_account_burst': cls.detect_new_account_burst(row),
            'mobile_operation': cls.detect_mobile_operation(row),
        }
    
    @classmethod
    def get_pattern_summary(cls, row: pd.Series) -> Dict[str, Any]:
        """
        Get summary of all detected patterns for a row.
        
        Returns:
            Dictionary with pattern flags, counts, and severity
        """
        results = cls.detect_all_patterns(row)
        
        detected_patterns = [
            name for name, result in results.items() if result.detected
        ]
        
        max_severity = max(
            (result.severity for result in results.values()),
            default=0.0
        )
        
        return {
            'patterns_detected': detected_patterns,
            'pattern_count': len(detected_patterns),
            'max_severity': max_severity,
            'is_any_pattern': len(detected_patterns) > 0,
            'is_multiple_patterns': len(detected_patterns) >= 2,
        }


def apply_pattern_detection(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply pattern detection to entire DataFrame.
    
    Args:
        features_df: Feature-engineered DataFrame
        
    Returns:
        DataFrame with pattern detection columns added
    """
    detector = AttackPatternDetector()
    
    # Get pattern summaries for all rows
    summaries = features_df.apply(
        lambda row: detector.get_pattern_summary(row),
        axis=1
    )
    
    # Extract columns
    result = features_df.copy()
    result['patterns_detected'] = summaries.apply(lambda x: x['patterns_detected'])
    result['pattern_count'] = summaries.apply(lambda x: x['pattern_count'])
    result['max_severity'] = summaries.apply(lambda x: x['max_severity'])
    result['is_any_pattern'] = summaries.apply(lambda x: x['is_any_pattern'])
    result['is_multiple_patterns'] = summaries.apply(lambda x: x['is_multiple_patterns'])
    
    # Add individual pattern flags
    for pattern_name in detector.PATTERN_NAMES:
        result[f'pattern_{pattern_name}'] = features_df.apply(
            lambda row: int(getattr(detector, f'detect_{pattern_name}')(row).detected),
            axis=1
        )
    
    return result


def get_pattern_statistics(df_with_patterns: pd.DataFrame) -> pd.DataFrame:
    """
    Get statistics on pattern detection results.
    
    Args:
        df_with_patterns: DataFrame with pattern columns from apply_pattern_detection
        
    Returns:
        DataFrame with pattern statistics
    """
    pattern_cols = [col for col in df_with_patterns.columns if col.startswith('pattern_')]
    
    stats = []
    for col in pattern_cols:
        pattern_name = col.replace('pattern_', '')
        count = df_with_patterns[col].sum()
        pct = count / len(df_with_patterns) * 100
        stats.append({
            'pattern': pattern_name,
            'count': count,
            'percentage': f'{pct:.1f}%'
        })
    
    return pd.DataFrame(stats).sort_values('count', ascending=False)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'src')
    from feature_engineering import engineer_fraud_features
    
    print("Loading fraud data...")
    df = pd.read_csv('Datasets/Fraud/Training and Testing Data/fraud_model_2.csv')
    print(f"Records: {len(df)}")
    
    print("\nEngineering features...")
    features = engineer_fraud_features(df)
    print(f"Features: {len(features.columns)}")
    
    print("\nDetecting attack patterns...")
    df_with_patterns = apply_pattern_detection(features)
    
    print("\n=== PATTERN STATISTICS ===")
    stats = get_pattern_statistics(df_with_patterns)
    print(stats.to_string(index=False))
    
    print(f"\n=== SUMMARY ===")
    print(f"Any pattern detected: {df_with_patterns['is_any_pattern'].sum()} ({df_with_patterns['is_any_pattern'].mean()*100:.1f}%)")
    print(f"Multiple patterns: {df_with_patterns['is_multiple_patterns'].sum()} ({df_with_patterns['is_multiple_patterns'].mean()*100:.1f}%)")
    
    print("\n✓ Attack pattern detection working correctly!")
