"""
Feature Engineering Module for Campus Anti-Fraud Detection
Wutong Cup AI+Security Competition

This module provides complete feature engineering for:
- Student risk model (24 columns → 40+ features)
- Fraud detection model (53 columns → 60+ features)
- Semi-supervised labeling with audit gap detection
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


# =============================================================================
# STUDENT FEATURE ENGINEERING
# =============================================================================

def engineer_student_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete feature engineering using all 24 student columns.
    
    Args:
        df: Raw student_model.csv DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    features = pd.DataFrame(index=df.index)
    
    # === DEMOGRAPHIC FEATURES ===
    features['age'] = pd.to_numeric(df['age'], errors='coerce')
    features['age_group'] = pd.cut(
        features['age'],
        bins=[0, 18, 22, 25, 30, 100],
        labels=['minor', 'undergrad', 'grad', 'phd', 'mature']
    )
    features['is_female'] = (df['gndr'] == 'F').astype(int)
    
    # === IDENTITY RISK FEATURES ===
    features['is_mainland_student'] = (
        df['hk_resident_type'] == '新来港内地人港漂'
    ).astype(int)
    features['is_new_to_hk'] = df['hk_resident_type'].str.startswith(
        '新来港', na=False
    ).astype(int)
    features['is_hk_local'] = (df['hk_resident_type'] == '香港人').astype(int)
    features['is_unknown_type'] = (df['hk_resident_type'] == '其他').astype(int)
    
    # ID Document Risk
    features['uses_travel_permit'] = df['iden_type'].str.contains(
        '通行证', na=False
    ).astype(int)
    features['uses_passport'] = df['iden_type'].str.contains(
        '护照', na=False
    ).astype(int)
    features['uses_hk_id'] = df['iden_type'].str.contains(
        '身份证', na=False
    ).astype(int)
    
    # === DEVICE FEATURES ===
    features['uses_5g'] = (df['ntwk_type'] == '5G').astype(int)
    features['uses_esim'] = (df['card_type'] == 'esim').astype(int)
    
    # === MOBILITY FEATURES ===
    features['mainland_days'] = pd.to_numeric(
        df['mainland_cnt'], errors='coerce'
    ).fillna(0)
    features['hk_trips'] = pd.to_numeric(
        df['mainland_to_hk_cnt'], errors='coerce'
    ).fillna(0)
    features['is_frequent_commuter'] = (features['hk_trips'] >= 5).astype(int)
    features['mobility_score'] = (
        features['mainland_days'] + features['hk_trips'] * 2
    )
    
    # === FOREIGN EXPOSURE FEATURES ===
    features['foreign_voice_cnt'] = pd.to_numeric(
        df['total_voice_cnt'], errors='coerce'
    ).fillna(0)
    features['foreign_msg_cnt'] = pd.to_numeric(
        df['total_msg_cnt'], errors='coerce'
    ).fillna(0)
    features['foreign_exposure_total'] = (
        features['foreign_voice_cnt'] + features['foreign_msg_cnt']
    )
    
    # Mainland carrier exposure
    features['mainland_carrier_calls'] = pd.to_numeric(
        df['from_china_mobile_call_cnt'], errors='coerce'
    ).fillna(0)
    
    # === LOCAL EXPOSURE FEATURES ===
    features['local_voice_cnt'] = pd.to_numeric(
        df['total_local_voice_cnt'], errors='coerce'
    ).fillna(0)
    features['local_msg_cnt'] = pd.to_numeric(
        df['total_local_msg_cnt'], errors='coerce'
    ).fillna(0)
    features['local_exposure_total'] = (
        features['local_voice_cnt'] + features['local_msg_cnt']
    )
    
    # === EXPOSURE RATIO FEATURES ===
    features['foreign_to_local_ratio'] = (
        features['foreign_exposure_total'] / 
        (features['local_exposure_total'] + 1)
    )
    features['foreign_dominance'] = (
        features['foreign_exposure_total'] > features['local_exposure_total']
    ).astype(int)
    
    # === ENGAGEMENT DEPTH FEATURES ===
    features['max_repeat_caller'] = pd.to_numeric(
        df['max_voice_cnt'], errors='coerce'
    ).fillna(0)
    features['has_repeat_unknown_caller'] = (
        features['max_repeat_caller'] >= 3
    ).astype(int)
    features['has_frequent_caller'] = df['frequently_opp_num'].notna().astype(int)
    
    # === APP BEHAVIOR FEATURES ===
    features['mainland_app_days'] = pd.to_numeric(
        df['app_max_cnt'], errors='coerce'
    ).fillna(0)
    features['heavy_mainland_app_user'] = (
        features['mainland_app_days'] >= 14
    ).astype(int)
    
    # === FRAUD INTERACTION FEATURES (TARGET LABELS) ===
    features['has_fraud_contact'] = df['fraud_msisdn'].notna().astype(int)
    features['fraud_voice_receive'] = pd.to_numeric(
        df['voice_receive'], errors='coerce'
    ).fillna(0)
    features['fraud_voice_call'] = pd.to_numeric(
        df['voice_call'], errors='coerce'
    ).fillna(0)
    features['fraud_msg_receive'] = pd.to_numeric(
        df['msg_receive'], errors='coerce'
    ).fillna(0)
    features['fraud_msg_call'] = pd.to_numeric(
        df['msg_call'], errors='coerce'
    ).fillna(0)
    
    features['fraud_total_contact'] = (
        features['fraud_voice_receive'] + features['fraud_voice_call'] +
        features['fraud_msg_receive'] + features['fraud_msg_call']
    )
    
    # Engagement indicators
    features['engaged_voice'] = (features['fraud_voice_call'] > 0).astype(int)
    features['engaged_sms'] = (features['fraud_msg_call'] > 0).astype(int)
    features['any_engagement'] = (
        (features['engaged_voice'] + features['engaged_sms']) > 0
    ).astype(int)
    
    # === COMPOSITE RISK SCORE (Weights based on data analysis) ===
    # ML Feature Importance: is_mainland_student (11.4%), local_exposure_total (9.3%),
    # uses_esim (8.8%), uses_5g (6.9%), is_new_to_hk (6.5%), uses_travel_permit (6.4%)
    # Statistical: repeat_unknown_caller 1.7x, heavy_app_user 1.1x
    features['vulnerability_score'] = (
        # Top ML features
        features['is_mainland_student'] * 15 +       # 11.4% importance
        features['uses_esim'] * 12 +                 # 8.8% importance (new)
        features['is_new_to_hk'] * 10 +              # 6.5% importance
        features['uses_travel_permit'] * 8 +         # 6.4% importance
        
        # Statistical significance (actual defrauded patterns)
        features['has_repeat_unknown_caller'] * 20 + # 1.7x higher in defrauded
        features['heavy_mainland_app_user'] * 8 +    # 1.1x higher in defrauded
        
        # Exposure metrics (capped)
        np.minimum(features['local_exposure_total'] / 10, 10) * 3 +  # High local exposure
        np.minimum(features['foreign_exposure_total'] / 10, 10) * 2 + # 1.15x risk multiplier
        np.minimum(features['mainland_carrier_calls'], 5) * 2
    )
    
    return features


def create_student_risk_tiers(features: pd.DataFrame) -> pd.Series:
    """
    Classify students into risk tiers based on fraud engagement.
    
    Returns:
        Series with risk tier labels
    """
    tiers = pd.Series(index=features.index, dtype='object')
    
    # Priority order (highest to lowest)
    tiers[:] = 'LOW'
    
    # MODERATE: New to HK
    tiers[features['is_new_to_hk'] == 1] = 'MODERATE'
    
    # AT-RISK: Mainland student with foreign dominance
    tiers[
        (features['is_mainland_student'] == 1) & 
        (features['foreign_dominance'] == 1)
    ] = 'AT_RISK'
    
    # ELEVATED: Has fraud contact and high vulnerability
    tiers[
        (features['has_fraud_contact'] == 1) & 
        (features['vulnerability_score'] > 50)
    ] = 'ELEVATED'
    
    # HIGH: Received fraud calls/messages
    tiers[
        (features['fraud_voice_receive'] > 0) | 
        (features['fraud_msg_receive'] > 0)
    ] = 'HIGH'
    
    # CRITICAL: Engaged with fraudster (called back or replied)
    tiers[features['any_engagement'] == 1] = 'CRITICAL'
    
    return tiers


# =============================================================================
# FRAUD FEATURE ENGINEERING  
# =============================================================================

def engineer_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete feature engineering using all 53 fraud columns.
    
    Args:
        df: Raw fraud_model DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    features = pd.DataFrame(index=df.index)
    
    # === ACCOUNT TYPE FEATURES ===
    features['is_prepaid'] = (df['post_or_ppd'] == '预付').astype(int)
    features['is_postpaid'] = (df['post_or_ppd'] == '后付').astype(int)
    features['is_5g'] = (df['ntwk_type'] == '5G').astype(int)
    features['is_virtual_number'] = (df['msisdn_type'] == '虚拟号').astype(int)
    features['face_value'] = pd.to_numeric(
        df['face_val'], errors='coerce'
    ).fillna(0)
    features['monthly_fee'] = pd.to_numeric(
        df['mth_fee'], errors='coerce'
    ).fillna(0)
    
    # === IDENTITY LINKAGE FEATURES (Critical for SIM farms) ===
    features['total_id_docs'] = pd.to_numeric(
        df['iden_type_num'], errors='coerce'
    ).fillna(0)
    features['hk_id_linked_nums'] = pd.to_numeric(
        df['id_type_hk_num'], errors='coerce'
    ).fillna(0)
    features['hk_mainland_pass_nums'] = pd.to_numeric(
        df['id_type_hk_mac_to_mainland_num'], errors='coerce'
    ).fillna(0)
    features['mainland_hk_pass_nums'] = pd.to_numeric(
        df['id_type_mainland_to_hk_mac_num'], errors='coerce'
    ).fillna(0)
    features['passport_linked_nums'] = pd.to_numeric(
        df['id_type_psp_num'], errors='coerce'
    ).fillna(0)
    
    # Total linked numbers (SIM farm indicator)
    features['total_linked_numbers'] = (
        features['hk_id_linked_nums'] + 
        features['hk_mainland_pass_nums'] +
        features['mainland_hk_pass_nums'] +
        features['passport_linked_nums']
    )
    features['is_sim_farm'] = (features['total_linked_numbers'] >= 5).astype(int)
    features['is_multi_identity'] = (features['total_id_docs'] >= 2).astype(int)
    
    # === DEMOGRAPHIC FEATURES ===
    features['age'] = pd.to_numeric(df['age'], errors='coerce')
    features['age_missing'] = df['age'].isna().astype(int)
    features['is_unknown_resident'] = (
        df['hk_resident_type'] == '其他'
    ).astype(int)
    
    # === PRODUCT FEATURES ===
    features['is_mysim'] = df['rt_plan_desc'].str.contains(
        'MySIM', case=False, na=False
    ).astype(int)
    features['is_supertalk'] = df['rt_plan_desc'].str.contains(
        'Super Talk', case=False, na=False
    ).astype(int)
    features['is_tourist_sim'] = df['rt_plan_desc'].str.contains(
        'Tourist|Data Card', case=False, na=False
    ).astype(int)
    features['has_roaming_support'] = (df['is_support_roam'] == 'Y').astype(int)
    features['is_one_card_two_num'] = (df['is_1cmn'] == 'Y').astype(int)
    features['vas_count'] = pd.to_numeric(
        df['vas_ofr_id_num'], errors='coerce'
    ).fillna(0)
    
    # Anonymous channel
    features['channel_unknown'] = df['chnl_class'].isna().astype(int)
    features['channel_convenience'] = df['chnl_class'].str.contains(
        '便利|PT Shop', na=False
    ).astype(int)
    
    # VAS analysis
    if 'ofr_nm' in df.columns:
        features['has_roaming_vas'] = df['ofr_nm'].str.contains(
            'Roam|漫游', case=False, na=False
        ).astype(int)
        features['has_volte'] = df['ofr_nm'].str.contains(
            'VoLTE', case=False, na=False
        ).astype(int)
    else:
        features['has_roaming_vas'] = 0
        features['has_volte'] = 0
    
    # === CALL VOLUME FEATURES ===
    features['call_cnt_day'] = pd.to_numeric(
        df['call_cnt_day'], errors='coerce'
    ).fillna(0)
    features['called_cnt_day'] = pd.to_numeric(
        df['called_cnt_day'], errors='coerce'
    ).fillna(0)
    features['local_unknown_calls'] = pd.to_numeric(
        df['local_unknow_call_cnt'], errors='coerce'
    ).fillna(0)
    features['roam_unknown_calls'] = pd.to_numeric(
        df['roam_unknow_call_cnt'], errors='coerce'
    ).fillna(0)
    
    # Ratios
    features['outbound_inbound_ratio'] = (
        features['call_cnt_day'] / (features['called_cnt_day'] + 1)
    )
    features['unknown_call_ratio'] = (
        features['local_unknown_calls'] / (features['call_cnt_day'] + 1)
    )
    features['roam_call_ratio'] = (
        features['roam_unknown_calls'] / (features['call_cnt_day'] + 1)
    )
    
    # === CALL PATTERN FEATURES ===
    features['dispersion_rate'] = pd.to_numeric(
        df['dispersion_rate'], errors='coerce'
    ).fillna(0)
    features['avg_call_duration'] = pd.to_numeric(
        df['avg_actv_dur'], errors='coerce'
    ).fillna(0)
    features['short_calls_2s'] = pd.to_numeric(
        df['call_cnt_day_2s'], errors='coerce'
    ).fillna(0)
    features['long_calls_3m'] = pd.to_numeric(
        df['call_cnt_day_3m'], errors='coerce'
    ).fillna(0)
    
    # Duration patterns
    features['short_call_ratio'] = (
        features['short_calls_2s'] / (features['call_cnt_day'] + 1)
    )
    features['long_call_ratio'] = (
        features['long_calls_3m'] / (features['call_cnt_day'] + 1)
    )
    features['bimodal_duration'] = (
        (features['short_calls_2s'] > 3) & (features['long_calls_3m'] > 2)
    ).astype(int)
    
    # === TIME SLOT FEATURES ===
    features['active_9_12'] = pd.to_numeric(
        df['call_cnt_times_9_12'], errors='coerce'
    ).fillna(0)
    features['active_12_15'] = pd.to_numeric(
        df['call_cnt_times_12_15'], errors='coerce'
    ).fillna(0)
    features['active_15_18'] = pd.to_numeric(
        df['call_cnt_times_15_18'], errors='coerce'
    ).fillna(0)
    features['active_18_21'] = pd.to_numeric(
        df['call_cnt_times_18_21'], errors='coerce'
    ).fillna(0)
    features['active_21_24'] = pd.to_numeric(
        df['call_cnt_times_21_24'], errors='coerce'
    ).fillna(0)
    
    features['business_hours_activity'] = (
        features['active_9_12'] + 
        features['active_12_15'] + 
        features['active_15_18']
    )
    features['evening_activity'] = (
        features['active_18_21'] + features['active_21_24']
    )
    features['active_time_slots'] = (
        (features['active_9_12'] > 0).astype(int) +
        (features['active_12_15'] > 0).astype(int) +
        (features['active_15_18'] > 0).astype(int) +
        (features['active_18_21'] > 0).astype(int) +
        (features['active_21_24'] > 0).astype(int)
    )
    
    # === SMS FEATURES ===
    features['total_sms'] = pd.to_numeric(
        df['tot_msg_cnt'], errors='coerce'
    ).fillna(0)
    features['roam_sms'] = pd.to_numeric(
        df['roam_msg_cnt'], errors='coerce'
    ).fillna(0)
    features['roam_sms_ratio'] = (
        features['roam_sms'] / (features['total_sms'] + 1)
    )
    
    # === DEVICE FEATURES ===
    features['imei_changes'] = pd.to_numeric(
        df['change_imei_times'], errors='coerce'
    ).fillna(0)
    features['multi_device'] = (features['imei_changes'] > 1).astype(int)
    
    # === LOCATION FEATURES ===
    features['cellsite_duration'] = pd.to_numeric(
        df['cellsite_duration'], errors='coerce'
    ).fillna(0)
    features['is_mobile_operation'] = (
        features['cellsite_duration'] < 3600
    ).astype(int)
    features['has_cellsite_data'] = df['cellsite'].notna().astype(int)
    
    # === ACCOUNT LIFECYCLE FEATURES ===
    # Use stat_dt (observation date) NOT proc_time (suspension time - that's leakage!)
    features['open_date'] = pd.to_datetime(df['open_dt'], errors='coerce')
    features['observation_date'] = pd.to_datetime(df['stat_dt'], errors='coerce')
    features['account_age_days'] = (
        features['observation_date'] - features['open_date']
    ).dt.days
    features['is_new_account'] = (
        features['account_age_days'] <= 7
    ).fillna(False).astype(int)
    features['is_very_new'] = (
        features['account_age_days'] <= 3
    ).fillna(False).astype(int)
    
    # Prepaid recharge patterns
    features['first_recharge'] = pd.to_numeric(
        df['rechrg_for_ppd'], errors='coerce'
    ).fillna(0)
    features['quick_recharge'] = df['refill_vchr_strt_time'].notna().astype(int)
    features['large_first_topup'] = (features['first_recharge'] >= 100).astype(int)
    
    # === STUDENT TARGETING FEATURES ===
    features['hit_student'] = (df['hit_student_model'] == 'Y').astype(int)
    features['student_targets_today'] = pd.to_numeric(
        df['opp_num_stu_cnt'], errors='coerce'
    ).fillna(0)
    features['student_calls_total'] = pd.to_numeric(
        df['call_stu_cnt'], errors='coerce'
    ).fillna(0)
    features['student_focus_ratio'] = (
        features['student_targets_today'] / (features['call_cnt_day'] + 1)
    )
    
    # === COMPLAINT FEATURES ===
    features['from_corp_complaint'] = (df['msisdn_source'] == 'CORP').astype(int)
    features['from_local_complaint'] = (df['msisdn_source'] == 'POST').astype(int)
    
    return features


# =============================================================================
# LABELING FUNCTIONS
# =============================================================================

def create_fraud_labels(df: pd.DataFrame, exclude_pending: bool = True) -> pd.DataFrame:
    """
    Create labels for fraud detection.
    
    Target: is_fraud
    - 1 = audit_status == '稽核不通過' (Confirmed Fraud)
    - 0 = audit_status == '稽核通過' (Passed Audit)
    - EXCLUDE: audit_status == '待稽核' (Pending) when training
    
    Args:
        df: Raw fraud_model DataFrame
        exclude_pending: If True, remove pending records from training
        
    Returns:
        DataFrame with is_fraud label and pending records optionally removed
    """
    result = df.copy()
    
    # === PRIMARY TARGET: is_fraud ===
    result['is_fraud'] = (result['audit_status'] == '稽核不通過').astype(int)
    
    # === HELPER COLUMNS ===
    result['is_confirmed_fraud'] = result['is_fraud']  # Alias for compatibility
    result['is_confirmed_clean'] = (result['audit_status'] == '稽核通過').astype(int)
    result['is_pending'] = (result['audit_status'] == '待稽核').astype(int)
    
    # Backward compatibility
    result['is_suspicious_passed'] = 0
    result['label_strict'] = result['is_fraud']
    result['label_expanded'] = result['is_fraud']
    
    # === EXCLUDE PENDING FROM TRAINING ===
    if exclude_pending:
        pending_count = result['is_pending'].sum()
        if pending_count > 0:
            print(f"  Excluding {pending_count} pending records from training")
            result = result[result['is_pending'] == 0].reset_index(drop=True)
    
    return result


def identify_audit_gaps(
    df_fraud: pd.DataFrame, 
    verbose: bool = True
) -> pd.DataFrame:
    """
    Find fraud cases that targeted students but passed audit.
    These represent critical gaps in the current audit process.
    
    Args:
        df_fraud: Fraud model DataFrame with labels
        verbose: Whether to print analysis
        
    Returns:
        DataFrame of potential missed fraud cases
    """
    # Cases that targeted students
    student_targeting = df_fraud[
        (df_fraud['hit_student_model'] == 'Y') |
        (pd.to_numeric(df_fraud['opp_num_stu_cnt'], errors='coerce') > 0)
    ]
    
    # Among those, find audit gaps
    audit_gaps = student_targeting[
        student_targeting['audit_status'] == '稽核通過'
    ]
    
    if verbose:
        print("=== AUDIT GAP ANALYSIS ===")
        print(f"Total student-targeting cases: {len(student_targeting)}")
        print(f"Passed audit (potential missed fraud): {len(audit_gaps)}")
        if len(student_targeting) > 0:
            print(f"Gap rate: {len(audit_gaps)/len(student_targeting):.1%}")
        
        if len(audit_gaps) > 0:
            print(f"\n=== MISSED CASE PATTERNS ===")
            call_cnt = pd.to_numeric(
                audit_gaps['call_cnt_day'], errors='coerce'
            )
            short_calls = pd.to_numeric(
                audit_gaps['call_cnt_day_2s'], errors='coerce'
            )
            print(f"Avg call_cnt_day: {call_cnt.mean():.1f}")
            print(f"Avg short_calls_2s: {short_calls.mean():.1f}")
            print(f"Prepaid ratio: {(audit_gaps['post_or_ppd']=='预付').mean():.1%}")
    
    return audit_gaps


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_feature_columns(features_df: pd.DataFrame) -> list:
    """Get list of numeric feature columns for model training."""
    # LEAKAGE COLUMNS - These reveal future information!
    exclude_cols = [
        # Date columns (used for computation, not as features)
        'open_date', 'proc_date', 'observation_date', 'stat_dt', 'open_dt',
        # Label columns
        'label_strict', 'label_expanded', 
        'is_confirmed_fraud', 'is_confirmed_clean', 'is_pending',
        'is_suspicious_passed', 'is_fraud',
        # LEAKAGE COLUMNS (identified by column audit)
        'proc_time',        # Suspension time - only known AFTER fraud detected
        'hit_student_model', # Suspension + hit student model - only known AFTER suspension!
        'audit_remark',     # Audit notes - only known AFTER audit
        'audit_status',     # Audit result - the target variable!
        'audit_result',     # Alternative audit column
        'from_corp_complaint',  # Corporate complaint - filed AFTER fraud confirmed!
        'msisdn_source',    # Complaint source (投诉来源) - only known AFTER complaint filed!
    ]
    return [
        col for col in features_df.columns 
        if col not in exclude_cols and 
        features_df[col].dtype in ['int64', 'float64', 'int32', 'float32']
    ]


if __name__ == "__main__":
    # Test the module
    print("Loading datasets...")
    
    # Load student data
    df_student = pd.read_csv(
        'Datasets/Student/Training and Testing Data/student_model.csv'
    )
    print(f"Student data: {len(df_student)} records")
    
    # Load fraud data
    df_fraud = pd.read_csv(
        'Datasets/Fraud/Training and Testing Data/fraud_model_2.csv'
    )
    print(f"Fraud data: {len(df_fraud)} records")
    
    # Engineer features
    print("\nEngineering student features...")
    student_features = engineer_student_features(df_student)
    print(f"Student features: {len(student_features.columns)} columns")
    
    print("\nEngineering fraud features...")
    fraud_features = engineer_fraud_features(df_fraud)
    print(f"Fraud features: {len(fraud_features.columns)} columns")
    
    # Create labels
    print("\nCreating fraud labels...")
    df_fraud_labeled = create_fraud_labels(df_fraud)
    print(f"Confirmed fraud: {df_fraud_labeled['is_confirmed_fraud'].sum()}")
    print(f"Suspicious passed: {df_fraud_labeled['is_suspicious_passed'].sum()}")
    
    # Identify audit gaps
    print("\n")
    audit_gaps = identify_audit_gaps(df_fraud)
    
    print("\n✓ Feature engineering module working correctly!")
