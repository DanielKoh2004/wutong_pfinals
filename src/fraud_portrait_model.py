"""
Wire Fraud User Portrait Model
Task 2: Fraud Number Portrait and Student Reach Analysis

This module:
1. Analyzes fraud_model_2 behavior patterns (confirmed fraud)
2. Links fraud numbers with student data to find reach patterns
3. Creates fraud user archetypes and statistical portraits
4. Generates strategy summary for student protection
5. Applies patterns to identify suspected fraud in fraud_model_1 and _2
"""

import os
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
from datetime import datetime

from feature_engineering import engineer_fraud_features

# =============================================================================
# DATA LOADING AND ANALYSIS
# =============================================================================

def load_fraud_and_student_data():
    """Load fraud and student data for analysis."""
    print("=" * 70)
    print("TASK 2: WIRE FRAUD USER PORTRAIT MODEL")
    print("=" * 70)
    
    # Load fraud_model_2 (has audit results - confirmed fraud)
    fraud_2 = pd.read_csv('Datasets/Fraud/Training and Testing Data/fraud_model_2.csv')
    print(f"\nFraud Model 2: {len(fraud_2)} records")
    
    # Load student data for reach analysis
    students = pd.read_csv('Datasets/Student/Training and Testing Data/student_model.csv')
    print(f"Student Data: {len(students)} records")
    
    # Identify confirmed fraud
    confirmed_fraud = fraud_2[fraud_2['audit_status'] == '稽核不通過'].copy()
    passed_audit = fraud_2[fraud_2['audit_status'] == '稽核通過'].copy()
    print(f"\nConfirmed Fraud (稽核不通過): {len(confirmed_fraud)}")
    print(f"Passed Audit (稽核通過): {len(passed_audit)}")
    
    return fraud_2, students, confirmed_fraud, passed_audit


# =============================================================================
# STATISTICAL PORTRAIT
# =============================================================================

def build_fraud_statistical_portrait(confirmed: pd.DataFrame, passed: pd.DataFrame) -> Dict:
    """Build statistical portrait of confirmed fraud vs passed audit."""
    
    print("\n" + "=" * 70)
    print("STATISTICAL PORTRAIT OF WIRE FRAUD USERS")
    print("=" * 70)
    
    portrait = {}
    
    # === PREPAID VS POSTPAID ===
    print("\n=== PREPAID VS POSTPAID ===")
    prepaid_fraud = (confirmed['post_or_ppd'] == '预付').mean() * 100
    prepaid_passed = (passed['post_or_ppd'] == '预付').mean() * 100
    portrait['prepaid'] = {
        'fraud_pct': round(prepaid_fraud, 1),
        'passed_pct': round(prepaid_passed, 1),
        'risk_multiplier': round(prepaid_fraud / max(prepaid_passed, 0.1), 2)
    }
    print(f"Prepaid: Fraud={prepaid_fraud:.1f}%, Passed={prepaid_passed:.1f}%")
    print(f"  -> Prepaid is {prepaid_fraud/max(prepaid_passed, 0.1):.1f}x more likely in fraud")
    
    # === CALL VOLUME ===
    print("\n=== CALL VOLUME ===")
    call_fraud = pd.to_numeric(confirmed['call_cnt_day'], errors='coerce').mean()
    call_passed = pd.to_numeric(passed['call_cnt_day'], errors='coerce').mean()
    portrait['call_cnt_day'] = {
        'fraud_mean': round(call_fraud, 1),
        'passed_mean': round(call_passed, 1),
        'risk_multiplier': round(call_fraud / max(call_passed, 0.1), 2)
    }
    print(f"Calls/Day: Fraud={call_fraud:.1f}, Passed={call_passed:.1f}")
    
    # === ID DOCUMENT SHARING (SIM Farm indicator) ===
    print("\n=== ID DOCUMENT SHARING (SIM Farm) ===")
    id_fraud = pd.to_numeric(confirmed['id_type_hk_num'], errors='coerce').mean()
    id_passed = pd.to_numeric(passed['id_type_hk_num'], errors='coerce').mean()
    portrait['id_sharing'] = {
        'fraud_mean': round(id_fraud, 2),
        'passed_mean': round(id_passed, 2),
        'risk_multiplier': round(id_fraud / max(id_passed, 0.1), 2)
    }
    print(f"IDs per holder: Fraud={id_fraud:.2f}, Passed={id_passed:.2f}")
    
    sim_farm_fraud = (pd.to_numeric(confirmed['id_type_hk_num'], errors='coerce') >= 3).mean() * 100
    sim_farm_passed = (pd.to_numeric(passed['id_type_hk_num'], errors='coerce') >= 3).mean() * 100
    portrait['sim_farm'] = {
        'fraud_pct': round(sim_farm_fraud, 1),
        'passed_pct': round(sim_farm_passed, 1)
    }
    print(f"SIM Farm (3+ IDs): Fraud={sim_farm_fraud:.1f}%, Passed={sim_farm_passed:.1f}%")
    
    # === DISPERSION RATE ===
    print("\n=== CALL DISPERSION ===")
    disp_fraud = pd.to_numeric(confirmed['dispersion_rate'], errors='coerce').mean()
    disp_passed = pd.to_numeric(passed['dispersion_rate'], errors='coerce').mean()
    portrait['dispersion'] = {
        'fraud_mean': round(disp_fraud, 3),
        'passed_mean': round(disp_passed, 3)
    }
    print(f"Dispersion Rate: Fraud={disp_fraud:.3f}, Passed={disp_passed:.3f}")
    
    # === STUDENT TARGETING ===
    print("\n=== STUDENT TARGETING ===")
    student_fraud = (confirmed['hit_student_model'] == 'Y').mean() * 100
    student_passed = (passed['hit_student_model'] == 'Y').mean() * 100
    portrait['student_targeting'] = {
        'fraud_pct': round(student_fraud, 1),
        'passed_pct': round(student_passed, 1),
        'risk_multiplier': round(student_fraud / max(student_passed, 0.1), 2)
    }
    print(f"Targets Students: Fraud={student_fraud:.1f}%, Passed={student_passed:.1f}%")
    
    # === ROAMING ===
    print("\n=== ROAMING BEHAVIOR ===")
    roaming_fraud = (confirmed['is_support_roam'] == 'Y').mean() * 100
    roaming_passed = (passed['is_support_roam'] == 'Y').mean() * 100
    portrait['roaming'] = {
        'fraud_pct': round(roaming_fraud, 1),
        'passed_pct': round(roaming_passed, 1)
    }
    print(f"Has Roaming: Fraud={roaming_fraud:.1f}%, Passed={roaming_passed:.1f}%")
    
    # === ACCOUNT AGE ===
    print("\n=== ACCOUNT AGE ===")
    # Check if 'acct_age' or similar column exists
    age_col = 'acct_age' if 'acct_age' in confirmed.columns else None
    if age_col:
        new_fraud = (confirmed[age_col] == '新入网').mean() * 100
        new_passed = (passed[age_col] == '新入网').mean() * 100
    else:
        new_fraud = 0
        new_passed = 0
    portrait['new_account'] = {
        'fraud_pct': round(new_fraud, 1),
        'passed_pct': round(new_passed, 1)
    }
    print(f"New Account: Fraud={new_fraud:.1f}%, Passed={new_passed:.1f}%")
    
    return portrait


# =============================================================================
# FRAUD USER ARCHETYPES
# =============================================================================

def build_fraud_archetypes(confirmed: pd.DataFrame) -> List[Dict]:
    """Create fraud user archetypes based on behavior patterns."""
    
    print("\n" + "=" * 70)
    print("WIRE FRAUD USER ARCHETYPES")
    print("=" * 70)
    
    archetypes = []
    
    # === ARCHETYPE 1: SIM Farm Operator ===
    simfarm_mask = pd.to_numeric(confirmed['id_type_hk_num'], errors='coerce') >= 3
    simfarm = confirmed[simfarm_mask]
    
    if len(simfarm) > 0:
        archetypes.append({
            'name': 'SIM Farm Operator (SIM卡农场)',
            'description': 'Multiple SIM cards registered under same or few IDs',
            'count': len(simfarm),
            'pct_of_fraud': round(len(simfarm) / len(confirmed) * 100, 1),
            'characteristics': {
                'avg_ids': round(pd.to_numeric(simfarm['id_type_hk_num'], errors='coerce').mean(), 1),
                'prepaid_pct': round((simfarm['post_or_ppd'] == '预付').mean() * 100, 1),
                'avg_calls_day': round(pd.to_numeric(simfarm['call_cnt_day'], errors='coerce').mean(), 1)
            },
            'detection_rule': 'id_type_hk_num >= 3'
        })
        print(f"\n1. SIM FARM OPERATOR (SIM卡农场)")
        print(f"   Count: {len(simfarm)} ({len(simfarm)/len(confirmed)*100:.1f}% of fraud)")
        print(f"   Avg IDs: {pd.to_numeric(simfarm['id_type_hk_num'], errors='coerce').mean():.1f}")
    
    # === ARCHETYPE 2: High Volume Caller (Robocaller) ===
    call_cnt = pd.to_numeric(confirmed['call_cnt_day'], errors='coerce')
    robocall_mask = call_cnt >= 50
    robocall = confirmed[robocall_mask]
    
    if len(robocall) > 0:
        archetypes.append({
            'name': 'Robocaller (机器人拨打)',
            'description': 'Extremely high call volume per day',
            'count': len(robocall),
            'pct_of_fraud': round(len(robocall) / len(confirmed) * 100, 1),
            'characteristics': {
                'avg_calls_day': round(call_cnt[robocall_mask].mean(), 1),
                'max_calls_day': round(call_cnt[robocall_mask].max(), 1),
                'prepaid_pct': round((robocall['post_or_ppd'] == '预付').mean() * 100, 1)
            },
            'detection_rule': 'call_cnt_day >= 50'
        })
        print(f"\n2. ROBOCALLER (机器人拨打)")
        print(f"   Count: {len(robocall)} ({len(robocall)/len(confirmed)*100:.1f}% of fraud)")
        print(f"   Avg Calls/Day: {call_cnt[robocall_mask].mean():.1f}")
    
    # === ARCHETYPE 3: Student Targeter ===
    student_mask = confirmed['hit_student_model'] == 'Y'
    student_targeter = confirmed[student_mask]
    
    if len(student_targeter) > 0:
        archetypes.append({
            'name': 'Student Targeter (学生针对者)',
            'description': 'Specifically targets student phone numbers',
            'count': len(student_targeter),
            'pct_of_fraud': round(len(student_targeter) / len(confirmed) * 100, 1),
            'characteristics': {
                'prepaid_pct': round((student_targeter['post_or_ppd'] == '预付').mean() * 100, 1),
                'avg_calls_day': round(pd.to_numeric(student_targeter['call_cnt_day'], errors='coerce').mean(), 1)
            },
            'detection_rule': 'hit_student_model == "Y"'
        })
        print(f"\n3. STUDENT TARGETER (学生针对者)")
        print(f"   Count: {len(student_targeter)} ({len(student_targeter)/len(confirmed)*100:.1f}% of fraud)")
    
    # === ARCHETYPE 4: New Account Fraudster ===
    age_col = 'acct_age' if 'acct_age' in confirmed.columns else None
    new_fraudster = pd.DataFrame()  # Initialize empty
    if age_col:
        new_mask = confirmed[age_col] == '新入网'
        new_fraudster = confirmed[new_mask]
    
        if len(new_fraudster) > 0:
            archetypes.append({
                'name': 'New Account Fraudster (新账户欺诈)',
                'description': 'Uses newly registered accounts for fraud',
                'count': len(new_fraudster),
                'pct_of_fraud': round(len(new_fraudster) / len(confirmed) * 100, 1),
                'characteristics': {
                    'prepaid_pct': round((new_fraudster['post_or_ppd'] == '预付').mean() * 100, 1),
                    'avg_calls_day': round(pd.to_numeric(new_fraudster['call_cnt_day'], errors='coerce').mean(), 1)
                },
                'detection_rule': 'acct_age == "新入网"'
            })
            print(f"\n4. NEW ACCOUNT FRAUDSTER (新账户欺诈)")
            print(f"   Count: {len(new_fraudster)} ({len(new_fraudster)/len(confirmed)*100:.1f}% of fraud)")
    
    # === ARCHETYPE 5: Prepaid Fraud Operator ===
    prepaid_mask = confirmed['post_or_ppd'] == '预付'
    prepaid = confirmed[prepaid_mask]
    
    if len(prepaid) > 0:
        archetypes.append({
            'name': 'Prepaid Fraud Operator (预付费欺诈)',
            'description': 'Uses prepaid SIM cards to avoid tracing',
            'count': len(prepaid),
            'pct_of_fraud': round(len(prepaid) / len(confirmed) * 100, 1),
            'characteristics': {
                'sim_farm_pct': round((pd.to_numeric(prepaid['id_type_hk_num'], errors='coerce') >= 3).mean() * 100, 1),
                'avg_calls_day': round(pd.to_numeric(prepaid['call_cnt_day'], errors='coerce').mean(), 1)
            },
            'detection_rule': 'post_or_ppd == "预付"'
        })
        print(f"\n5. PREPAID FRAUD OPERATOR (预付费欺诈)")
        print(f"   Count: {len(prepaid)} ({len(prepaid)/len(confirmed)*100:.1f}% of fraud)")
    
    return archetypes


# =============================================================================
# STUDENT REACH ANALYSIS
# =============================================================================

def analyze_student_reach(confirmed: pd.DataFrame, students: pd.DataFrame) -> Dict:
    """Analyze how fraud numbers reach and affect students."""
    
    print("\n" + "=" * 70)
    print("STUDENT REACH ANALYSIS")
    print("=" * 70)
    
    reach_analysis = {}
    
    # === A: Which fraud numbers contact most students ===
    print("\n=== FRAUD NUMBERS TARGETING STUDENTS ===")
    student_targeting = confirmed[confirmed['hit_student_model'] == 'Y']
    total_targeting = len(student_targeting)
    total_fraud = len(confirmed)
    
    reach_analysis['targeting_rate'] = {
        'total_fraud': int(total_fraud),
        'targeting_students': int(total_targeting),
        'targeting_pct': round(total_targeting / total_fraud * 100, 1)
    }
    print(f"Fraud numbers targeting students: {total_targeting}/{total_fraud} ({total_targeting/total_fraud*100:.1f}%)")
    
    # === B: What patterns lead to student engagement ===
    print("\n=== PATTERNS IN STUDENT-TARGETING FRAUD ===")
    if len(student_targeting) > 0:
        non_student = confirmed[confirmed['hit_student_model'] != 'Y']
        
        # Compare key metrics
        patterns = {}
        for col, name in [
            ('call_cnt_day', 'Avg Calls/Day'),
            ('dispersion_rate', 'Dispersion Rate'),
            ('id_type_hk_num', 'ID Sharing')
        ]:
            student_val = pd.to_numeric(student_targeting[col], errors='coerce').mean()
            non_val = pd.to_numeric(non_student[col], errors='coerce').mean()
            patterns[name] = {
                'student_targeting': round(student_val, 2),
                'non_student': round(non_val, 2),
                'difference': round(student_val - non_val, 2)
            }
            print(f"  {name}: Student-targeting={student_val:.2f}, Other={non_val:.2f}")
        
        reach_analysis['engagement_patterns'] = patterns
    
    # === C: Time analysis (placeholder - would need actual time data) ===
    print("\n=== FRAUD TIME PATTERNS ===")
    # Since we don't have actual timestamp data, we note this for future enhancement
    reach_analysis['time_patterns'] = {
        'note': 'Requires timestamp data for detailed time analysis',
        'recommendation': 'Track fraud call timing for peak hour detection'
    }
    print("  (Requires timestamp data for detailed analysis)")
    
    # === Engagement success rate ===
    print("\n=== STUDENT ENGAGEMENT ANALYSIS ===")
    # Analyze students who engaged with fraud
    from feature_engineering import engineer_student_features
    student_features = engineer_student_features(students)
    engaged = student_features[student_features['any_engagement'] == 1]
    received = student_features[
        (student_features['fraud_voice_receive'] > 0) | 
        (student_features['fraud_msg_receive'] > 0)
    ]
    
    engagement_rate = len(engaged) / max(len(received), 1) * 100
    reach_analysis['engagement'] = {
        'students_contacted': int(len(received)),
        'students_engaged': int(len(engaged)),
        'engagement_rate': round(engagement_rate, 1)
    }
    print(f"  Students contacted: {len(received)}")
    print(f"  Students engaged: {len(engaged)}")
    print(f"  Engagement rate: {engagement_rate:.1f}%")
    
    return reach_analysis


# =============================================================================
# STRATEGY GENERATION
# =============================================================================

def generate_strategy(portrait: Dict, archetypes: List[Dict], reach: Dict) -> Dict:
    """Generate strategy summary for student protection."""
    
    print("\n" + "=" * 70)
    print("STRATEGY SUMMARY FOR STUDENT PROTECTION")
    print("=" * 70)
    
    strategy = {
        'technical_rules': [],
        'operational_guidance': [],
        'warning_criteria': []
    }
    
    # === A: Technical Rules for Interception ===
    print("\n=== TECHNICAL INTERCEPTION RULES ===")
    
    # Rule 1: Prepaid with high call volume
    if portrait['prepaid']['risk_multiplier'] > 1.5:
        rule = {
            'id': 'RULE_01',
            'name': 'High-Volume Prepaid',
            'condition': 'post_or_ppd == "预付" AND call_cnt_day > 30',
            'risk_weight': 25,
            'rationale': f"Prepaid is {portrait['prepaid']['risk_multiplier']}x more likely in fraud"
        }
        strategy['technical_rules'].append(rule)
        print(f"  {rule['id']}: {rule['name']} (+{rule['risk_weight']} points)")
    
    # Rule 2: SIM Farm detection
    rule = {
        'id': 'RULE_02',
        'name': 'SIM Farm Detection',
        'condition': 'id_type_hk_num >= 3',
        'risk_weight': 30,
        'rationale': 'Multiple SIMs under same ID indicates organized fraud'
    }
    strategy['technical_rules'].append(rule)
    print(f"  {rule['id']}: {rule['name']} (+{rule['risk_weight']} points)")
    
    # Rule 3: New account with high activity
    rule = {
        'id': 'RULE_03',
        'name': 'New Account High Activity',
        'condition': 'acct_age == "新入网" AND call_cnt_day > 20',
        'risk_weight': 20,
        'rationale': 'New accounts with high call volume are suspicious'
    }
    strategy['technical_rules'].append(rule)
    print(f"  {rule['id']}: {rule['name']} (+{rule['risk_weight']} points)")
    
    # Rule 4: Student targeting
    if portrait['student_targeting']['risk_multiplier'] > 1.2:
        rule = {
            'id': 'RULE_04',
            'name': 'Student Number Targeting',
            'condition': 'hit_student_model == "Y"',
            'risk_weight': 15,
            'rationale': f"Student targeting is {portrait['student_targeting']['risk_multiplier']}x in fraud"
        }
        strategy['technical_rules'].append(rule)
        print(f"  {rule['id']}: {rule['name']} (+{rule['risk_weight']} points)")
    
    # Rule 5: Robocall behavior
    rule = {
        'id': 'RULE_05',
        'name': 'Robocall Detection',
        'condition': 'call_cnt_day >= 50',
        'risk_weight': 25,
        'rationale': 'Extremely high call volume indicates automated dialing'
    }
    strategy['technical_rules'].append(rule)
    print(f"  {rule['id']}: {rule['name']} (+{rule['risk_weight']} points)")
    
    # === B: Operational Guidance ===
    print("\n=== OPERATIONAL GUIDANCE ===")
    
    guidance = [
        {
            'priority': 'HIGH',
            'action': 'Monitor prepaid card registrations with >3 IDs per holder',
            'frequency': 'Real-time',
            'rationale': 'SIM farms are primary fraud infrastructure'
        },
        {
            'priority': 'HIGH',
            'action': 'Flag new accounts with >20 calls/day within first week',
            'frequency': 'Daily',
            'rationale': 'New fraudulent accounts show high activity early'
        },
        {
            'priority': 'MEDIUM',
            'action': 'Cross-reference with student number database for targeting detection',
            'frequency': 'Weekly',
            'rationale': f"{reach['targeting_rate']['targeting_pct']}% of fraud targets students"
        },
        {
            'priority': 'MEDIUM',
            'action': 'Review passed audit cases with high-risk patterns',
            'frequency': 'Weekly',
            'rationale': 'Audit gaps exist where fraud patterns match but audit passed'
        }
    ]
    strategy['operational_guidance'] = guidance
    for g in guidance:
        print(f"  [{g['priority']}] {g['action']}")
    
    # === C: Warning Criteria for Students ===
    print("\n=== STUDENT WARNING CRITERIA ===")
    
    warning_criteria = [
        {
            'tier': 'CRITICAL',
            'condition': 'Student engaged with fraudster (called back/replied)',
            'action': 'Immediate personal outreach + security office notification',
            'count': reach['engagement']['students_engaged']
        },
        {
            'tier': 'HIGH',
            'condition': 'Student received calls from confirmed fraud numbers',
            'action': 'Push warning notification + fraud awareness resources',
            'count': reach['engagement']['students_contacted']
        },
        {
            'tier': 'ELEVATED',
            'condition': 'Mainland student with high foreign call exposure',
            'action': 'Targeted awareness campaign',
            'count': 'TBD'
        }
    ]
    strategy['warning_criteria'] = warning_criteria
    for w in warning_criteria:
        print(f"  [{w['tier']}] {w['condition']}")
    
    return strategy


# =============================================================================
# SUSPECTED FRAUD IDENTIFICATION
# =============================================================================

def identify_suspected_fraud(df: pd.DataFrame, strategy: Dict) -> pd.DataFrame:
    """Apply fraud rules to identify suspected fraud numbers."""
    
    print("\n" + "=" * 70)
    print("APPLYING FRAUD DETECTION RULES")
    print("=" * 70)
    
    # Calculate risk score for each record
    df = df.copy()
    df['fraud_risk_score'] = 0
    
    for rule in strategy['technical_rules']:
        rule_id = rule['id']
        weight = rule['risk_weight']
        condition = rule['condition']
        
        # Parse and apply condition
        if 'post_or_ppd == "预付"' in condition and 'call_cnt_day' in condition:
            mask = (df['post_or_ppd'] == '预付') & (pd.to_numeric(df['call_cnt_day'], errors='coerce') > 30)
        elif 'id_type_hk_num >= 3' in condition:
            mask = pd.to_numeric(df['id_type_hk_num'], errors='coerce') >= 3
        elif 'acct_age == "新入网"' in condition:
            if 'acct_age' in df.columns:
                mask = (df['acct_age'] == '新入网') & (pd.to_numeric(df['call_cnt_day'], errors='coerce') > 20)
            else:
                mask = pd.Series([False] * len(df))
        elif 'hit_student_model == "Y"' in condition:
            mask = df['hit_student_model'] == 'Y'
        elif 'call_cnt_day >= 50' in condition:
            mask = pd.to_numeric(df['call_cnt_day'], errors='coerce') >= 50
        else:
            continue
        
        df.loc[mask, 'fraud_risk_score'] += weight
        triggered = mask.sum()
        print(f"  {rule_id}: {triggered} records triggered (+{weight} points)")
    
    # Classify risk level
    df['fraud_risk_level'] = 'LOW'
    df.loc[df['fraud_risk_score'] >= 20, 'fraud_risk_level'] = 'MEDIUM'
    df.loc[df['fraud_risk_score'] >= 40, 'fraud_risk_level'] = 'HIGH'
    df.loc[df['fraud_risk_score'] >= 60, 'fraud_risk_level'] = 'CRITICAL'
    
    # Statistics
    print(f"\n  Risk Level Distribution:")
    for level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
        count = (df['fraud_risk_level'] == level).sum()
        pct = count / len(df) * 100
        print(f"    {level}: {count} ({pct:.1f}%)")
    
    return df


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_fraud_results(portrait, archetypes, reach, strategy, results_dir='Datasets/Fraud/Results'):
    """Save all fraud analysis results."""
    
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
    
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f'{results_dir}/fraud_portrait.json', 'w', encoding='utf-8') as f:
        json.dump(portrait, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    with open(f'{results_dir}/fraud_archetypes.json', 'w', encoding='utf-8') as f:
        json.dump(archetypes, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    with open(f'{results_dir}/student_reach_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(reach, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    with open(f'{results_dir}/fraud_strategy.json', 'w', encoding='utf-8') as f:
        json.dump(strategy, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)
    print(f"- {results_dir}/fraud_portrait.json")
    print(f"- {results_dir}/fraud_archetypes.json")
    print(f"- {results_dir}/student_reach_analysis.json")
    print(f"- {results_dir}/fraud_strategy.json")


# =============================================================================
# MAIN
# =============================================================================

def run_fraud_portrait_analysis():
    """Run complete fraud portrait analysis."""
    
    # Step 1: Load data
    fraud_2, students, confirmed, passed = load_fraud_and_student_data()
    
    # Step 2: Build statistical portrait
    portrait = build_fraud_statistical_portrait(confirmed, passed)
    
    # Step 3: Create archetypes
    archetypes = build_fraud_archetypes(confirmed)
    
    # Step 4: Analyze student reach
    reach = analyze_student_reach(confirmed, students)
    
    # Step 5: Generate strategy
    strategy = generate_strategy(portrait, archetypes, reach)
    
    # Step 6: Save results
    save_fraud_results(portrait, archetypes, reach, strategy)
    
    print("\n" + "=" * 70)
    print("TASK 2 COMPLETE: WIRE FRAUD USER PORTRAIT MODEL")
    print("=" * 70)
    
    return portrait, archetypes, reach, strategy


if __name__ == "__main__":
    run_fraud_portrait_analysis()
