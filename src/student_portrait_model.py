"""
High-Risk Student Portrait Model
Task 1: Wire Fraud Student Risk Identification

This module:
1. Analyzes characteristics of students contacted by fraud numbers
2. Builds statistical portraits and archetypes
3. Creates ML classification model for risk identification
4. Generates interpretable rules based on data analysis
"""

import os
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json

from feature_engineering import engineer_student_features

# =============================================================================
# DATA LOADING AND ANALYSIS
# =============================================================================

def load_and_analyze_students():
    """Load student data and identify defrauded students."""
    print("=" * 70)
    print("TASK 1: HIGH-RISK STUDENT PORTRAIT MODEL")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv('Datasets/Student/Training and Testing Data/student_model.csv')
    features = engineer_student_features(df)
    
    print(f"\nTotal students: {len(df):,}")
    
    # Target: Students who RECEIVED fraud calls/messages
    defrauded_mask = (
        (features['fraud_voice_receive'] > 0) |
        (features['fraud_msg_receive'] > 0)
    )
    
    defrauded = features[defrauded_mask].copy()
    non_defrauded = features[~defrauded_mask].copy()
    
    print(f"Defrauded students (received fraud contact): {len(defrauded):,}")
    print(f"Non-defrauded students: {len(non_defrauded):,}")
    print(f"Fraud rate: {len(defrauded)/len(df)*100:.2f}%")
    
    # Add engagement level
    defrauded['engagement_level'] = 'RECEIVED_ONLY'
    defrauded.loc[defrauded['fraud_voice_call'] > 0, 'engagement_level'] = 'CALLED_BACK'
    defrauded.loc[defrauded['fraud_msg_call'] > 0, 'engagement_level'] = 'REPLIED'
    defrauded.loc[
        (defrauded['fraud_voice_call'] > 0) & (defrauded['fraud_msg_call'] > 0), 
        'engagement_level'
    ] = 'CALLED_AND_REPLIED'
    
    return df, features, defrauded, non_defrauded


# =============================================================================
# STATISTICAL PORTRAIT
# =============================================================================

def build_statistical_portrait(defrauded: pd.DataFrame, non_defrauded: pd.DataFrame) -> Dict:
    """Build aggregate statistical portrait of defrauded students."""
    
    print("\n" + "=" * 70)
    print("STATISTICAL PORTRAIT OF DEFRAUDED STUDENTS")
    print("=" * 70)
    
    portrait = {}
    
    # === DEMOGRAPHICS ===
    print("\n=== DEMOGRAPHICS ===")
    
    # Age
    avg_age_fraud = defrauded['age'].mean()
    avg_age_non = non_defrauded['age'].mean()
    portrait['age'] = {
        'defrauded_mean': round(avg_age_fraud, 1),
        'non_defrauded_mean': round(avg_age_non, 1),
        'difference': round(avg_age_fraud - avg_age_non, 1)
    }
    print(f"Average Age: Defrauded={avg_age_fraud:.1f}, Non-defrauded={avg_age_non:.1f}")
    
    # Mainland student ratio
    mainland_fraud = defrauded['is_mainland_student'].mean() * 100
    mainland_non = non_defrauded['is_mainland_student'].mean() * 100
    portrait['mainland_student'] = {
        'defrauded_pct': round(mainland_fraud, 1),
        'non_defrauded_pct': round(mainland_non, 1),
        'risk_multiplier': round(mainland_fraud / max(mainland_non, 0.1), 2)
    }
    print(f"Mainland Student: Defrauded={mainland_fraud:.1f}%, Non-defrauded={mainland_non:.1f}%")
    print(f"  -> Mainland students are {mainland_fraud/max(mainland_non, 0.1):.1f}x more likely to be targeted")
    
    # New to HK
    new_hk_fraud = defrauded['is_new_to_hk'].mean() * 100
    new_hk_non = non_defrauded['is_new_to_hk'].mean() * 100
    portrait['new_to_hk'] = {
        'defrauded_pct': round(new_hk_fraud, 1),
        'non_defrauded_pct': round(new_hk_non, 1)
    }
    print(f"New to HK: Defrauded={new_hk_fraud:.1f}%, Non-defrauded={new_hk_non:.1f}%")
    
    # === COMMUNICATION PATTERNS ===
    print("\n=== COMMUNICATION PATTERNS ===")
    
    # Foreign exposure
    foreign_fraud = defrauded['foreign_exposure_total'].mean()
    foreign_non = non_defrauded['foreign_exposure_total'].mean()
    portrait['foreign_exposure'] = {
        'defrauded_mean': round(foreign_fraud, 1),
        'non_defrauded_mean': round(foreign_non, 1),
        'risk_multiplier': round(foreign_fraud / max(foreign_non, 0.1), 2)
    }
    print(f"Foreign Call/Msg Exposure: Defrauded={foreign_fraud:.1f}, Non-defrauded={foreign_non:.1f}")
    
    # Foreign dominance
    dom_fraud = defrauded['foreign_dominance'].mean() * 100
    dom_non = non_defrauded['foreign_dominance'].mean() * 100
    portrait['foreign_dominance'] = {
        'defrauded_pct': round(dom_fraud, 1),
        'non_defrauded_pct': round(dom_non, 1)
    }
    print(f"Foreign > Local Calls: Defrauded={dom_fraud:.1f}%, Non-defrauded={dom_non:.1f}%")
    
    # Mainland carrier calls
    mc_fraud = defrauded['mainland_carrier_calls'].mean()
    mc_non = non_defrauded['mainland_carrier_calls'].mean()
    portrait['mainland_carrier_calls'] = {
        'defrauded_mean': round(mc_fraud, 1),
        'non_defrauded_mean': round(mc_non, 1)
    }
    print(f"Mainland Carrier Calls: Defrauded={mc_fraud:.1f}, Non-defrauded={mc_non:.1f}")
    
    # Repeat unknown caller
    repeat_fraud = defrauded['has_repeat_unknown_caller'].mean() * 100
    repeat_non = non_defrauded['has_repeat_unknown_caller'].mean() * 100
    portrait['repeat_unknown_caller'] = {
        'defrauded_pct': round(repeat_fraud, 1),
        'non_defrauded_pct': round(repeat_non, 1)
    }
    print(f"Has Repeat Unknown Caller: Defrauded={repeat_fraud:.1f}%, Non-defrauded={repeat_non:.1f}%")
    
    # === APP BEHAVIOR ===
    print("\n=== APP BEHAVIOR ===")
    
    app_fraud = defrauded['heavy_mainland_app_user'].mean() * 100
    app_non = non_defrauded['heavy_mainland_app_user'].mean() * 100
    portrait['heavy_app_user'] = {
        'defrauded_pct': round(app_fraud, 1),
        'non_defrauded_pct': round(app_non, 1)
    }
    print(f"Heavy Mainland App User: Defrauded={app_fraud:.1f}%, Non-defrauded={app_non:.1f}%")
    
    # === DEVICE & PLAN ===
    print("\n=== DEVICE & PLAN ===")
    
    esim_fraud = defrauded['uses_esim'].mean() * 100
    esim_non = non_defrauded['uses_esim'].mean() * 100
    portrait['uses_esim'] = {
        'defrauded_pct': round(esim_fraud, 1),
        'non_defrauded_pct': round(esim_non, 1)
    }
    print(f"Uses eSIM: Defrauded={esim_fraud:.1f}%, Non-defrauded={esim_non:.1f}%")
    
    fiveg_fraud = defrauded['uses_5g'].mean() * 100
    fiveg_non = non_defrauded['uses_5g'].mean() * 100
    portrait['uses_5g'] = {
        'defrauded_pct': round(fiveg_fraud, 1),
        'non_defrauded_pct': round(fiveg_non, 1)
    }
    print(f"Uses 5G: Defrauded={fiveg_fraud:.1f}%, Non-defrauded={fiveg_non:.1f}%")
    
    # === ENGAGEMENT BREAKDOWN ===
    print("\n=== ENGAGEMENT BREAKDOWN ===")
    eng_counts = defrauded['engagement_level'].value_counts()
    portrait['engagement_distribution'] = eng_counts.to_dict()
    for level, count in eng_counts.items():
        pct = count / len(defrauded) * 100
        print(f"  {level}: {count} ({pct:.1f}%)")
    
    return portrait


# =============================================================================
# STUDENT ARCHETYPES
# =============================================================================

def build_archetypes(defrauded: pd.DataFrame) -> List[Dict]:
    """Create student archetypes based on clustering characteristics."""
    
    print("\n" + "=" * 70)
    print("HIGH-RISK STUDENT ARCHETYPES")
    print("=" * 70)
    
    archetypes = []
    
    # === ARCHETYPE 1: Mainland Freshman ===
    archetype1_mask = (
        (defrauded['is_mainland_student'] == 1) &
        (defrauded['age'] <= 22) &
        (defrauded['is_new_to_hk'] == 1)
    )
    archetype1 = defrauded[archetype1_mask]
    
    if len(archetype1) > 0:
        archetypes.append({
            'name': 'Mainland Freshman (内地新生)',
            'description': 'Young mainland students recently arrived in HK, unfamiliar with local systems',
            'count': len(archetype1),
            'pct_of_defrauded': round(len(archetype1) / len(defrauded) * 100, 1),
            'characteristics': {
                'age_range': '18-22',
                'avg_foreign_exposure': round(archetype1['foreign_exposure_total'].mean(), 1),
                'mainland_app_user_pct': round(archetype1['heavy_mainland_app_user'].mean() * 100, 1),
                'engagement_rate': round(archetype1['any_engagement'].mean() * 100, 1)
            },
            'risk_factors': ['New to HK', 'Mainland ID', 'Heavy mainland app usage', 'Language barrier']
        })
        print(f"\n1. MAINLAND FRESHMAN (内地新生)")
        print(f"   Count: {len(archetype1)} ({len(archetype1)/len(defrauded)*100:.1f}% of defrauded)")
        print(f"   Age: 18-22, New to HK, Uses mainland apps")
        print(f"   Engagement rate: {archetype1['any_engagement'].mean()*100:.1f}%")
    
    # === ARCHETYPE 2: Graduate Researcher ===
    archetype2_mask = (
        (defrauded['age'] >= 23) &
        (defrauded['age'] <= 30) &
        (defrauded['foreign_dominance'] == 1)
    )
    archetype2 = defrauded[archetype2_mask]
    
    if len(archetype2) > 0:
        archetypes.append({
            'name': 'Graduate Researcher (研究生)',
            'description': 'Graduate/PhD students with high international communication',
            'count': len(archetype2),
            'pct_of_defrauded': round(len(archetype2) / len(defrauded) * 100, 1),
            'characteristics': {
                'age_range': '23-30',
                'avg_foreign_exposure': round(archetype2['foreign_exposure_total'].mean(), 1),
                'mainland_carrier_calls': round(archetype2['mainland_carrier_calls'].mean(), 1)
            },
            'risk_factors': ['High foreign call volume', 'Academic connections', 'International travel']
        })
        print(f"\n2. GRADUATE RESEARCHER (研究生)")
        print(f"   Count: {len(archetype2)} ({len(archetype2)/len(defrauded)*100:.1f}% of defrauded)")
        print(f"   Age: 23-30, High foreign call dominance")
    
    # === ARCHETYPE 3: Frequent Commuter ===
    archetype3_mask = (
        (defrauded['is_frequent_commuter'] == 1) &
        (defrauded['mainland_days'] > 10)
    )
    archetype3 = defrauded[archetype3_mask]
    
    if len(archetype3) > 0:
        archetypes.append({
            'name': 'Frequent Commuter (常返内地)',
            'description': 'Students who frequently travel between HK and Mainland',
            'count': len(archetype3),
            'pct_of_defrauded': round(len(archetype3) / len(defrauded) * 100, 1),
            'characteristics': {
                'avg_mainland_days': round(archetype3['mainland_days'].mean(), 1),
                'avg_hk_trips': round(archetype3['hk_trips'].mean(), 1)
            },
            'risk_factors': ['Cross-border sim usage', 'Vulnerable during travel', 'Multiple contact points']
        })
        print(f"\n3. FREQUENT COMMUTER (常返内地)")
        print(f"   Count: {len(archetype3)} ({len(archetype3)/len(defrauded)*100:.1f}% of defrauded)")
        print(f"   Frequent cross-border travel")
    
    # === ARCHETYPE 4: Repeat Target ===
    archetype4_mask = (
        (defrauded['has_repeat_unknown_caller'] == 1) &
        (defrauded['max_repeat_caller'] >= 5)
    )
    archetype4 = defrauded[archetype4_mask]
    
    if len(archetype4) > 0:
        archetypes.append({
            'name': 'Repeat Target (多次被骚扰)',
            'description': 'Students receiving repeated calls from unknown numbers',
            'count': len(archetype4),
            'pct_of_defrauded': round(len(archetype4) / len(defrauded) * 100, 1),
            'characteristics': {
                'avg_repeat_calls': round(archetype4['max_repeat_caller'].mean(), 1),
                'engagement_rate': round(archetype4['any_engagement'].mean() * 100, 1)
            },
            'risk_factors': ['Phone number exposed', 'Persistent targeting', 'Higher engagement risk']
        })
        print(f"\n4. REPEAT TARGET (多次被骚扰)")
        print(f"   Count: {len(archetype4)} ({len(archetype4)/len(defrauded)*100:.1f}% of defrauded)")
        print(f"   Received 5+ calls from same unknown number")
    
    # === ARCHETYPE 5: Engaged Victim ===
    archetype5 = defrauded[defrauded['any_engagement'] == 1]
    
    if len(archetype5) > 0:
        archetypes.append({
            'name': 'Engaged Victim (已回应受害者)',
            'description': 'Students who actively responded to fraud attempts',
            'count': len(archetype5),
            'pct_of_defrauded': round(len(archetype5) / len(defrauded) * 100, 1),
            'characteristics': {
                'called_back': int((archetype5['fraud_voice_call'] > 0).sum()),
                'replied_sms': int((archetype5['fraud_msg_call'] > 0).sum()),
                'avg_age': round(archetype5['age'].mean(), 1),
                'mainland_pct': round(archetype5['is_mainland_student'].mean() * 100, 1)
            },
            'risk_factors': ['Already engaged', 'Highest risk', 'May have shared information']
        })
        print(f"\n5. ENGAGED VICTIM (已回应受害者) - CRITICAL")
        print(f"   Count: {len(archetype5)} ({len(archetype5)/len(defrauded)*100:.1f}% of defrauded)")
        print(f"   Called back: {(archetype5['fraud_voice_call']>0).sum()}")
        print(f"   Replied SMS: {(archetype5['fraud_msg_call']>0).sum()}")
    
    return archetypes


# =============================================================================
# ML RISK MODEL
# =============================================================================

def build_risk_model(features: pd.DataFrame) -> Dict:
    """Build ML model for risk identification."""
    
    print("\n" + "=" * 70)
    print("BUILDING ML RISK IDENTIFICATION MODEL")
    print("=" * 70)
    
    # Target: received fraud contact
    y = (
        (features['fraud_voice_receive'] > 0) |
        (features['fraud_msg_receive'] > 0)
    ).astype(int)
    
    # Features for model
    feature_cols = [
        'is_mainland_student', 'is_new_to_hk', 'is_hk_local',
        'age', 'is_female',
        'foreign_exposure_total', 'local_exposure_total', 
        'foreign_to_local_ratio', 'foreign_dominance',
        'mainland_carrier_calls', 'mainland_days', 'hk_trips',
        'max_repeat_caller', 'has_repeat_unknown_caller',
        'mainland_app_days', 'heavy_mainland_app_user',
        'uses_5g', 'uses_esim', 'uses_travel_permit'
    ]
    
    X = features[feature_cols].fillna(0)
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Positive samples (defrauded): {y.sum()}")
    print(f"Negative samples: {len(y) - y.sum()}")
    
    # Train XGBoost
    try:
        from xgboost import XGBClassifier
        from sklearn.model_selection import cross_val_score
        
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=len(y[y==0]) / max(len(y[y==1]), 1),  # Handle imbalance
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        print(f"\nCross-validation AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        # Train final model
        model.fit(X, y)
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Risk Factors:")
        for i, row in importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        model_results = {
            'cv_auc': round(cv_scores.mean(), 3),
            'cv_std': round(cv_scores.std(), 3),
            'top_features': importance.head(10).to_dict('records')
        }
        
        return model, model_results, importance
        
    except ImportError:
        print("XGBoost not available, using feature analysis only")
        return None, None, None


# =============================================================================
# INTERPRETABLE RULES
# =============================================================================

def generate_rules(portrait: Dict, importance: pd.DataFrame = None) -> List[Dict]:
    """Generate interpretable rules based on data analysis."""
    
    print("\n" + "=" * 70)
    print("DATA-DRIVEN RISK IDENTIFICATION RULES")
    print("=" * 70)
    
    rules = []
    
    # Rule 1: Mainland Student
    if portrait['mainland_student']['risk_multiplier'] > 2:
        rules.append({
            'id': 'RULE_01',
            'name': 'Mainland Student Flag',
            'condition': 'hk_resident_type == "新来港内地人港漂"',
            'weight': 25,
            'rationale': f"Mainland students are {portrait['mainland_student']['risk_multiplier']}x more likely to be targeted"
        })
        print(f"\nRULE 1: Mainland Student (+25 points)")
        print(f"  Rationale: {portrait['mainland_student']['risk_multiplier']}x higher targeting rate")
    
    # Rule 2: High Foreign Exposure
    if portrait['foreign_exposure']['risk_multiplier'] > 1.5:
        rules.append({
            'id': 'RULE_02',
            'name': 'High Foreign Exposure',
            'condition': 'foreign_exposure_total > 50',
            'weight': 20,
            'rationale': f"High foreign call exposure indicates cross-border connections"
        })
        print(f"\nRULE 2: High Foreign Exposure (+20 points)")
        print(f"  Condition: foreign_exposure > 50 calls/messages")
    
    # Rule 3: Foreign Dominance
    if portrait['foreign_dominance']['defrauded_pct'] > portrait['foreign_dominance']['non_defrauded_pct'] * 1.5:
        rules.append({
            'id': 'RULE_03',
            'name': 'Foreign Call Dominance',
            'condition': 'foreign_exposure > local_exposure',
            'weight': 15,
            'rationale': 'More foreign than local calls suggests mainland-focused communication'
        })
        print(f"\nRULE 3: Foreign Call Dominance (+15 points)")
    
    # Rule 4: New to HK
    rules.append({
        'id': 'RULE_04',
        'name': 'New to Hong Kong',
        'condition': 'is_new_to_hk == 1',
        'weight': 15,
        'rationale': 'New arrivals less familiar with HK systems and scam tactics'
    })
    print(f"\nRULE 4: New to Hong Kong (+15 points)")
    
    # Rule 5: Heavy Mainland App User
    if portrait['heavy_app_user']['defrauded_pct'] > 0:
        rules.append({
            'id': 'RULE_05',
            'name': 'Heavy Mainland App User',
            'condition': 'mainland_app_days >= 14',
            'weight': 10,
            'rationale': 'Heavy WeChat/Alipay usage indicates mainland-centric lifestyle'
        })
        print(f"\nRULE 5: Heavy Mainland App User (+10 points)")
    
    # Rule 6: Repeat Unknown Caller
    rules.append({
        'id': 'RULE_06',
        'name': 'Repeat Unknown Caller',
        'condition': 'max_repeat_caller >= 3',
        'weight': 15,
        'rationale': 'Multiple calls from same unknown number indicates targeting'
    })
    print(f"\nRULE 6: Repeat Unknown Caller (+15 points)")
    
    # Rule 7: Age-based vulnerability
    rules.append({
        'id': 'RULE_07',
        'name': 'Young Student Age',
        'condition': 'age <= 22',
        'weight': 10,
        'rationale': 'Younger students (undergrad) show higher vulnerability'
    })
    print(f"\nRULE 7: Young Student Age (+10 points)")
    
    # Rule 8: Travel Permit User
    rules.append({
        'id': 'RULE_08',
        'name': 'Uses Travel Permit',
        'condition': 'iden_type contains "通行证"',
        'weight': 5,
        'rationale': 'Non-permanent resident using travel document'
    })
    print(f"\nRULE 8: Uses Travel Permit (+5 points)")
    
    print(f"\n=== TOTAL POSSIBLE SCORE: {sum(r['weight'] for r in rules)} ===")
    
    return rules


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(portrait, archetypes, rules, model_results):
    """Save all results to JSON files."""
    
    os.makedirs('Datasets/Student/Results', exist_ok=True)
    
    # Save portrait
    with open('Datasets/Student/Results/high_risk_portrait.json', 'w', encoding='utf-8') as f:
        json.dump(portrait, f, indent=2, ensure_ascii=False)
    
    # Save archetypes
    with open('Datasets/Student/Results/student_archetypes.json', 'w', encoding='utf-8') as f:
        json.dump(archetypes, f, indent=2, ensure_ascii=False)
    
    # Save rules
    with open('Datasets/Student/Results/risk_identification_rules.json', 'w', encoding='utf-8') as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)
    
    # Save model results if available
    if model_results:
        with open('Datasets/Student/Results/ml_model_results.json', 'w', encoding='utf-8') as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)
    print("- Datasets/Student/Results/high_risk_portrait.json")
    print("- Datasets/Student/Results/student_archetypes.json")
    print("- Datasets/Student/Results/risk_identification_rules.json")
    if model_results:
        print("- Datasets/Student/Results/ml_model_results.json")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the complete high-risk student portrait analysis."""
    
    # Step 1: Load and analyze
    df, features, defrauded, non_defrauded = load_and_analyze_students()
    
    # Step 2: Build statistical portrait
    portrait = build_statistical_portrait(defrauded, non_defrauded)
    
    # Step 3: Create archetypes
    archetypes = build_archetypes(defrauded)
    
    # Step 4: Build ML model
    model, model_results, importance = build_risk_model(features)
    
    # Step 5: Generate interpretable rules
    rules = generate_rules(portrait, importance)
    
    # Step 6: Save results
    save_results(portrait, archetypes, rules, model_results)
    
    print("\n" + "=" * 70)
    print("TASK 1 COMPLETE: HIGH-RISK STUDENT PORTRAIT MODEL")
    print("=" * 70)
    
    return portrait, archetypes, rules, model_results


if __name__ == "__main__":
    main()
