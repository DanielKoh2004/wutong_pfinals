"""
Student Risk Model Module for Campus Anti-Fraud Detection
Wutong Cup AI+Security Competition

This module implements:
- Student risk tier classification
- SHAP-based interpretable risk factors
- Vulnerability scoring
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Optional, List, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: shap not installed. SHAP explanations will not be available.")


class StudentRiskModel:
    """
    Student risk assessment model with interpretable outputs.
    """
    
    RISK_TIERS = ['LOW', 'MODERATE', 'AT_RISK', 'ELEVATED', 'HIGH', 'CRITICAL']
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 8,
        random_state: int = 42
    ):
        """
        Initialize the student risk model.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.model = None
        self.feature_names = None
        self.tier_encoder = LabelEncoder()
        self.tier_encoder.fit(self.RISK_TIERS)
        self.is_fitted = False
    
    def save(self, filepath: str):
        """Save trained model to file."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit_from_features() first.")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'random_state': self.random_state
        }
        joblib.dump(model_data, filepath)
        print(f"Student risk model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load trained model from file."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        # Handle both old and new save formats
        self.is_fitted = model_data.get('is_fitted', True)  # Default to True for old format
        self.n_estimators = model_data.get('n_estimators', 100)
        self.max_depth = model_data.get('max_depth', 8)
        self.random_state = model_data.get('random_state', 42)
        print(f"Student risk model loaded from {filepath}")
        return self
        
    def fit_from_features(
        self,
        features: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> 'StudentRiskModel':
        """
        Train the risk model using engineered features.
        
        TARGET: Binary classification - Contacted vs Not Contacted
        - 1 (Contacted): Students who received fraud calls/msgs
        - 0 (Not Contacted): Students with no fraud contact
        
        Args:
            features: Engineered student features DataFrame
            feature_names: Optional list of feature names to use
        """
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            # Exclude label-related columns from features
            exclude = [
                'has_fraud_contact', 'fraud_voice_receive', 'fraud_voice_call',
                'fraud_msg_receive', 'fraud_msg_call', 'fraud_total_contact',
                'engaged_voice', 'engaged_sms', 'any_engagement'
            ]
            self.feature_names = [
                col for col in features.columns
                if col not in exclude and 
                features[col].dtype in ['int64', 'float64', 'int32', 'float32']
            ]
        
        # === TARGET 1: CONTACTED (Primary Training Target) ===
        # Students who received calls/msgs from fraud numbers
        contacted = (
            (features.get('fraud_voice_receive', 0) > 0) |
            (features.get('fraud_msg_receive', 0) > 0)
        ).astype(int)
        
        print(f"  Training Target: Contacted")
        print(f"    Contacted (1): {contacted.sum()}")
        print(f"    Not Contacted (0): {(contacted == 0).sum()}")
        
        # Prepare features
        X = features[self.feature_names].apply(pd.to_numeric, errors='coerce')
        X = X.fillna(0)
        
        # Train binary classifier with class balancing for imbalanced data
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            class_weight='balanced'  # Handle imbalanced data
        )
        self.model.fit(X, contacted)
        self.is_fitted = True
        
        print(f"  Model trained: RandomForest with class_weight='balanced'")
        
        return self
    
    def _create_risk_labels(self, features: pd.DataFrame) -> pd.Series:
        """Create risk tier labels based on fraud interaction features."""
        labels = pd.Series(index=features.index, dtype='object')
        
        # Default tier
        labels[:] = 'LOW'
        
        # MODERATE: New to HK
        if 'is_new_to_hk' in features.columns:
            labels[features['is_new_to_hk'] == 1] = 'MODERATE'
        
        # AT_RISK: Mainland student with foreign dominance
        if 'is_mainland_student' in features.columns and 'foreign_dominance' in features.columns:
            labels[
                (features['is_mainland_student'] == 1) & 
                (features['foreign_dominance'] == 1)
            ] = 'AT_RISK'
        
        # ELEVATED: Has fraud contact with high vulnerability
        if 'has_fraud_contact' in features.columns and 'vulnerability_score' in features.columns:
            labels[
                (features['has_fraud_contact'] == 1) & 
                (features['vulnerability_score'] > 50)
            ] = 'ELEVATED'
        
        # HIGH: Received fraud calls/messages
        if 'fraud_voice_receive' in features.columns:
            labels[features['fraud_voice_receive'] > 0] = 'HIGH'
        if 'fraud_msg_receive' in features.columns:
            labels[features['fraud_msg_receive'] > 0] = 'HIGH'
        
        # CRITICAL: Engaged with fraudster
        if 'any_engagement' in features.columns:
            labels[features['any_engagement'] == 1] = 'CRITICAL'
        
        return labels
    
    def predict_risk_score(self, features: pd.DataFrame) -> pd.Series:
        """
        Predict risk probability score (0.0 to 1.0).
        
        Output: P(contacted) - probability of being targeted by fraudsters.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit_from_features() first.")
        
        X = features[self.feature_names].apply(pd.to_numeric, errors='coerce')
        X = X.fillna(0)
        
        # Get P(contacted) - probability of class 1
        proba = self.model.predict_proba(X)
        if proba.shape[1] == 2:
            return pd.Series(proba[:, 1], index=features.index)
        else:
            return pd.Series(proba[:, 0], index=features.index)
    
    def predict_tier(self, features: pd.DataFrame) -> pd.Series:
        """
        Predict risk tier for students.
        
        Flow:
        1. Get probability score from model (0.0 to 1.0)
        2. Convert probability to tier (Low, Moderate, High, etc.)
        3. CRITICAL OVERRIDE: If engaged (voice_call > 0) → force CRITICAL
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit_from_features() first.")
        
        # Step 1: Get probability scores
        risk_scores = self.predict_risk_score(features)
        
        # Step 2: Convert probability to tiers
        tiers = pd.Series(index=features.index, dtype='object')
        tiers[:] = 'LOW'
        tiers[risk_scores > 0.2] = 'MODERATE'
        tiers[risk_scores > 0.4] = 'AT_RISK'
        tiers[risk_scores > 0.6] = 'ELEVATED'
        tiers[risk_scores > 0.8] = 'HIGH'
        
        # Step 3: CRITICAL OVERRIDE - Engaged students
        # If student already called back or replied, force CRITICAL regardless of score
        if 'fraud_voice_call' in features.columns:
            engaged_voice = features['fraud_voice_call'] > 0
            tiers[engaged_voice] = 'CRITICAL'
        if 'engaged_voice' in features.columns:
            tiers[features['engaged_voice'] == 1] = 'CRITICAL'
        if 'any_engagement' in features.columns:
            tiers[features['any_engagement'] == 1] = 'CRITICAL'
        
        return tiers
    
    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict probability for each risk tier."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit_from_features() first.")
        
        X = features[self.feature_names].apply(pd.to_numeric, errors='coerce')
        X = X.fillna(0)
        
        proba = self.model.predict_proba(X)
        # Use only the classes present in the trained model
        class_names = self.tier_encoder.inverse_transform(self.model.classes_)
        return pd.DataFrame(
            proba,
            columns=class_names,
            index=features.index
        )
    
    def get_risk_factors(
        self,
        student_features: pd.Series,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get top risk factors for a single student.
        Uses feature values to identify key risk drivers.
        
        Args:
            student_features: Single row of student features
            top_n: Number of top factors to return
            
        Returns:
            List of risk factor dictionaries
        """
        risk_factors = []
        
        # Define interpretable risk explanations
        factor_explanations = {
            'is_mainland_student': '作为新来港内地学生，更容易成为电诈目标',
            'is_new_to_hk': '新来港人士，对当地诈骗手法不熟悉',
            'foreign_dominance': '境外通讯比本地通讯多，面临更多境外诈骗风险',
            'has_repeat_unknown_caller': '频繁接收相同陌生号码来电',
            'heavy_mainland_app_user': '大量使用大陆App，可能暴露个人信息',
            'uses_travel_permit': '使用通行证登记，可能被针对性诈骗',
            'foreign_voice_cnt': '接听多个陌生境外号码',
            'mainland_carrier_calls': '接到多个大陆运营商来电',
            'has_fraud_contact': '已与已知诈骗号码有接触',
            'engaged_voice': '已回拨过诈骗号码',
            'engaged_sms': '已回复过诈骗短信',
            'vulnerability_score': '综合风险评分较高',
            'mobility_score': '频繁跨境通勤',
            'is_frequent_commuter': '经常往返大陆与香港',
        }
        
        for feature_name in self.feature_names:
            value = student_features.get(feature_name, 0)
            if pd.isna(value):
                value = 0
                
            # Check if this is a risk indicator
            if feature_name in factor_explanations and value > 0:
                risk_factors.append({
                    'factor': feature_name,
                    'value': value,
                    'explanation': factor_explanations[feature_name]
                })
        
        # Sort by value (assuming higher = more risk)
        risk_factors.sort(key=lambda x: x['value'], reverse=True)
        
        return risk_factors[:top_n]
    
    def get_shap_explanation(
        self,
        features: pd.DataFrame,
        sample_size: int = 100
    ) -> Optional[Dict[str, Any]]:
        """Get SHAP-based feature importance."""
        if not HAS_SHAP:
            print("SHAP not available. Install with: pip install shap")
            return None
        
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit_from_features() first.")
        
        X = features[self.feature_names].apply(pd.to_numeric, errors='coerce')
        X = X.fillna(0)
        
        if len(X) > sample_size:
            X = X.sample(sample_size, random_state=42)
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        
        # Average SHAP importance across all classes
        if isinstance(shap_values, list):
            avg_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            avg_shap = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': avg_shap
        }).sort_values('shap_importance', ascending=False)
        
        return {
            'feature_importance': importance_df,
            'shap_values': shap_values
        }
    
    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Get feature importance from model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit_from_features() first.")
        
        importance = self.model.feature_importances_
        return pd.DataFrame({
        }).sort_values('importance', ascending=False).head(top_n)


def generate_student_risk_report(
    features: pd.DataFrame,
    model: StudentRiskModel,
    student_index: int
) -> Dict[str, Any]:
    """
    Generate a detailed risk report for a specific student.
    
    Args:
        features: Engineered features DataFrame
        model: Trained StudentRiskModel
        student_index: Index of the student
        
    Returns:
        Dictionary with complete risk report
    """
    student = features.iloc[student_index]
    
    # Get predictions
    tier = model.predict_tier(features.iloc[[student_index]]).iloc[0]
    proba = model.predict_proba(features.iloc[[student_index]])
    
    # Get risk factors
    risk_factors = model.get_risk_factors(student)
    
    return {
        'student_index': student_index,
        'risk_tier': tier,
        'tier_probabilities': proba.iloc[0].to_dict(),
        'top_risk_factors': risk_factors,
        'vulnerability_score': student.get('vulnerability_score', 0),
        'foreign_exposure': student.get('foreign_exposure_total', 0),
        'has_fraud_contact': bool(student.get('has_fraud_contact', 0)),
        'any_engagement': bool(student.get('any_engagement', 0))
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'src')
    from feature_engineering import engineer_student_features
    
    print("Loading student data...")
    df = pd.read_csv('Datasets/Student/Training and Testing Data/student_model.csv')
    print(f"Records: {len(df)}")
    
    print("\nEngineering features...")
    features = engineer_student_features(df)
    print(f"Features: {len(features.columns)}")
    
    print("\nTraining student risk model...")
    model = StudentRiskModel()
    model.fit_from_features(features)
    
    print("\nPredicting risk tiers...")
    tiers = model.predict_tier(features)
    
    print("\n=== RISK TIER DISTRIBUTION ===")
    tier_counts = tiers.value_counts()
    for tier in StudentRiskModel.RISK_TIERS:
        count = tier_counts.get(tier, 0)
        pct = count / len(tiers) * 100
        print(f"{tier:12s}: {count:6d} ({pct:5.1f}%)")
    
    print("\n=== TOP 10 FEATURES ===")
    importance = model.get_feature_importance(10)
    print(importance.to_string(index=False))
    
    # Generate sample report for a high-risk student
    high_risk_indices = features[features['any_engagement'] == 1].index
    if len(high_risk_indices) > 0:
        print("\n=== SAMPLE RISK REPORT (High-Risk Student) ===")
        sample_idx = high_risk_indices[0]
        report = generate_student_risk_report(features, model, sample_idx)
        print(f"Risk Tier: {report['risk_tier']}")
        print(f"Vulnerability Score: {report['vulnerability_score']}")
        print(f"Has Fraud Contact: {report['has_fraud_contact']}")
        print(f"Any Engagement: {report['any_engagement']}")
        if report['top_risk_factors']:
            print("Top Risk Factors:")
            for factor in report['top_risk_factors'][:3]:
                print(f"  - {factor['explanation']}")
    
    # Save model
    print("\nSaving model...")
    model.save('models/student_risk_model.pkl')
    
    print("\n✓ Student risk model training complete!")
