"""
Fraud Detection Model Module for Campus Anti-Fraud Detection
Wutong Cup AI+Security Competition

This module implements:
- XGBoost classifier for fraud detection
- Isolation Forest for anomaly detection
- Ensemble combining both + rules
- SHAP-based model interpretability
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    recall_score, precision_score, f1_score, roc_auc_score
)
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: xgboost not installed. Using sklearn GradientBoosting instead.")
    from sklearn.ensemble import GradientBoostingClassifier

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: shap not installed. SHAP explanations will not be available.")


class FraudDetectionModel:
    """
    XGBoost-based fraud detection model with optional SHAP explanations.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        scale_pos_weight: float = None,
        random_state: int = 42
    ):
        """
        Initialize the fraud detection model.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            scale_pos_weight: Weight for positive class (for imbalanced data)
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        
        self.model = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> 'FraudDetectionModel':
        """
        Train the fraud detection model.
        
        Args:
            X: Feature DataFrame
            y: Binary labels (1=fraud, 0=clean)
            feature_names: Optional list of feature names
        """
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = list(X.columns)
        
        # Convert to numeric and handle missing values
        X_numeric = X[self.feature_names].apply(pd.to_numeric, errors='coerce')
        X_numeric = X_numeric.fillna(0)
        
        # Calculate scale_pos_weight if not provided
        if self.scale_pos_weight is None:
            neg_count = (y == 0).sum()
            pos_count = (y == 1).sum()
            self.scale_pos_weight = neg_count / max(pos_count, 1)
        
        # Create and train model
        if HAS_XGBOOST:
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                scale_pos_weight=self.scale_pos_weight,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state
            )
        
        self.model.fit(X_numeric, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud labels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_numeric = X[self.feature_names].apply(pd.to_numeric, errors='coerce')
        X_numeric = X_numeric.fillna(0)
        
        return self.model.predict(X_numeric)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_numeric = X[self.feature_names].apply(pd.to_numeric, errors='coerce')
        X_numeric = X_numeric.fillna(0)
        
        return self.model.predict_proba(X_numeric)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if HAS_XGBOOST:
            importance = self.model.feature_importances_
        else:
            importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def get_shap_explanations(
        self,
        X: pd.DataFrame,
        sample_size: int = 100
    ) -> Optional[Dict[str, Any]]:
        """
        Get SHAP explanations for model predictions.
        
        Args:
            X: Feature DataFrame
            sample_size: Number of samples to explain
            
        Returns:
            Dictionary with SHAP values and feature importance
        """
        if not HAS_SHAP:
            print("SHAP not available. Install with: pip install shap")
            return None
        
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_numeric = X[self.feature_names].apply(pd.to_numeric, errors='coerce')
        X_numeric = X_numeric.fillna(0)
        
        # Sample if too large
        if len(X_numeric) > sample_size:
            X_sample = X_numeric.sample(sample_size, random_state=42)
        else:
            X_sample = X_numeric
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        # Get mean absolute SHAP values for feature importance
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_importance = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': mean_abs_shap
        }).sort_values('shap_importance', ascending=False)
        
        return {
            'shap_values': shap_values,
            'feature_importance': shap_importance,
            'expected_value': explainer.expected_value
        }
    
    def save(self, path: str):
        """Save model to disk."""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'params': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'scale_pos_weight': self.scale_pos_weight
            }
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'FraudDetectionModel':
        """Load model from disk."""
        model_data = joblib.load(path)
        
        instance = cls(
            n_estimators=model_data['params']['n_estimators'],
            max_depth=model_data['params']['max_depth'],
            learning_rate=model_data['params']['learning_rate'],
            scale_pos_weight=model_data['params']['scale_pos_weight']
        )
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = True
        
        return instance


class AnomalyDetector:
    """
    Isolation Forest-based anomaly detector for fraud detection.
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees
            random_state: Random seed
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.feature_names = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(
        self,
        X: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> 'AnomalyDetector':
        """Train the anomaly detector."""
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = list(X.columns)
        
        X_numeric = X[self.feature_names].apply(pd.to_numeric, errors='coerce')
        X_numeric = X_numeric.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_numeric)
        
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomalies (1=normal, -1=anomaly)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_numeric = X[self.feature_names].apply(pd.to_numeric, errors='coerce')
        X_numeric = X_numeric.fillna(0)
        X_scaled = self.scaler.transform(X_numeric)
        
        return self.model.predict(X_scaled)
    
    def predict_fraud(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud labels (1=fraud, 0=clean)."""
        predictions = self.predict(X)
        return (predictions == -1).astype(int)
    
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Get anomaly scores (lower = more anomalous)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_numeric = X[self.feature_names].apply(pd.to_numeric, errors='coerce')
        X_numeric = X_numeric.fillna(0)
        X_scaled = self.scaler.transform(X_numeric)
        
        return self.model.decision_function(X_scaled)


class FraudDetectionEnsemble:
    """
    Ensemble combining XGBoost, Isolation Forest, and Rules for fraud detection.
    """
    
    def __init__(
        self,
        xgb_weight: float = 0.5,
        iforest_weight: float = 0.2,
        rules_weight: float = 0.3,
        threshold: float = 0.5
    ):
        """
        Initialize ensemble.
        
        Args:
            xgb_weight: Weight for XGBoost predictions
            iforest_weight: Weight for Isolation Forest
            rules_weight: Weight for rule-based predictions
            threshold: Decision threshold
        """
        self.xgb_weight = xgb_weight
        self.iforest_weight = iforest_weight
        self.rules_weight = rules_weight
        self.threshold = threshold
        
        self.xgb_model = FraudDetectionModel()
        self.iforest_model = AnomalyDetector()
        self.feature_names = None
        self.is_fitted = False
        
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> 'FraudDetectionEnsemble':
        """Train all models in the ensemble."""
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            # Get numeric columns only
            self.feature_names = [
                col for col in X.columns 
                if X[col].dtype in ['int64', 'float64', 'int32', 'float32']
            ]
        
        print("Training XGBoost model...")
        self.xgb_model.fit(X, y, self.feature_names)
        
        print("Training Isolation Forest...")
        # Train on fraud samples to learn fraud patterns
        X_fraud = X[y == 1]
        contamination = min(0.3, 1 - len(X_fraud) / len(X))
        self.iforest_model.contamination = max(contamination, 0.05)
        self.iforest_model.fit(X, self.feature_names)
        
        self.is_fitted = True
        print("Ensemble training complete!")
        
        return self
    
    def predict_proba_ensemble(
        self,
        X: pd.DataFrame,
        rule_flags: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Get ensemble fraud probability scores.
        
        Args:
            X: Feature DataFrame
            rule_flags: Optional binary series from rule evaluation
            
        Returns:
            Array of fraud probabilities
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        # XGBoost probability
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        
        # Isolation Forest score (normalized to 0-1)
        iforest_scores = -self.iforest_model.decision_function(X)
        iforest_proba = (iforest_scores - iforest_scores.min()) / (
            iforest_scores.max() - iforest_scores.min() + 1e-10
        )
        
        # Rule flags (default to 0 if not provided)
        if rule_flags is not None:
            rules_proba = rule_flags.values.astype(float)
        else:
            rules_proba = np.zeros(len(X))
        
        # Weighted ensemble
        ensemble_proba = (
            self.xgb_weight * xgb_proba +
            self.iforest_weight * iforest_proba +
            self.rules_weight * rules_proba
        )
        
        return ensemble_proba
    
    def predict(
        self,
        X: pd.DataFrame,
        rule_flags: Optional[pd.Series] = None
    ) -> np.ndarray:
        """Predict fraud labels."""
        proba = self.predict_proba_ensemble(X, rule_flags)
        return (proba >= self.threshold).astype(int)
    
    def save(self, model_dir: str):
        """Save all models in the ensemble."""
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        self.xgb_model.save(str(model_path / 'xgb_model.pkl'))
        joblib.dump(self.iforest_model, str(model_path / 'iforest_model.pkl'))
        
        # Save ensemble config
        config = {
            'xgb_weight': self.xgb_weight,
            'iforest_weight': self.iforest_weight,
            'rules_weight': self.rules_weight,
            'threshold': self.threshold,
            'feature_names': self.feature_names
        }
        joblib.dump(config, str(model_path / 'ensemble_config.pkl'))
        print(f"Ensemble saved to {model_dir}")


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Evaluate model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        'recall': recall_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }
    
    if y_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    metrics['true_positives'] = cm[1, 1]
    metrics['false_positives'] = cm[0, 1]
    metrics['false_negatives'] = cm[1, 0]
    metrics['true_negatives'] = cm[0, 0]
    
    return metrics


if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'src')
    from feature_engineering import engineer_fraud_features, create_fraud_labels, get_feature_columns
    # Black sample engine removed - features audited for leakage
    
    print("Loading fraud data...")
    df = pd.read_csv('Datasets/Fraud/Training and Testing Data/fraud_model_2.csv')
    print(f"Records: {len(df)}")
    
    print("\nCreating labels...")
    df_labeled = create_fraud_labels(df)
    
    print("\nEngineering features...")
    features = engineer_fraud_features(df_labeled)
    
    # Get feature columns for model
    feature_cols = get_feature_columns(features)
    print(f"Feature columns: {len(feature_cols)}")
    
    # Create labels
    y = df_labeled['is_confirmed_fraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} (Fraud: {y_train.sum()})")
    print(f"Test set: {len(X_test)} (Fraud: {y_test.sum()})")
    
    # Train ensemble
    print("\n=== TRAINING ENSEMBLE ===")
    ensemble = FraudDetectionEnsemble()
    ensemble.fit(X_train, y_train, feature_cols)
    
    # Pure ML prediction (no rules)
    print("\nPredicting...")
    rule_flags = pd.Series([False] * len(X_test), index=X_test.index)
    y_pred = ensemble.predict(X_test, rule_flags)
    y_proba = ensemble.predict_proba_ensemble(X_test, rule_flags)
    
    # Evaluate
    print("\n=== EVALUATION RESULTS ===")
    metrics = evaluate_model(y_test, y_pred, y_proba)
    print(f"Recall: {metrics['recall']*100:.1f}%")
    print(f"Precision: {metrics['precision']*100:.1f}%")
    print(f"F1 Score: {metrics['f1']*100:.1f}%")
    print(f"AUC-ROC: {metrics['auc_roc']*100:.1f}%")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}")
    print(f"  FN: {metrics['false_negatives']}, TN: {metrics['true_negatives']}")
    
    # Feature importance
    print("\n=== TOP 10 FEATURES ===")
    importance = ensemble.xgb_model.get_feature_importance(10)
    print(importance.to_string(index=False))
    
    # Save model
    print("\nSaving model...")
    ensemble.save('models')
    
    print("\n✓ Fraud detection model training complete!")
