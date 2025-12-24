"""
Campus Anti-Fraud Demo - Gradio Interface
Wutong Cup AI+Security Competition

Interactive demo showcasing:
1. Risk Report Generator - Select student, generate fraud risk report
2. Personalized Warning - Generate anti-fraud education for at-risk students
3. Conversational Query - Ask questions about fraud data in any language
"""

import os
import sys

# Load .env file for API keys
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, 'src')

import gradio as gr
import pandas as pd
from typing import Optional

# Import our modules
from feature_engineering import engineer_student_features
from llm_integration import (
    GroqClient, 
    RiskReportGenerator, 
    PersonalizedWarningGenerator, 
    FraudQueryInterface,
    StudentRiskProfile,
    FraudProfile
)

# =============================================================================
# DATA LOADING (Using pre-trained results from main.py)
# =============================================================================

def load_pretrained_data():
    """Load data trained by main.py from saved results."""
    print("Loading pre-trained data from main.py results...")
    
    # Load student risk results
    with open('Datasets/Student/Results/student_risk_results.json', 'r', encoding='utf-8') as f:
        student_results = json.load(f)
    print(f"  Student results: {student_results['total_students']} students, {student_results['defrauded_count']} defrauded")
    
    # Load student archetypes if available
    try:
        with open('Datasets/Student/Results/student_archetypes.json', 'r', encoding='utf-8') as f:
            archetypes = json.load(f)
        print(f"  Archetypes: {len(archetypes)} student types")
    except FileNotFoundError:
        archetypes = []
    
    # Load high-risk portrait
    try:
        with open('Datasets/Student/Results/high_risk_portrait.json', 'r', encoding='utf-8') as f:
            portrait = json.load(f)
    except FileNotFoundError:
        portrait = {}
    
    # Now load the actual student data for display
    print("Loading student data for display...")
    df = pd.read_csv('Datasets/Student/Training and Testing Data/student_model.csv')
    features = engineer_student_features(df)
    
    # Identify contacted students (TRAINING DATA - will be excluded from demo)
    contacted_mask = (
        (features['fraud_voice_receive'] > 0) |
        (features['fraud_msg_receive'] > 0)
    )
    
    # Identify engaged students (VALIDATION SET)
    engaged_mask = features['any_engagement'] == 1
    
    # Add features to df for display
    df['vulnerability_score'] = features['vulnerability_score']
    df['has_fraud_contact'] = features['has_fraud_contact']
    df['any_engagement'] = features['any_engagement']
    df['fraud_voice_receive'] = features['fraud_voice_receive']
    df['fraud_voice_call'] = features['fraud_voice_call']
    df['fraud_msg_receive'] = features['fraud_msg_receive']
    df['fraud_msg_call'] = features['fraud_msg_call']
    df['is_mainland'] = features['is_mainland_student']
    df['foreign_exposure'] = features['foreign_exposure_total']
    
    # For demo: Get NON-CONTACTED students only (these are the predictions)
    # Contacted students are training data, not shown in demo
    non_contacted_df = df[~contacted_mask].copy()
    
    # Classify risk tier for non-contacted based on model predictions
    # (This will be computed in get_student_choices using the trained model)
    non_contacted_df['risk_tier'] = 'PREDICTED'  # Will be set by model
    
    print(f"  Total students: {len(df)}")
    print(f"  Contacted (training data, excluded): {contacted_mask.sum()}")
    print(f"  Engaged (validation set): {engaged_mask.sum()}")
    print(f"  Non-contacted (for predictions): {len(non_contacted_df)}")
    
    # For demo, we show top predicted risk (sorted later by model score)
    high_risk_df = non_contacted_df.copy()
    
    # Load fraud data context
    fraud_results_path = 'Datasets/Fraud/Results/fraud_detection_results.json'
    try:
        with open(fraud_results_path, 'r', encoding='utf-8') as f:
            fraud_results = json.load(f)
    except FileNotFoundError:
        fraud_results = {}
    
    return df, features, high_risk_df, student_results, portrait, archetypes, fraud_results


# Load data at startup
print("Initializing demo with pre-trained data...")
import json
STUDENT_DF, STUDENT_FEATURES, HIGH_RISK_DF, STUDENT_RESULTS, PORTRAIT, ARCHETYPES, FRAUD_RESULTS = load_pretrained_data()

# Load VALIDATION/PREDICTION fraud data for Risk Report tab (NOT training data)
print("Loading fraud prediction data for Risk Report...")
from feature_engineering import engineer_fraud_features
FRAUD_DF = pd.read_csv('Datasets/Fraud/Training and Testing Data/validate_data.csv')
FRAUD_FEATURES = engineer_fraud_features(FRAUD_DF)
print(f"  Fraud prediction records: {len(FRAUD_DF)}")

# Load fraud detection model for predictions
try:
    from models.fraud_detection_model import FraudDetectionModel
    FRAUD_MODEL = FraudDetectionModel.load('models/xgb_model.pkl')
    print(f"  Fraud detection model loaded successfully")
    FRAUD_MODEL_LOADED = True
except Exception as e:
    print(f"  Warning: Could not load fraud model: {e}")
    FRAUD_MODEL = None
    FRAUD_MODEL_LOADED = False

# Data context for LLM queries (from training results)
DATA_CONTEXT = {
    'total_students': STUDENT_RESULTS.get('total_students', len(STUDENT_DF)),
    'defrauded_count': STUDENT_RESULTS.get('defrauded_count', 0),
    'engaged_count': STUDENT_RESULTS.get('engaged_count', 0),
    'fraud_rate': STUDENT_RESULTS.get('fraud_rate', 0),
    'high_risk_students': len(HIGH_RISK_DF),
    'tier_distribution': STUDENT_RESULTS.get('tier_distribution', {}),
    'risk_factors': STUDENT_RESULTS.get('risk_factors', []),
    'fraud_results': FRAUD_RESULTS
}


# =============================================================================
# STUDENT SELECTOR
# =============================================================================

def get_student_choices():
    """Get list of NON-CONTACTED students with model-predicted risk scores."""
    choices = []
    
    # Load trained model for probability scoring
    try:
        from models.student_risk_model import StudentRiskModel
        model = StudentRiskModel()
        model.load('models/student_risk_model.pkl')
        use_model = True
        print(f"  Loaded student risk model successfully")
    except Exception as e:
        print(f"  Warning: Could not load student model: {e}")
        use_model = False
        model = None
    
    # Limit to top 200 students for performance
    student_subset = HIGH_RISK_DF.head(200)
    
    # BATCH PREDICTION - much faster than row-by-row
    if use_model and model is not None:
        try:
            # Get valid indices
            valid_indices = [idx for idx in student_subset.index if idx in STUDENT_FEATURES.index]
            if valid_indices:
                batch_features = STUDENT_FEATURES.loc[valid_indices]
                scores = model.predict_risk_score(batch_features) * 100
                score_dict = dict(zip(valid_indices, scores))
            else:
                score_dict = {}
        except Exception as e:
            print(f"  Warning: Batch prediction failed: {e}")
            score_dict = {}
    else:
        score_dict = {}
    
    # Build student_scores list
    student_scores = []
    for idx, row in student_subset.iterrows():
        student_type = "Mainland" if row.get('is_mainland', False) else "Local"
        
        # Get model probability from batch prediction
        prob_score = score_dict.get(idx, min(row.get('vulnerability_score', 0) * 10, 100))
        prob_score = float(prob_score)
        
        # Assign risk tier based on probability
        if prob_score >= 80:
            risk = "CRITICAL"
        elif prob_score >= 60:
            risk = "HIGH"
        elif prob_score >= 40:
            risk = "ELEVATED"
        elif prob_score >= 20:
            risk = "AT_RISK"
        else:
            risk = "LOW"
        
        student_scores.append((idx, prob_score, risk, student_type))
    
    # Sort by risk score (highest first)
    student_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 200 highest risk predictions
    for idx, prob_score, risk, student_type in student_scores[:200]:
        label = f"[{risk}] ID:{idx} - {student_type} | PREDICTED | Score:{prob_score:.0f}"
        choices.append((label, idx))
    
    print(f"  Top predicted risk students: {len(choices)} (sorted by model score)")
    return choices


def get_fraud_choices():
    """Get list of fraud phone numbers with model predictions for Risk Report dropdown."""
    from feature_engineering import get_feature_columns
    choices = []
    
    # Get audit status for filtering - only show PREDICTED (pending) records
    # Exclude training data (confirmed fraud/clean) - those are already labeled
    if 'audit_status' in FRAUD_DF.columns:
        pending_mask = FRAUD_DF['audit_status'] == '待稽核'  # Pending audit
        fraud_subset = FRAUD_DF[pending_mask].copy()
    else:
        fraud_subset = FRAUD_DF.copy()
    
    print(f"  Fraud records for prediction: {len(fraud_subset)}")
    
    # Limit for performance and use batch prediction
    fraud_subset = fraud_subset.head(200)
    
    # Get feature columns for prediction
    feature_cols = get_feature_columns(FRAUD_FEATURES)
    
    # BATCH PREDICTION - much faster than row-by-row
    if FRAUD_MODEL_LOADED and FRAUD_MODEL is not None:
        try:
            batch_features = FRAUD_FEATURES.loc[fraud_subset.index, feature_cols]
            proba = FRAUD_MODEL.predict_proba(batch_features)
            # Handle 2D array output - get fraud probability (class 1)
            if hasattr(proba, 'shape') and len(proba.shape) > 1 and proba.shape[1] > 1:
                scores = proba[:, 1] * 100
            else:
                scores = proba[:, 0] * 100 if len(proba.shape) > 1 else proba * 100
        except Exception as e:
            print(f"  Warning: Batch prediction failed: {e}")
            scores = [50.0] * len(fraud_subset)
    else:
        scores = [50.0] * len(fraud_subset)
    
    # Build fraud_scores list
    fraud_scores = []
    for i, (idx, row) in enumerate(fraud_subset.iterrows()):
        prob_score = float(scores[i]) if hasattr(scores, '__iter__') else 50.0
        ntwk_type = str(row.get('ntwk_type', 'Unknown'))
        hit_student = "Student" if str(row.get('hit_student_model', 'N')) == 'Y' else ""
        fraud_scores.append((idx, prob_score, ntwk_type, hit_student))
    
    # Sort by risk score (highest first)
    fraud_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Create choices
    for idx, prob_score, ntwk_type, hit_student in fraud_scores[:200]:
        tier = "CRITICAL" if prob_score >= 80 else "HIGH" if prob_score >= 60 else "ELEVATED" if prob_score >= 40 else "MODERATE"
        label = f"[{tier}] ID:{idx} - {ntwk_type} {hit_student} | Score:{prob_score:.0f}"
        choices.append((label, idx))
    
    return choices


def get_risk_factors(features_row: pd.Series) -> list:
    """Get list of demographic risk factors that contribute to high probability."""
    risk_factors = []
    
    # Define risk factor explanations
    factor_checks = [
        ('is_mainland_student', 'Mainland student (commonly targeted)'),
        ('is_new_to_hk', 'New to Hong Kong (unfamiliar with local scams)'),
        ('foreign_dominance', 'More foreign than local calls (cross-border exposure)'),
        ('has_repeat_unknown_caller', 'Repeat calls from same unknown number'),
        ('heavy_mainland_app_user', 'Heavy mainland app user (data exposure)'),
        ('uses_travel_permit', 'Uses travel permit (cross-border identity)'),
        ('has_fraud_contact', 'Contact with known fraud numbers'),
        ('uses_esim', 'Uses eSIM (potentially disposable)'),
        ('is_frequent_commuter', 'Frequent cross-border commuter'),
    ]
    
    for feature, explanation in factor_checks:
        if features_row.get(feature, 0) > 0:
            risk_factors.append(f"⚠️ {explanation}")
    
    return risk_factors


def get_student_profile(student_idx: int) -> StudentRiskProfile:
    """Get StudentRiskProfile from index."""
    if student_idx is None:
        return None
    
    row = HIGH_RISK_DF.loc[student_idx]
    features_row = STUDENT_FEATURES.loc[student_idx]
    
    # Determine risk tier
    if features_row.get('any_engagement', False):
        tier = "CRITICAL"
    elif features_row.get('has_fraud_contact', False):
        tier = "HIGH"
    elif features_row.get('vulnerability_score', 0) > 50:
        tier = "ELEVATED"
    else:
        tier = "AT_RISK"
    
    # Build risk factors list
    risk_factors = []
    if features_row.get('is_mainland_student', False):
        risk_factors.append('mainland_student')
    if features_row.get('foreign_dominance', False):
        risk_factors.append('high_foreign_exposure')
    if features_row.get('has_repeat_unknown_caller', False):
        risk_factors.append('repeat_unknown_caller')
    if features_row.get('heavy_mainland_app_user', False):
        risk_factors.append('heavy_mainland_app_user')
    if features_row.get('has_fraud_contact', False):
        risk_factors.append('confirmed_fraud_contact')
    
    return StudentRiskProfile(
        risk_tier=tier,
        vulnerability_score=float(features_row.get('vulnerability_score', 0)),
        is_mainland_student=bool(features_row.get('is_mainland_student', False)),
        foreign_exposure=int(features_row.get('foreign_exposure_total', 0)),
        has_fraud_contact=bool(features_row.get('has_fraud_contact', False)),
        fraud_voice_receive=int(features_row.get('fraud_voice_receive', 0)),
        any_engagement=bool(features_row.get('any_engagement', False)),
        top_risk_factors=risk_factors[:5]
    )


def display_student_info(student_idx):
    """Display student information with model probability and risk factors."""
    if student_idx is None:
        return "Please select a student"
    
    try:
        idx = int(student_idx)
        row = HIGH_RISK_DF.loc[idx]
        features_row = STUDENT_FEATURES.loc[idx]
        
        # Get model-based probability score
        try:
            from models.student_risk_model import StudentRiskModel
            model = StudentRiskModel()
            model.load('models/student_risk_model.pkl')
            features_df = STUDENT_FEATURES.loc[[idx]]
            prob_score = model.predict_risk_score(features_df).iloc[0] * 100
        except:
            prob_score = min(features_row.get('vulnerability_score', 0) * 10, 100)
        
        # Determine engagement status
        called_back = features_row.get('fraud_voice_call', 0) > 0
        replied = features_row.get('fraud_msg_call', 0) > 0
        
        if called_back and replied:
            engagement_status = "🚨 CALLED BACK + REPLIED TO FRAUDSTER"
        elif called_back:
            engagement_status = "🚨 CALLED BACK FRAUDSTER"
        elif replied:
            engagement_status = "🚨 REPLIED TO FRAUDSTER"
        else:
            engagement_status = "Received fraud calls only (no response)"
        
        # Get risk factors that caused high probability
        risk_factors = get_risk_factors(features_row)
        risk_factors_str = "\n".join(risk_factors) if risk_factors else "No specific risk factors identified"
        
        info = f"""
### Student Profile (ID: {idx})

| Attribute | Value |
|-----------|-------|
| **Risk Score** | **{prob_score:.0f}/100** (Model Probability) |
| **Age** | {row.get('age', 'N/A')} |
| **Type** | {'Mainland Student' if features_row.get('is_mainland_student', False) else 'Local'} |
| **Foreign Calls** | {features_row.get('foreign_exposure_total', 0):.0f} |
| **Fraud Calls Received** | {features_row.get('fraud_voice_receive', 0):.0f} |
| **Fraud Messages Received** | {features_row.get('fraud_msg_receive', 0):.0f} |
| **Engagement Status** | **{engagement_status}** |

### 🎯 Risk Factors (Why High Probability)
{risk_factors_str}
"""
        return info
    except Exception as e:
        return f"Error loading student: {e}"


def display_fraud_info(fraud_idx):
    """Display fraud case information with model prediction."""
    if fraud_idx is None:
        return "Please select a fraud case"
    
    try:
        from feature_engineering import get_feature_columns
        idx = int(fraud_idx)
        row = FRAUD_DF.loc[idx]
        features_row = FRAUD_FEATURES.loc[idx]
        
        # Get model prediction
        if FRAUD_MODEL_LOADED and FRAUD_MODEL is not None:
            try:
                feature_cols = get_feature_columns(FRAUD_FEATURES)
                features_for_pred = FRAUD_FEATURES.loc[[idx], feature_cols]
                proba = FRAUD_MODEL.predict_proba(features_for_pred)
                # Handle 2D array output - get fraud probability (class 1)
                if hasattr(proba, 'shape') and len(proba.shape) > 1:
                    prob_score = float(proba[0, 1]) * 100 if proba.shape[1] > 1 else float(proba[0, 0]) * 100
                else:
                    prob_score = float(proba[0]) * 100
            except:
                prob_score = 50.0
        else:
            prob_score = 50.0
        
        # Determine tier
        if prob_score >= 80:
            tier = "🚨 CRITICAL"
        elif prob_score >= 60:
            tier = "⚠️ HIGH"
        elif prob_score >= 40:
            tier = "📊 ELEVATED"
        else:
            tier = "📋 MODERATE"
        
        # Get fraud characteristics (convert to Python types)
        ntwk_type = str(row.get('ntwk_type', 'Unknown'))
        hit_student = "Yes" if str(row.get('hit_student_model', 'N')) == 'Y' else "No"
        dispersion = float(row.get('dispersion_rate', 0) or 0)
        total_calls = float(row.get('call_cnt_day', 0) or 0)
        audit_status = str(row.get('audit_status', 'Unknown'))
        
        info = f"""
### Fraud Case Analysis (ID: {idx})

| Attribute | Value |
|-----------|-------|
| **Risk Score** | **{prob_score:.0f}/100** (Model Prediction) |
| **Risk Tier** | {tier} |
| **Network Type** | {ntwk_type} |
| **Targets Students** | {hit_student} |
| **Daily Calls** | {total_calls:.0f} |
| **Dispersion Rate** | {dispersion:.4f} |
| **Audit Status** | {audit_status} |
"""
        return info
    except Exception as e:
        return f"Error loading fraud case: {e}"


def generate_fraud_report(fraud_idx, language):
    """Generate fraud risk report using LLM."""
    api_key, error = initialize_llm()
    if error:
        return error
    
    if fraud_idx is None:
        return "Please select a fraud case"
    
    try:
        from feature_engineering import get_feature_columns
        idx = int(fraud_idx)
        row = FRAUD_DF.loc[idx]
        features_row = FRAUD_FEATURES.loc[idx]
        
        # Get fraud details (convert to Python types)
        ntwk_type = str(row.get('ntwk_type', 'Unknown'))
        hit_student = str(row.get('hit_student_model', 'N')) == 'Y'
        dispersion = float(row.get('dispersion_rate', 0) or 0)
        total_calls = float(row.get('call_cnt_day', 0) or 0)
        
        # Get model prediction
        if FRAUD_MODEL_LOADED and FRAUD_MODEL is not None:
            try:
                feature_cols = get_feature_columns(FRAUD_FEATURES)
                features_for_pred = FRAUD_FEATURES.loc[[idx], feature_cols]
                proba = FRAUD_MODEL.predict_proba(features_for_pred)
                # Handle 2D array output - get fraud probability (class 1)
                if hasattr(proba, 'shape') and len(proba.shape) > 1:
                    prob_score = float(proba[0, 1]) * 100 if proba.shape[1] > 1 else float(proba[0, 0]) * 100
                else:
                    prob_score = float(proba[0]) * 100
            except:
                prob_score = 50.0
        else:
            prob_score = 50.0
        
        # Build prompt
        prompt = f"""Analyze this fraud phone number case:
        
- Network Type: {ntwk_type}
- Targets Students: {"Yes" if hit_student else "No"}
- Model Risk Score: {prob_score:.0f}/100
- Daily Call Volume: {total_calls:.0f}
- Dispersion Rate: {dispersion:.4f}

Generate a brief fraud risk analysis report that includes:
1. Risk Assessment (based on the score and characteristics)
2. Likely Fraud Type (based on patterns)
3. Recommended Actions

Language: {language}"""
        
        # Use LLM to generate report
        llm = GroqClient(api_key)
        response = llm.generate(prompt)
        return response
    except Exception as e:
        return f"Error generating report: {e}"


# =============================================================================
# LLM GENERATORS
# =============================================================================

def initialize_llm():
    """Initialize LLM client with API key."""
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        return None, "GROQ_API_KEY not set. Please set the environment variable."
    return api_key, None


def generate_risk_report(student_idx, language):
    """Generate risk report for selected student."""
    api_key, error = initialize_llm()
    if error:
        return error
    
    if student_idx is None:
        return "Please select a student first."
    
    try:
        idx = int(student_idx)
        profile = get_student_profile(idx)
        
        # Create generator
        generator = RiskReportGenerator(api_key)
        
        # Modify prompt for language
        original_prompt = generator.SYSTEM_PROMPT
        if language == "Chinese (中文)":
            generator.SYSTEM_PROMPT = original_prompt.replace(
                "Default to English if language is unclear",
                "Always respond in Chinese (Simplified Chinese preferred)"
            )
        
        # Create a FraudProfile-like object from student data
        fraud_profile = FraudProfile(
            is_flagged=profile.has_fraud_contact or profile.any_engagement,
            pattern_count=len(profile.top_risk_factors),
            patterns_detected=profile.top_risk_factors,
            triggered_rules=[f"risk_factor_{f}" for f in profile.top_risk_factors[:3]],
            call_cnt_day=profile.foreign_exposure,
            is_prepaid=True,  # Default assumption
            is_sim_farm=False,
            hit_student=True
        )
        
        report = generator.generate_report(fraud_profile)
        return report
    except Exception as e:
        return f"Error generating report: {e}"


def generate_warning(student_idx, language):
    """Generate personalized warning for selected student."""
    api_key, error = initialize_llm()
    if error:
        return error
    
    if student_idx is None:
        return "Please select a student first."
    
    try:
        idx = int(student_idx)
        profile = get_student_profile(idx)
        
        # Create generator
        generator = PersonalizedWarningGenerator(api_key)
        
        # Modify prompt for language
        if language == "Chinese (中文)":
            generator.SYSTEM_PROMPT = generator.SYSTEM_PROMPT.replace(
                "Default to English if language is unclear",
                "Always respond in Chinese (Simplified Chinese preferred)"
            )
        
        warning = generator.generate_warning(profile)
        return warning
    except Exception as e:
        return f"Error generating warning: {e}"


def query_fraud_data(user_query):
    """Process conversational query about fraud data."""
    api_key, error = initialize_llm()
    if error:
        return error
    
    if not user_query.strip():
        return "Please enter a question."
    
    try:
        # Create query interface with data context
        query_interface = FraudQueryInterface(api_key, DATA_CONTEXT)
        response = query_interface.query(user_query)
        return response
    except Exception as e:
        return f"Error processing query: {e}"


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

def create_app():
    """Create the Gradio application."""
    
    student_choices = get_student_choices()
    fraud_choices = get_fraud_choices()
    
    with gr.Blocks(
        title="Campus Anti-Fraud Demo",
        theme=gr.themes.Soft()
    ) as app:
        
        gr.Markdown("""
# 🛡️ Campus Anti-Fraud Detection System
## Wutong Cup AI+Security Competition Demo

This demo showcases LLM-powered fraud detection and student protection features.
        """)
        
        with gr.Tabs():
            # ===== TAB 1: Fraud Risk Report =====
            with gr.TabItem("📊 Fraud Risk Report"):
                gr.Markdown("### Analyze fraud phone numbers with model predictions (excluding training data)")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        fraud_select_1 = gr.Dropdown(
                            choices=[(label, str(idx)) for label, idx in fraud_choices],
                            label="Select Fraud Phone Number",
                            info="Choose a pending case for fraud risk analysis"
                        )
                        language_1 = gr.Dropdown(
                            choices=["English", "Chinese (中文)"],
                            value="English",
                            label="Output Language"
                        )
                        fraud_info_1 = gr.Markdown("Select a fraud case to see details")
                        generate_btn_1 = gr.Button("Generate Fraud Report", variant="primary")
                    
                    with gr.Column(scale=2):
                        report_output = gr.Textbox(
                            label="Fraud Risk Report",
                            lines=15,
                            placeholder="Fraud analysis report will appear here..."
                        )
                
                fraud_select_1.change(
                    fn=display_fraud_info,
                    inputs=[fraud_select_1],
                    outputs=[fraud_info_1]
                )
                generate_btn_1.click(
                    fn=generate_fraud_report,
                    inputs=[fraud_select_1, language_1],
                    outputs=[report_output]
                )
            
            # ===== TAB 2: Personalized Warning =====
            with gr.TabItem("⚠️ Personalized Warning"):
                gr.Markdown("### Generate personalized anti-fraud education for students")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        student_select_2 = gr.Dropdown(
                            choices=[(label, str(idx)) for label, idx in student_choices],
                            label="Select Student",
                            info="Choose a student to generate personalized warning"
                        )
                        language_2 = gr.Dropdown(
                            choices=["English", "Chinese (中文)"],
                            value="English",
                            label="Output Language"
                        )
                        student_info_2 = gr.Markdown("Select a student to see their profile")
                        generate_btn_2 = gr.Button("Generate Warning Message", variant="primary")
                    
                    with gr.Column(scale=2):
                        warning_output = gr.Textbox(
                            label="Personalized Warning",
                            lines=15,
                            placeholder="Warning message will appear here..."
                        )
                
                student_select_2.change(
                    fn=display_student_info,
                    inputs=[student_select_2],
                    outputs=[student_info_2]
                )
                generate_btn_2.click(
                    fn=generate_warning,
                    inputs=[student_select_2, language_2],
                    outputs=[warning_output]
                )
            
            # ===== TAB 3: Conversational Query =====
            with gr.TabItem("💬 Fraud Data Query"):
                gr.Markdown("""
### Ask questions about fraud data in any language
The system will respond in the same language you use (English or 中文).
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        query_input = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g., 'What are the main fraud patterns targeting students?' or '今天有多少学生被诈骗号码联系？'",
                            lines=3
                        )
                        query_btn = gr.Button("Ask Question", variant="primary")
                        
                        gr.Markdown("**Suggested Questions:**")
                        gr.Markdown("""
- What are the top fraud patterns we're seeing?
- How many SIM farm cases did we detect?
- Compare fraud rates between postpaid and prepaid
- 显示针对学生的诈骗模式
- 有多少高风险学生被确认欺诈号码联系过？
                        """)
                    
                    with gr.Column(scale=2):
                        query_output = gr.Textbox(
                            label="Response",
                            lines=15,
                            placeholder="Response will appear here..."
                        )
                
                query_btn.click(
                    fn=query_fraud_data,
                    inputs=[query_input],
                    outputs=[query_output]
                )
        
        # Footer
        gr.Markdown("""
---
**Data Summary (from main.py training):**
- 📊 Total Students: {:,} | Defrauded: {:,} | Fraud Rate: {:.2f}%
- ⚠️ High-Risk (demo): {:,} | Engaged: {:,}
- 🎯 Tier Distribution: CRITICAL={:,}, HIGH={:,}
        """.format(
            DATA_CONTEXT['total_students'],
            DATA_CONTEXT['defrauded_count'],
            DATA_CONTEXT['fraud_rate'],
            DATA_CONTEXT['high_risk_students'],
            DATA_CONTEXT['engaged_count'],
            DATA_CONTEXT['tier_distribution'].get('CRITICAL', 0),
            DATA_CONTEXT['tier_distribution'].get('HIGH', 0)
        ))
    
    return app


if __name__ == "__main__":
    # Set API key (for demo)
    if not os.environ.get('GROQ_API_KEY'):
        print("Warning: GROQ_API_KEY not set. Set it before running.")
        print("Run: set GROQ_API_KEY=your_api_key_here")
    
    app = create_app()
    app.launch(
        share=False,  # Set to True for public URL
        server_name="127.0.0.1",
        server_port=7860
    )
