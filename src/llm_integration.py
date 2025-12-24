"""
LLM Integration Module for Campus Anti-Fraud Detection
Wutong Cup AI+Security Competition

This module integrates Groq LLM for:
1. Risk Report Generator - Natural language explanations for fraud flags
2. Personalized Anti-Fraud Education - Targeted warning messages for students
3. Conversational Fraud Query - Natural language Q&A about fraud data

Note: Set GROQ_API_KEY environment variable or pass api_key to classes.
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    print("Warning: groq not installed. Run: pip install groq")


@dataclass
class StudentRiskProfile:
    """Student risk profile for LLM processing."""
    risk_tier: str
    vulnerability_score: float
    is_mainland_student: bool
    foreign_exposure: int
    has_fraud_contact: bool
    fraud_voice_receive: int
    any_engagement: bool
    top_risk_factors: List[str]


@dataclass
class FraudProfile:
    """Fraud profile for LLM processing."""
    is_flagged: bool
    pattern_count: int
    patterns_detected: List[str]
    triggered_rules: List[str]
    call_cnt_day: float
    is_prepaid: bool
    is_sim_farm: bool
    hit_student: bool


class GroqClient:
    """Groq API client for fast LLM inference."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key (or set GROQ_API_KEY env var)
            model: Model to use (default: llama-3.3-70b-versatile)
        """
        if not HAS_GROQ:
            raise ImportError("groq not installed. Run: pip install groq")
        
        self.api_key = api_key or os.environ.get('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set. Pass api_key or set env var.")
        
        self.client = Groq(api_key=self.api_key)
        self.model = model
    
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """Generate response from Groq with retry logic."""
        import time
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model,
                    temperature=0.7,
                    max_tokens=1024
                )
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e)
                if 'rate' in error_str.lower() or 'limit' in error_str.lower():
                    wait_time = (attempt + 1) * 5  # 5s, 10s, 15s
                    print(f"Rate limit hit. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    return f"Error generating response: {error_str}"
        
        return "Error: Rate limit exceeded after retries. Please try again later."


# =============================================================================
# FEATURE 1: RISK REPORT GENERATOR
# =============================================================================

class RiskReportGenerator:
    """
    Generate natural language risk reports for fraud flagged numbers.
    Provides explainable AI output for why a number was flagged.
    """
    
    SYSTEM_PROMPT = """You are a fraud analysis expert at a Hong Kong telecom company (CMHK).
Your task is to generate clear, professional risk reports explaining why a phone number 
has been flagged as potential fraud. Be concise but thorough.

IMPORTANT: Detect the language of the input/context:
- If input is in English, respond in English only
- If input is in Chinese (Simplified or Traditional), respond in Chinese only
- Default to English if language is unclear

Focus on:
- Key indicators that triggered the flag
- Risk level assessment
- Recommended actions

REAL-WORLD RESOURCES to include when relevant:
- Hong Kong Police Cyber Security and Technology Crime Bureau: https://www.police.gov.hk/ppp_en/04_crime_matters/tcd/
- Anti-Deception Coordination Centre (ADCC) Hotline: 18222
- Hong Kong Police Emergency: 999
- Non-emergency Police Report: 2527 7177
- Cyber Security Information Portal: https://www.cybersecurity.hk/
"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = GroqClient(api_key)
    
    def generate_report(self, fraud_profile: FraudProfile) -> str:
        """
        Generate a natural language risk report for a fraud profile.
        
        Args:
            fraud_profile: FraudProfile dataclass with fraud indicators
            
        Returns:
            Bilingual risk report string
        """
        prompt = f"""{self.SYSTEM_PROMPT}

## Fraud Profile Data:
- Flagged: {fraud_profile.is_flagged}
- Risk Patterns Detected: {fraud_profile.pattern_count} ({', '.join(fraud_profile.patterns_detected)})
- Triggered Rules: {', '.join(fraud_profile.triggered_rules)}
- Daily Call Volume: {fraud_profile.call_cnt_day}
- Account Type: {'Prepaid' if fraud_profile.is_prepaid else 'Postpaid'}
- SIM Farm Indicator: {fraud_profile.is_sim_farm}
- Student Targeting: {fraud_profile.hit_student}

Generate a professional risk report explaining why this number is flagged.
"""
        return self.client.generate(prompt)
    
    def generate_batch_summary(self, profiles: List[FraudProfile]) -> str:
        """Generate a summary report for multiple fraud cases."""
        
        total = len(profiles)
        sim_farm_count = sum(1 for p in profiles if p.is_sim_farm)
        student_targeting = sum(1 for p in profiles if p.hit_student)
        patterns = {}
        for p in profiles:
            for pattern in p.patterns_detected:
                patterns[pattern] = patterns.get(pattern, 0) + 1
        
        top_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        
        prompt = f"""{self.SYSTEM_PROMPT}

## Batch Fraud Summary:
- Total Flagged Numbers: {total}
- SIM Farm Cases: {sim_farm_count} ({sim_farm_count/total*100:.1f}%)
- Student Targeting Cases: {student_targeting} ({student_targeting/total*100:.1f}%)
- Top Patterns: {top_patterns}

Generate an executive summary of this fraud batch for the security team.
"""
        return self.client.generate(prompt)


# =============================================================================
# FEATURE 2: PERSONALIZED ANTI-FRAUD EDUCATION
# =============================================================================

class PersonalizedWarningGenerator:
    """
    Generate personalized anti-fraud education messages for at-risk students.
    Messages are tailored to the student's specific risk profile.
    """
    
    SYSTEM_PROMPT = """You are a student safety officer at a Hong Kong university.
Your task is to generate personalized, friendly but serious warning messages 
for students who may be at risk of telecom fraud.

IMPORTANT: Detect the language of the input/context:
- If student profile or context is in English, respond in English only
- If in Chinese (Simplified or Traditional), respond in Chinese only
- For mainland students (内地学生), prefer Simplified Chinese
- Default to English if language is unclear

Tone: Caring, informative, not alarming but clear about the risk.
Include: Specific risk factors, practical advice, resources for help.

REAL-WORLD RESOURCES (always include these):
- Anti-Deception Coordination Centre (ADCC) Hotline: 18222 (防骗易热线)
- Hong Kong Police Emergency: 999
- Scameter App: https://cyberdefender.hk/scameter/ (check suspicious calls/messages)
- "Scam Alert" Telegram Channel: https://t.me/CyberDefender_HKPF
- University Security Office (remind student to contact their school)
- Hong Kong Police Cyber Security: https://www.police.gov.hk/ppp_en/04_crime_matters/tcd/

Common Fraud Types in HK targeting students:
- Impersonation scams (假冒官员): Pretending to be police, immigration, or customs officers
- Online job scams (网上求职骗案): Fake jobs requiring upfront payment
- Investment scams (投资骗案): Cryptocurrency, forex trading
- Romance scams (网上情缘骗案): Building trust then asking for money
"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = GroqClient(api_key)
    
    def generate_warning(self, student_profile: StudentRiskProfile) -> str:
        """
        Generate a personalized warning message for a student.
        
        Args:
            student_profile: StudentRiskProfile dataclass
            
        Returns:
            Bilingual personalized warning message
        """
        student_type = "mainland student (内地学生)" if student_profile.is_mainland_student else "student"
        engagement_status = "already engaged with fraudster" if student_profile.any_engagement else "received contact from suspected fraud numbers"
        
        from datetime import datetime
        current_date = datetime.now().strftime('%Y年%m月%d日' if student_profile.is_mainland_student else '%B %d, %Y')
        
        prompt = f"""{self.SYSTEM_PROMPT}

## Current Date: {current_date}

## Student Risk Profile:
- Risk Level: {student_profile.risk_tier}
- Student Type: {student_type}
- Vulnerability Score: {student_profile.vulnerability_score}/100
- Foreign Call Exposure: {student_profile.foreign_exposure} calls
- Has Fraud Contact: {student_profile.has_fraud_contact}
- Status: {engagement_status}
- Key Risk Factors: {', '.join(student_profile.top_risk_factors[:3])}

Generate a personalized, caring warning message for this student.
Tailor the advice to their specific situation.
Include at least 2 real-world resources (hotline, website) they can use.
"""
        return self.client.generate(prompt)
    
    def generate_campaign_message(self, target_group: str, risk_summary: Dict) -> str:
        """
        Generate a warning message for a group campaign.
        
        Args:
            target_group: e.g., "mainland students", "new students"
            risk_summary: Summary stats about the group
            
        Returns:
            Campaign message suitable for mass distribution
        """
        prompt = f"""{self.SYSTEM_PROMPT}

## Campaign Target: {target_group}
## Risk Statistics:
{json.dumps(risk_summary, indent=2)}

Generate a general awareness message for this student group.
Include common scam tactics and prevention tips.
Make it shareable (suitable for WeChat, WhatsApp).
"""
        return self.client.generate(prompt)


# =============================================================================
# FEATURE 3: CONVERSATIONAL FRAUD QUERY INTERFACE
# =============================================================================

class FraudQueryInterface:
    """
    Natural language query interface for fraud analysis.
    Allows auditors to query the fraud database using natural language.
    """
    
    SYSTEM_PROMPT = """You are an AI fraud analyst assistant for CMHK telecom.
You help security auditors query and understand fraud data using natural language.

IMPORTANT: Respond in the SAME LANGUAGE as the user's query:
- If user asks in English, respond in English only
- If user asks in Chinese (Simplified or Traditional), respond in Chinese only
- Match the user's language exactly

You have access to the following data:
- fraud_model_2: 12,508 confirmed fraud numbers
- fraud_model_1_1: 18,341 suspected numbers
- fraud_model_1_2: 10,957 suspected numbers
- student_model: 57,713 student records
- validate_data: 4,217 test records

Available features you can analyze:
- call_cnt_day: Daily outbound calls
- hit_student_model: Whether number targeted students
- post_or_ppd: Prepaid vs Postpaid
- id_type_hk_num: Number of SIMs linked to same ID
- dispersion_rate: Call diversity
- audit_status: Audit result (稽核通過/稽核不通過)

When user asks a query, provide:
1. A clear answer to their question
2. Relevant data insights
3. Suggested follow-up queries
"""
    
    def __init__(self, api_key: Optional[str] = None, data_context: Optional[Dict] = None):
        """
        Initialize query interface.
        
        Args:
            api_key: Gemini API key
            data_context: Optional dict with summary stats about the data
        """
        self.client = GroqClient(api_key)
        self.data_context = data_context or {}
        self.conversation_history = []
    
    def query(self, user_question: str) -> str:
        """
        Process a natural language query about fraud data.
        
        Args:
            user_question: User's question in natural language
            
        Returns:
            Bilingual response with data insights
        """
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_question})
        
        # Build prompt with context
        history_context = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.conversation_history[-5:]  # Last 5 messages
        ])
        
        prompt = f"""{self.SYSTEM_PROMPT}

## Current Data Context:
{json.dumps(self.data_context, indent=2)}

## Conversation History:
{history_context}

## Current Query:
{user_question}

Provide a helpful, data-driven response. If the query requires specific data 
that you don't have, explain what data would be needed and suggest how to get it.
"""
        
        response = self.client.generate(prompt)
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def set_data_context(self, context: Dict):
        """Update the data context for more accurate responses."""
        self.data_context = context
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_suggested_queries(self) -> List[str]:
        """Get suggested queries for new users."""
        return [
            "Show me prepaid fraud cases targeting students this week",
            "What are the top 3 fraud patterns we're seeing?",
            "How many SIM farm cases did we detect?",
            "Compare fraud rates between postpaid and prepaid",
            "Which university has the most at-risk students?",
            "今天有多少学生被诈骗号码联系？",
            "显示最近一周的诈骗趋势",
        ]


# =============================================================================
# DEMO / TESTING
# =============================================================================

def demo():
    """Demonstrate the LLM integration features."""
    
    print("=" * 70)
    print("LLM INTEGRATION DEMO")
    print("Campus Anti-Fraud Solution - Wutong Cup")
    print("=" * 70)
    
    # Check for API key
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        print("\nError: GROQ_API_KEY environment variable not set.")
        print("Run: set GROQ_API_KEY=your_api_key_here (Windows)")
        print("Or:  export GROQ_API_KEY=your_api_key_here (Linux/Mac)")
        return
    
    # Demo 1: Risk Report Generator
    print("\n" + "=" * 70)
    print("DEMO 1: RISK REPORT GENERATOR")
    print("=" * 70)
    
    report_gen = RiskReportGenerator(api_key)
    
    fraud_profile = FraudProfile(
        is_flagged=True,
        pattern_count=3,
        patterns_detected=['sim_farm', 'student_targeting', 'robocall_burst'],
        triggered_rules=['rule_05_student', 'rule_06_sim_farm', 'rule_01_high_volume'],
        call_cnt_day=45.0,
        is_prepaid=True,
        is_sim_farm=True,
        hit_student=True
    )
    
    print("\nGenerating risk report...")
    report = report_gen.generate_report(fraud_profile)
    print("\n--- RISK REPORT ---")
    print(report)
    
    # Demo 2: Personalized Warning
    print("\n" + "=" * 70)
    print("DEMO 2: PERSONALIZED WARNING GENERATOR")
    print("=" * 70)
    
    warning_gen = PersonalizedWarningGenerator(api_key)
    
    student_profile = StudentRiskProfile(
        risk_tier='HIGH',
        vulnerability_score=75.0,
        is_mainland_student=True,
        foreign_exposure=12,
        has_fraud_contact=True,
        fraud_voice_receive=3,
        any_engagement=False,
        top_risk_factors=['mainland_student', 'high_foreign_exposure', 'repeat_unknown_caller']
    )
    
    print("\nGenerating personalized warning...")
    warning = warning_gen.generate_warning(student_profile)
    print("\n--- PERSONALIZED WARNING ---")
    print(warning)
    
    # Demo 3: Conversational Query
    print("\n" + "=" * 70)
    print("DEMO 3: CONVERSATIONAL QUERY INTERFACE")
    print("=" * 70)
    
    query_interface = FraudQueryInterface(
        api_key,
        data_context={
            'total_fraud': 12508,
            'student_at_risk': 836,
            'sim_farm_cases': 11243,
            'student_targeting': 222,
            'validation_recall': '83.8%'
        }
    )
    
    print("\nSuggested queries:")
    for q in query_interface.get_suggested_queries()[:3]:
        print(f"  - {q}")
    
    print("\nProcessing query: 'What are the main fraud patterns targeting students?'")
    response = query_interface.query("What are the main fraud patterns targeting students?")
    print("\n--- QUERY RESPONSE ---")
    print(response)
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    demo()
