"""
Interactive CLI Demonstration Script for Review-1 Presentation.
Evaluates sample code-mixed clinical queries across English, Hinglish, Kanglish, and Tanglish.
Outputs formatted clinical extractions and early risk assessments.
"""

import os
import sys
import json

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from extraction.extractor import PatientInformationExtractor

DEMO_CASES = [
    {
        "title": "Hinglish Acute Presentation (Cardiac & Diabetic Red-Flag)",
        "query": "Doctor saab, chest mein bahut tez heavy pressure feel ho raha hai since morning and patient is known diabetic with high blood pressure."
    },
    {
        "title": "Kanglish Outpatient Case (Fever & Throat Infection)",
        "query": "Nanage last 2 days inda severe fever ide and throat full infection aagide, swallowing madoke thumba novu barthide."
    },
    {
        "title": "Tanglish Chronic Cardiac Patient (Chest Pain on Blood Thinners)",
        "query": "Enaku past 4 days ah severe chest pain irukku, and I am cardiac patient on blood thinners since 2 years."
    },
    {
        "title": "Hinglish GI Symptom (Abdominal Cramping & Vomiting)",
        "query": "Sir mere pet mein unbearable cramping ho rahi hai since 2 days, and bohot zyada continuous ulti aur weakness hai."
    },
    {
        "title": "English Respiratory Distress (Chronic Smoker)",
        "query": "Patient is chronic smoker for 15 years, complaining of acute breathlessness and chest tightness for the last 6 hours."
    }
]

def run_cli_demo():
    print("=" * 80)
    print("DAYANANDA SAGAR UNIVERSITY - SCHOOL OF ENGINEERING")
    print("DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING (AI&ML)")
    print("Course: Natural Language Models (24AM3503) · Review-1 Demonstration")
    print("Project: Multilingual Patient Symptom & Risk-Indicator Extraction Framework")
    print("=" * 80)
    print("\nInitializing NLP Extraction Pipeline...")

    extractor = PatientInformationExtractor()

    for idx, case in enumerate(DEMO_CASES, 1):
        print("\n" + "-" * 80)
        print(f"CASE {idx}: {case['title']}")
        print(f"Input Text: \"{case['query']}\"")
        print("-" * 80)

        result = extractor.process(case["query"], model_name="Ensemble (Consensus)")
        ling = result["linguistic_analysis"]
        risk = result["clinical_risk_assessment"]
        findings = result["structured_findings"]

        print(f"• Detected Language : {ling['detected_language']}")
        print(f"• Code-Mixing Index : {ling['code_mixing_index'] * 100:.1f}%")
        print(f"• Triage Risk Level : [{risk['risk_level']}] (Score: {risk['risk_score']}/100)")
        print(f"• Urgency Action    : {risk['recommended_urgency']}")

        print("\n  Extracted Clinical Entities:")
        print(f"    - Symptoms       : {[e['text'] for e in findings['symptoms']]}")
        print(f"    - Risk Indicators: {[e['text'] for e in findings['risk_indicators']]}")
        print(f"    - Severity       : {[e['text'] for e in findings['severity']]}")
        print(f"    - Duration       : {[e['text'] for e in findings['duration']]}")
        print(f"    - Body Parts     : {[e['text'] for e in findings['body_parts']]}")

        if risk["clinical_rationale"]:
            print("\n  Clinical Decision Support Rationale:")
            for r in risk["clinical_rationale"]:
                print(f"    * {r}")

    print("\n" + "=" * 80)
    print("Demonstration successfully completed!")
    print("=" * 80)

if __name__ == "__main__":
    run_cli_demo()
