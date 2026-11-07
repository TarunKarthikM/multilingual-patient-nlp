"""
Clinical Risk Assessment & Triage Engine.
Evaluates extracted symptoms, risk indicators, severity, and duration
to calculate a Risk Stratification Level (CRITICAL / HIGH / MODERATE / LOW)
and an early triage score (0-100), strictly for decision-support as scoped in SDG 3.
"""

from typing import List, Dict, Any

RED_FLAG_INDICATORS = {
    "unconscious", "collapse", "blood in cough", "chest pain radiating to arm",
    "previous heart attack", "stroke", "blood thinners", "cancer patient",
    "cardiac patient", "pregnant"
}

HIGH_RISK_SYMPTOMS = {
    "chest pain", "breathlessness", "saas lene mein dikkat", "shortness of breath",
    "unconscious", "collapse", "dizziness", "chakkar"
}

SEVERITY_WEIGHTS = {
    "severe": 25,
    "acute": 25,
    "extreme": 30,
    "unbearable": 30,
    "bahut tez": 25,
    "bohot zyada": 25,
    "thumba jaasthi": 25,
    "romba": 25,
    "moderate": 15,
    "continuous": 18,
    "persistent": 18,
    "mild": 5,
    "halka": 5,
    "swalpa": 5
}

class ClinicalRiskEngine:
    def __init__(self):
        pass

    def evaluate_risk(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates risk score (0-100) and assigns triage tier.
        """
        score = 10 # Baseline presentation score
        reasons = []
        red_flags_found = []

        # Categorize extracted entities
        symptoms = [e["text"].lower() for e in entities if e["label"] == "SYMPTOM"]
        risk_indicators = [e["text"].lower() for e in entities if e["label"] == "RISK_INDICATOR"]
        severities = [e["text"].lower() for e in entities if e["label"] == "SEVERITY"]
        durations = [e["text"].lower() for e in entities if e["label"] == "DURATION"]
        body_parts = [e["text"].lower() for e in entities if e["label"] == "BODY_PART"]

        # 1. Evaluate Red Flags in Risk Indicators
        for rk in risk_indicators:
            matched_flag = False
            for rf in RED_FLAG_INDICATORS:
                if rf in rk:
                    score += 35
                    red_flags_found.append(rk)
                    reasons.append(f"Critical risk indicator detected: '{rk}'")
                    matched_flag = True
                    break
            if not matched_flag:
                score += 15
                reasons.append(f"Pre-existing risk factor noted: '{rk}'")

        # 2. Evaluate Symptoms
        for sym in symptoms:
            for hs in HIGH_RISK_SYMPTOMS:
                if hs in sym:
                    score += 25
                    reasons.append(f"High-priority symptom identified: '{sym}'")
                    break
            else:
                score += 10

        # 3. Evaluate Severity modifiers
        max_sev_weight = 0
        for sev in severities:
            for s_key, w in SEVERITY_WEIGHTS.items():
                if s_key in sev and w > max_sev_weight:
                    max_sev_weight = w
        if max_sev_weight > 0:
            score += max_sev_weight
            reasons.append(f"Severity modifier elevates presentation urgency (+{max_sev_weight} pts)")

        # 4. Critical combination rules (Cardiac / Respiratory triage)
        is_chest_complaint = any("chest" in b or "chaati" in b or "nenju" in b for b in body_parts) or \
                             any("chest" in s or "nenju" in s for s in symptoms)
        has_cardiac_or_bp = any("heart" in r or "blood pressure" in r or "bp" in r or "cardiac" in r for r in risk_indicators)
        has_severe = any(w >= 20 for w in [max_sev_weight])

        if is_chest_complaint and (has_cardiac_or_bp or has_severe):
            score += 25
            red_flags_found.append("Potential Acute Coronary / Cardiopulmonary Syndrome")
            reasons.append("Red flag: Chest symptoms combined with cardiac history / severe intensity.")

        # Cap score at 100
        score = min(max(score, 5), 100)

        # Determine Tier
        if score >= 75 or len(red_flags_found) >= 2:
            tier = "CRITICAL"
            action = "Immediate Emergency / Red Triage Assessment Required"
            badge_class = "danger"
        elif score >= 50:
            tier = "HIGH"
            action = "Urgent Outpatient / Clinical Review Recommended"
            badge_class = "warning"
        elif score >= 25:
            tier = "MODERATE"
            action = "Standard Consultation & Symptomatic Monitoring"
            badge_class = "info"
        else:
            tier = "LOW"
            action = "Routine Care / Home Observation"
            badge_class = "success"

        return {
            "risk_score": score,
            "risk_level": tier,
            "recommended_urgency": action,
            "badge_class": badge_class,
            "red_flags": list(set(red_flags_found)),
            "clinical_rationale": reasons if reasons else ["No acute alarm symptoms identified."]
        }

if __name__ == "__main__":
    engine = ClinicalRiskEngine()
    test_entities = [
        {"label": "BODY_PART", "text": "chest"},
        {"label": "SYMPTOM", "text": "pain"},
        {"label": "SEVERITY", "text": "severe"},
        {"label": "RISK_INDICATOR", "text": "diabetic patient"},
        {"label": "RISK_INDICATOR", "text": "previous heart attack"}
    ]
    print(engine.evaluate_risk(test_entities))
