"""
Refined Clinical Named Entity Recognizer & Span Categorizer.
Solves boundary ambiguities, honorific misclassifications, and category bleeding.
Accurately decomposes and categorizes into the 5 target clinical categories:
1. SYMPTOM
2. RISK_INDICATOR
3. SEVERITY
4. DURATION
5. BODY_PART
"""

import re
from typing import List, Dict, Any

RISK_INDICATORS = [
    "high blood pressure",
    "blood pressure",
    "cardiac patient",
    "heart patient",
    "on blood thinners",
    "blood thinners",
    "previous heart attack",
    "heart attack",
    "chronic smoker",
    "heavy smoker",
    "smoker",
    "diabetic patient",
    "known diabetic",
    "diabetic",
    "diabetes",
    "hypertensive",
    "hypertension",
    "asthma history",
    "asthma patient",
    "asthma",
    "chronic kidney disease",
    "kidney disease",
    "cancer patient",
    "cancer",
    "stroke",
    "pregnant 7 months",
    "pregnant",
    "unconscious",
    "collapse",
    "chronic alcoholic",
    "alcoholic"
]

# Anatomical body locations (excluding ambiguous honorifics like standalone 'Sir')
BODY_PARTS = [
    "left arm",
    "right arm",
    "upper abdominal",
    "abdominal",
    "throat",
    "gala",
    "chest",
    "chaati",
    "nenju",
    "stomach",
    "abdomen",
    "pet",
    "hotte",
    "head",
    "thale",
    "tale",
    "thala",
    "left leg",
    "right leg",
    "leg",
    "kaalu",
    "back",
    "kamar",
    "neck",
    "lungs",
    "eyes",
    "kannu",
    "joints"
]

SEVERITIES = [
    "bahut tez",
    "bohot zyada",
    "thumba jaasthi",
    "unbearable",
    "excruciating",
    "severe",
    "acute",
    "extreme",
    "continuous",
    "persistent",
    "moderate",
    "heavy",
    "thumba",
    "romba",
    "jaasthi",
    "halka halka",
    "halka",
    "swalpa",
    "slight",
    "mild",
    "sudden"
]

SYMPTOMS = [
    # Multi-word and compound symptoms
    "breathlessness",
    "shortness of breath",
    "saas lene mein dikkat",
    "saas lene me dikkat",
    "saans fulna",
    "saans phoolna",
    "breathing difficulty",
    "breathing problem",
    "dry cough",
    "blood in cough",
    "sore throat",
    "gala kharab",
    "infection",
    "swallowing difficulty",
    "swallowing pain",
    "headache",
    "head ache",
    "sir dard",
    "sar dard",
    "thale novu",
    "tale novu",
    "thala vali",
    "nenju vali",
    "vomiting sensation",
    "loose motion",
    "burning sensation",
    "cramping",
    "cramp",
    "swelling",
    "dizziness",
    "chakkar",
    "chakkarr",
    "shivering",
    "kampa",
    "chali",
    "thand lagna",
    "weakness",
    "very weak",
    "cough",
    "khansi",
    "vomiting",
    "ulti",
    "fever",
    "tightness",
    "pressure",
    "pain",
    "dard",
    "novu",
    "vali"
]

DURATION_PATTERNS = [
    r'\b(?:since\s+(?:morning|yesterday|last\s+night|night|\d+\s+(?:days?|weeks?|months?|years?|hours?)))\b',
    r'\b(?:pichle\s+\d+\s+(?:days?|din|hafte|weeks?|ghante|hours?)\s*(?:se|inda)?)\b',
    r'\b(?:\d+\s+(?:days?|din|hafte|weeks?|ghante|hours?|years?|saal)\s*se)\b',
    r'\b(?:last\s+\d+\s+(?:days?|dina|weeks?|hours?)\s*inda)\b',
    r'\b(?:past\s+\d+\s+(?:days?|weeks?|months?|years?|hours?)(?:\s+ah)?)\b',
    r'\b(?:for\s+(?:the\s+last\s+|past\s+)?\d+\s+(?:days?|hours?|weeks?|months?|years?))\b',
    r'\b(?:kal\s+raat\s+se|subah\s+se|ninne\s+inda|nethu\s+night)\b',
    r'\b(?:for\s+\d+\s+years)\b',
    r'\b(?:since\s+\d+\s+years)\b'
]

class RefinedClinicalNER:
    def __init__(self):
        pass

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        t_lower = text.lower()
        spans = []

        # 1. Durations (Highest priority for multi-word temporal expressions)
        for pat in DURATION_PATTERNS:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                spans.append({
                    "start": m.start(),
                    "end": m.end(),
                    "text": text[m.start():m.end()],
                    "label": "DURATION",
                    "priority": 10
                })

        # 2. Risk Indicators (high priority multi-word matches)
        for term in RISK_INDICATORS:
            for m in re.finditer(r'\b' + re.escape(term) + r'\b', t_lower):
                spans.append({
                    "start": m.start(),
                    "end": m.end(),
                    "text": text[m.start():m.end()],
                    "label": "RISK_INDICATOR",
                    "priority": 9
                })

        # 3. Body Parts (high priority anatomical localization)
        for term in BODY_PARTS:
            for m in re.finditer(r'\b' + re.escape(term) + r'\b', t_lower):
                spans.append({
                    "start": m.start(),
                    "end": m.end(),
                    "text": text[m.start():m.end()],
                    "label": "BODY_PART",
                    "priority": 8
                })

        # 4. Symptoms
        for term in SYMPTOMS:
            for m in re.finditer(r'\b' + re.escape(term) + r'\b', t_lower):
                s, e = m.start(), m.end()
                # Disambiguate 'pressure' inside 'blood pressure'
                if term == "pressure":
                    prefix = t_lower[max(0, s-12):s]
                    if "blood" in prefix:
                        continue
                spans.append({
                    "start": s,
                    "end": e,
                    "text": text[s:e],
                    "label": "SYMPTOM",
                    "priority": 7
                })

        # 5. Severities
        for term in SEVERITIES:
            for m in re.finditer(r'\b' + re.escape(term) + r'\b', t_lower):
                spans.append({
                    "start": m.start(),
                    "end": m.end(),
                    "text": text[m.start():m.end()],
                    "label": "SEVERITY",
                    "priority": 6
                })

        # Resolve overlaps by priority and length
        spans.sort(key=lambda x: (x["priority"], x["end"] - x["start"]), reverse=True)
        accepted = []

        for candidate in spans:
            cs, ce = candidate["start"], candidate["end"]
            collision = False
            for acc in accepted:
                as_, ae_ = acc["start"], acc["end"]
                if not (ce <= as_ or cs >= ae_):
                    collision = True
                    break
            if not collision:
                accepted.append(candidate)

        # Sort in text order
        accepted.sort(key=lambda x: x["start"])

        return [
            {
                "label": a["label"],
                "text": a["text"],
                "start": a["start"],
                "end": a["end"]
            }
            for a in accepted
        ]
