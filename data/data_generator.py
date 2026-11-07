"""
Data Generator and Dataset Builder for Multilingual Code-Mixed Patient Text.
Entities Annotated:
1. SYMPTOM (e.g., fever, sir dard, chest pain, khansi, breathlessness)
2. RISK_INDICATOR (e.g., diabetic patient, high blood pressure, smoker, past heart attack)
3. SEVERITY (e.g., severe, bahut tez, mild, acute, unbearable, halka halka)
4. DURATION (e.g., 3 days se, pichle ek hafte se, since yesterday, 2 vaara inda)
5. BODY_PART (e.g., chest, gala, pet, head, kaalu, nenju)

Languages Covered:
- Hinglish (Hindi + English Romanized & Devanagari)
- Kanglish (Kannada + English Romanized & Kannada script)
- Tanglish (Tamil + English Romanized)
- English (Clinical colloquial)
"""

import json
import random
import os
import re

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Curated high-fidelity annotated clinical queries across language groups
SEED_PATIENT_RECORDS = [
    # --- Hinglish (Romanized Hindi + English) ---
    {
        "text": "Mujhe pichle 3 days se bahut tez headache ho raha hai and gala mein severe pain hai.",
        "language": "Hinglish",
        "script": "Latin",
        "entities": [
            {"start": 6, "end": 23, "label": "DURATION", "text": "pichle 3 days se"},
            {"start": 24, "end": 33, "label": "SEVERITY", "text": "bahut tez"},
            {"start": 34, "end": 42, "label": "SYMPTOM", "text": "headache"},
            {"start": 60, "end": 64, "label": "BODY_PART", "text": "gala"},
            {"start": 70, "end": 76, "label": "SEVERITY", "text": "severe"},
            {"start": 77, "end": 81, "label": "SYMPTOM", "text": "pain"}
        ]
    },
    {
        "text": "Patient is a known diabetic since 5 years and now having extreme chest pain radiating to left arm since morning.",
        "language": "English",
        "script": "Latin",
        "entities": [
            {"start": 19, "end": 27, "label": "RISK_INDICATOR", "text": "diabetic"},
            {"start": 34, "end": 41, "label": "DURATION", "text": "5 years"},
            {"start": 57, "end": 64, "label": "SEVERITY", "text": "extreme"},
            {"start": 65, "end": 70, "label": "BODY_PART", "text": "chest"},
            {"start": 71, "end": 75, "label": "SYMPTOM", "text": "pain"},
            {"start": 89, "end": 97, "label": "BODY_PART", "text": "left arm"},
            {"start": 98, "end": 111, "label": "DURATION", "text": "since morning"}
        ]
    },
    {
        "text": "Sir mere pet mein unbearable cramping ho rahi hai since 2 days, and thoda mild fever bhi hai.",
        "language": "Hinglish",
        "script": "Latin",
        "entities": [
            {"start": 9, "end": 12, "label": "BODY_PART", "text": "pet"},
            {"start": 18, "end": 28, "label": "SEVERITY", "text": "unbearable"},
            {"start": 29, "end": 37, "label": "SYMPTOM", "text": "cramping"},
            {"start": 50, "end": 62, "label": "DURATION", "text": "since 2 days"},
            {"start": 74, "end": 78, "label": "SEVERITY", "text": "mild"},
            {"start": 79, "end": 84, "label": "SYMPTOM", "text": "fever"}
        ]
    },
    {
        "text": "Mummy ko high blood pressure ki problem hai aur kal raat se acute breathlessness aur dry cough hai.",
        "language": "Hinglish",
        "script": "Latin",
        "entities": [
            {"start": 9, "end": 28, "label": "RISK_INDICATOR", "text": "high blood pressure"},
            {"start": 47, "end": 59, "label": "DURATION", "text": "kal raat se"},
            {"start": 60, "end": 65, "label": "SEVERITY", "text": "acute"},
            {"start": 66, "end": 80, "label": "SYMPTOM", "text": "breathlessness"},
            {"start": 85, "end": 94, "label": "SYMPTOM", "text": "dry cough"}
        ]
    },
    {
        "text": "Bhai mujhe pichle 1 week se continuous ulti aur loose motion ho raha hai, feeling very weak.",
        "language": "Hinglish",
        "script": "Latin",
        "entities": [
            {"start": 11, "end": 24, "label": "DURATION", "text": "pichle 1 week"},
            {"start": 28, "end": 38, "label": "SEVERITY", "text": "continuous"},
            {"start": 39, "end": 43, "label": "SYMPTOM", "text": "ulti"},
            {"start": 48, "end": 60, "label": "SYMPTOM", "text": "loose motion"},
            {"start": 82, "end": 91, "label": "SYMPTOM", "text": "very weak"}
        ]
    },
    {
        "text": "Doctor saab, chest mein heavy pressure feel ho raha hai and patient is chronic smoker for 10 years.",
        "language": "Hinglish",
        "script": "Latin",
        "entities": [
            {"start": 13, "end": 18, "label": "BODY_PART", "text": "chest"},
            {"start": 24, "end": 38, "label": "SYMPTOM", "text": "heavy pressure"},
            {"start": 71, "end": 85, "label": "RISK_INDICATOR", "text": "chronic smoker"},
            {"start": 90, "end": 98, "label": "DURATION", "text": "10 years"}
        ]
    },
    {
        "text": "Mere left leg mein moderate swelling hai since yesterday, चलने में bahut pain ho raha hai.",
        "language": "Hinglish",
        "script": "Mixed",
        "entities": [
            {"start": 5, "end": 13, "label": "BODY_PART", "text": "left leg"},
            {"start": 19, "end": 27, "label": "SEVERITY", "text": "moderate"},
            {"start": 28, "end": 36, "label": "SYMPTOM", "text": "swelling"},
            {"start": 41, "end": 56, "label": "DURATION", "text": "since yesterday"},
            {"start": 68, "end": 73, "label": "SEVERITY", "text": "bahut"},
            {"start": 74, "end": 78, "label": "SYMPTOM", "text": "pain"}
        ]
    },
    # --- Kanglish (Kannada + English Romanized & Regional context) ---
    {
        "text": "Nanage last 2 days inda severe fever ide and throat full infection aagide.",
        "language": "Kanglish",
        "script": "Latin",
        "entities": [
            {"start": 7, "end": 22, "label": "DURATION", "text": "last 2 days inda"},
            {"start": 23, "end": 29, "label": "SEVERITY", "text": "severe"},
            {"start": 30, "end": 35, "label": "SYMPTOM", "text": "fever"},
            {"start": 44, "end": 50, "label": "BODY_PART", "text": "throat"},
            {"start": 56, "end": 65, "label": "SYMPTOM", "text": "infection"}
        ]
    },
    {
        "text": "Patient ge asthma history ide, eega sudden aagi breathing difficulty barthide since 1 hour.",
        "language": "Kanglish",
        "script": "Latin",
        "entities": [
            {"start": 11, "end": 25, "label": "RISK_INDICATOR", "text": "asthma history"},
            {"start": 36, "end": 42, "label": "SEVERITY", "text": "sudden"},
            {"start": 48, "end": 68, "label": "SYMPTOM", "text": "breathing difficulty"},
            {"start": 78, "end": 90, "label": "DURATION", "text": "since 1 hour"}
        ]
    },
    {
        "text": "Nanna thale li unbearable novu ide pichle 3 ghante inda, dizziness kuda aagthide.",
        "language": "Kanglish",
        "script": "Latin",
        "entities": [
            {"start": 6, "end": 11, "label": "BODY_PART", "text": "thale"},
            {"start": 15, "end": 25, "label": "SEVERITY", "text": "unbearable"},
            {"start": 26, "end": 30, "label": "SYMPTOM", "text": "novu"},
            {"start": 35, "end": 54, "label": "DURATION", "text": "pichle 3 ghante inda"},
            {"start": 56, "end": 65, "label": "SYMPTOM", "text": "dizziness"}
        ]
    },
    {
        "text": "Hotte novu thumba jaasthi aagide since night, mild vomiting sensation ide.",
        "language": "Kanglish",
        "script": "Latin",
        "entities": [
            {"start": 0, "end": 5, "label": "BODY_PART", "text": "Hotte"},
            {"start": 6, "end": 10, "label": "SYMPTOM", "text": "novu"},
            {"start": 11, "end": 25, "label": "SEVERITY", "text": "thumba jaasthi"},
            {"start": 33, "end": 44, "label": "DURATION", "text": "since night"},
            {"start": 46, "end": 50, "label": "SEVERITY", "text": "mild"},
            {"start": 51, "end": 69, "label": "SYMPTOM", "text": "vomiting sensation"}
        ]
    },
    # --- Tanglish (Tamil + English Romanized) ---
    {
        "text": "Enaku past 4 days ah severe chest pain irukku, and I am cardiac patient on blood thinners.",
        "language": "Tanglish",
        "script": "Latin",
        "entities": [
            {"start": 6, "end": 21, "label": "DURATION", "text": "past 4 days ah"},
            {"start": 22, "end": 28, "label": "SEVERITY", "text": "severe"},
            {"start": 29, "end": 34, "label": "BODY_PART", "text": "chest"},
            {"start": 35, "end": 39, "label": "SYMPTOM", "text": "pain"},
            {"start": 56, "end": 71, "label": "RISK_INDICATOR", "text": "cardiac patient"},
            {"start": 75, "end": 89, "label": "RISK_INDICATOR", "text": "blood thinners"}
        ]
    },
    {
        "text": "Appa ku romba head ache and high fever irukku since 2 days, unconscious aagi collapse aagitaaru.",
        "language": "Tanglish",
        "script": "Latin",
        "entities": [
            {"start": 8, "end": 13, "label": "SEVERITY", "text": "romba"},
            {"start": 14, "end": 23, "label": "SYMPTOM", "text": "head ache"},
            {"start": 28, "end": 32, "label": "SEVERITY", "text": "high"},
            {"start": 33, "end": 38, "label": "SYMPTOM", "text": "fever"},
            {"start": 46, "end": 58, "label": "DURATION", "text": "since 2 days"},
            {"start": 60, "end": 71, "label": "RISK_INDICATOR", "text": "unconscious"},
            {"start": 77, "end": 85, "label": "RISK_INDICATOR", "text": "collapse"}
        ]
    },
    # --- Native Scripts (Devanagari & Kannada) ---
    {
        "text": "मरीज को 3 दिन से तेज बुखार है और छाती में severe tightness महसूस हो रही है।",
        "language": "Hindi-Mixed",
        "script": "Devanagari-Latin",
        "entities": [
            {"start": 8, "end": 13, "label": "DURATION", "text": "3 दिन"},
            {"start": 17, "end": 20, "label": "SEVERITY", "text": "तेज"},
            {"start": 21, "end": 26, "label": "SYMPTOM", "text": "बुखार"},
            {"start": 33, "end": 38, "label": "BODY_PART", "text": "छाती"},
            {"start": 43, "end": 49, "label": "SEVERITY", "text": "severe"},
            {"start": 50, "end": 59, "label": "SYMPTOM", "text": "tightness"}
        ]
    },
    {
        "text": "ನನಗೆ 2 ದಿನದಿಂದ severe headache ಇದೆ ಮತ್ತು ಕಣ್ಣುಗಳಲ್ಲಿ burning sensation ಆಗ್ತಿದೆ.",
        "language": "Kannada-Mixed",
        "script": "Kannada-Latin",
        "entities": [
            {"start": 5, "end": 14, "label": "DURATION", "text": "2 ದಿನದಿಂದ"},
            {"start": 15, "end": 21, "label": "SEVERITY", "text": "severe"},
            {"start": 22, "end": 30, "label": "SYMPTOM", "text": "headache"},
            {"start": 40, "end": 51, "label": "BODY_PART", "text": "ಕಣ್ಣುಗಳಲ್ಲಿ"},
            {"start": 52, "end": 69, "label": "SYMPTOM", "text": "burning sensation"}
        ]
    },
    # --- English Standard & Teleconsultation ---
    {
        "text": "Complaining of persistent dry cough and mild sore throat for 5 days, no past medical illness.",
        "language": "English",
        "script": "Latin",
        "entities": [
            {"start": 15, "end": 25, "label": "SEVERITY", "text": "persistent"},
            {"start": 26, "end": 35, "label": "SYMPTOM", "text": "dry cough"},
            {"start": 40, "end": 44, "label": "SEVERITY", "text": "mild"},
            {"start": 45, "end": 56, "label": "SYMPTOM", "text": "sore throat"},
            {"start": 61, "end": 67, "label": "DURATION", "text": "5 days"}
        ]
    },
    {
        "text": "Severe upper abdominal pain radiating to back for past 12 hours with chronic alcoholic history.",
        "language": "English",
        "script": "Latin",
        "entities": [
            {"start": 0, "end": 6, "label": "SEVERITY", "text": "Severe"},
            {"start": 7, "end": 22, "label": "BODY_PART", "text": "upper abdominal"},
            {"start": 23, "end": 27, "label": "SYMPTOM", "text": "pain"},
            {"start": 41, "end": 45, "label": "BODY_PART", "text": "back"},
            {"start": 50, "end": 63, "label": "DURATION", "text": "past 12 hours"},
            {"start": 69, "end": 86, "label": "RISK_INDICATOR", "text": "chronic alcoholic"}
        ]
    }
]

# Vocabulary components for realistic combinatorial augmentation
VOCAB = {
    "durations": {
        "Hinglish": ["pichle 2 din se", "3 days se", "kal raat se", "ek hafte se", "pichle 4 ghante se", "subah se", "do hafte se"],
        "Kanglish": ["last 2 days inda", "3 dina inda", "ninne inda", "eradu varada inda", "beligge inda", "1 hour inda"],
        "Tanglish": ["past 3 days ah", "nethu night la irundhu", "2 weeks ah", "kalaila irundhu", "one month ah"],
        "English": ["for 3 days", "since yesterday", "for past 2 weeks", "since morning", "for the last 48 hours", "for 5 days"]
    },
    "severities": {
        "Hinglish": ["bahut tez", "severe", "mild", "halka halka", "unbearable", "bohot zyada", "acute"],
        "Kanglish": ["severe", "thumba jaasthi", "swalpa", "mild", "unbearable", "heavy"],
        "Tanglish": ["romba", "severe", "mild", "adhigama", "extreme", "chinna"],
        "English": ["severe", "mild", "acute", "moderate", "excruciating", "slight", "intense"]
    },
    "symptoms": [
        {"term": "fever", "body": "body"},
        {"term": "headache", "body": "head"},
        {"term": "sir dard", "body": "sir"},
        {"term": "chest pain", "body": "chest"},
        {"term": "breathlessness", "body": "chest"},
        {"term": "saas lene mein dikkat", "body": "chest"},
        {"term": "cough", "body": "throat"},
        {"term": "khansi", "body": "throat"},
        {"term": "sore throat", "body": "throat"},
        {"term": "gala kharab", "body": "gala"},
        {"term": "pet dard", "body": "pet"},
        {"term": "stomach pain", "body": "stomach"},
        {"term": "hotte novu", "body": "hotte"},
        {"term": "vomiting", "body": "stomach"},
        {"term": "ulti", "body": "pet"},
        {"term": "loose motion", "body": "stomach"},
        {"term": "swelling", "body": "leg"},
        {"term": "kaalu novu", "body": "kaalu"},
        {"term": "dizziness", "body": "head"},
        {"term": "chakkarr", "body": "sir"},
        {"term": "shivering", "body": "body"},
        {"term": "kampa", "body": "body"},
        {"term": "joint pain", "body": "joints"},
        {"term": "kamar dard", "body": "kamar"},
        {"term": "back pain", "body": "back"}
    ],
    "risk_indicators": [
        "diabetic patient",
        "high blood pressure",
        "hypertensive",
        "chronic smoker",
        "asthma patient",
        "previous heart attack",
        "pregnant 7 months",
        "kidney disease",
        "cancer patient",
        "on blood thinners",
        "unconscious",
        "chest pain radiating to arm",
        "blood in cough"
    ],
    "templates": {
        "Hinglish": [
            "Mujhe {duration} {severity} {symptom} ho raha hai aur {body_part} mein dard hai.",
            "Patient is {risk_indicator}, {duration} se {severity} {symptom} aur weakness feel ho rahi hai.",
            "Doctor sahab, {body_part} mein {severity} {symptom} hai {duration}, patient ko {risk_indicator} bhi hai.",
            "Mummy ko {duration} {severity} {symptom} hai aur unko {risk_indicator} ki history hai.",
            "Sir {body_part} me {severity} {symptom} hai {duration} and I am a {risk_indicator}.",
            "Pichle {duration} {symptom} bohot badh gaya hai, especially in {body_part}."
        ],
        "Kanglish": [
            "Nanage {duration} {severity} {symptom} aagide, mathu {body_part} novu thumba ide.",
            "Patient ge {risk_indicator} ide, {duration} {severity} {symptom} barthide.",
            "Doctor, {body_part} li {severity} {symptom} ide {duration}, patient is {risk_indicator}.",
            "Nanna friend ge {duration} {severity} {symptom} aagide, he has {risk_indicator}."
        ],
        "Tanglish": [
            "Enaku {duration} {severity} {symptom} irukku, {body_part} romba valikuthu.",
            "Patient ku {risk_indicator} irukku, {duration} {severity} {symptom} irukku.",
            "Doctor, {body_part} la {severity} {symptom} {duration} irukku, patient is {risk_indicator}."
        ],
        "English": [
            "Patient reports {severity} {symptom} in {body_part} {duration}, with known history of {risk_indicator}.",
            "Experiencing {severity} {symptom} {duration}, patient is a {risk_indicator}.",
            "Complaints of {severity} {symptom} localized to {body_part} {duration}, diagnosed with {risk_indicator}."
        ]
    }
}

def build_annotated_example(template, duration, severity, symptom_obj, body_part, risk, lang):
    """Fills a template and dynamically calculates exact character span offsets for entities."""
    symptom = symptom_obj["term"]
    # Formulate populated string
    # We will build it segment by segment to guarantee 100% exact offsets
    parts = []
    entities = []

    # Token-safe string replacement with offset tracking
    text = template.format(
        duration=duration,
        severity=severity,
        symptom=symptom,
        body_part=body_part,
        risk_indicator=risk
    )

    # Find spans
    spans_to_find = [
        (symptom, "SYMPTOM"),
        (risk, "RISK_INDICATOR"),
        (severity, "SEVERITY"),
        (duration, "DURATION"),
        (body_part, "BODY_PART")
    ]

    for val, label in spans_to_find:
        if val in text:
            # find all non-overlapping matches
            for m in re.finditer(re.escape(val), text):
                s, e = m.start(), m.end()
                # Check overlap
                overlap = False
                for ex in entities:
                    if not (e <= ex["start"] or s >= ex["end"]):
                        overlap = True
                        break
                if not overlap:
                    entities.append({"start": s, "end": e, "label": label, "text": val})

    # Sort entities by start
    entities.sort(key=lambda x: x["start"])

    return {
        "text": text,
        "language": lang,
        "script": "Latin",
        "entities": entities
    }

def tokenize_with_bio(text, entities):
    """
    Tokenizes text into words and generates corresponding BIO tags.
    """
    # Simple regex tokenizer that preserves words and punctuation
    token_spans = []
    for match in re.finditer(r'\w+|[^\w\s]', text, re.UNICODE):
        token_spans.append((match.group(), match.start(), match.end()))

    tokens = []
    bio_tags = []

    for token, t_start, t_end in token_spans:
        assigned_tag = "O"
        for ent in entities:
            e_start = ent["start"]
            e_end = ent["end"]
            label = ent["label"]
            
            # Check overlap between token and entity
            if t_start >= e_start and t_end <= e_end:
                if t_start == e_start or len(bio_tags) == 0 or bio_tags[-1] == "O" or not bio_tags[-1].endswith(label):
                    assigned_tag = f"B-{label}"
                else:
                    assigned_tag = f"I-{label}"
                break
            elif t_start < e_end and t_end > e_start: # partial overlap
                assigned_tag = f"I-{label}"
                break

        tokens.append(token)
        bio_tags.append(assigned_tag)

    return tokens, bio_tags

def generate_synthetic_dataset(target_size=400, seed=42):
    random.seed(seed)
    dataset = list(SEED_PATIENT_RECORDS)

    languages = ["Hinglish", "Kanglish", "Tanglish", "English"]
    weights = [0.40, 0.25, 0.15, 0.20]

    while len(dataset) < target_size:
        lang = random.choices(languages, weights=weights)[0]
        tmpl = random.choice(VOCAB["templates"][lang])
        dur = random.choice(VOCAB["durations"][lang])
        sev = random.choice(VOCAB["severities"][lang])
        sym_obj = random.choice(VOCAB["symptoms"])
        risk = random.choice(VOCAB["risk_indicators"])
        body = sym_obj["body"]

        ex = build_annotated_example(tmpl, dur, sev, sym_obj, body, risk, lang)
        if len(ex["entities"]) >= 3:
            dataset.append(ex)

    # Process all examples to include tokens & bio_tags
    processed_dataset = []
    for i, ex in enumerate(dataset):
        tokens, bio_tags = tokenize_with_bio(ex["text"], ex["entities"])
        record = {
            "id": f"PAT_{i+1:04d}",
            "text": ex["text"],
            "language": ex.get("language", "Hinglish"),
            "script": ex.get("script", "Latin"),
            "entities": ex["entities"],
            "tokens": tokens,
            "bio_tags": bio_tags,
            "code_mixing_detected": ex.get("language", "") != "English"
        }
        processed_dataset.append(record)

    return processed_dataset

def split_and_save_dataset():
    data = generate_synthetic_dataset(target_size=450)
    random.shuffle(data)

    n_total = len(data)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    # Save JSON files
    with open(os.path.join(DATA_DIR, "annotated_dataset.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(DATA_DIR, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(DATA_DIR, "val.json"), "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(DATA_DIR, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    # Save CoNLL format files
    def save_conll(split_data, filename):
        with open(os.path.join(DATA_DIR, filename), "w", encoding="utf-8") as f:
            for item in split_data:
                for token, tag in zip(item["tokens"], item["bio_tags"]):
                    f.write(f"{token}\t{tag}\n")
                f.write("\n")

    save_conll(train_data, "train.conll")
    save_conll(val_data, "val.conll")
    save_conll(test_data, "test.conll")

    # Generate Dataset Summary statistics
    entity_counts = {}
    lang_counts = {}
    total_tokens = sum(len(x["tokens"]) for x in data)

    for item in data:
        lang_counts[item["language"]] = lang_counts.get(item["language"], 0) + 1
        for ent in item["entities"]:
            lbl = ent["label"]
            entity_counts[lbl] = entity_counts.get(lbl, 0) + 1

    summary = {
        "total_patient_records": n_total,
        "train_records": len(train_data),
        "val_records": len(val_data),
        "test_records": len(test_data),
        "total_tokens": total_tokens,
        "language_distribution": lang_counts,
        "entity_distribution": entity_counts,
        "target_entity_types": ["SYMPTOM", "RISK_INDICATOR", "SEVERITY", "DURATION", "BODY_PART"]
    }

    with open(os.path.join(DATA_DIR, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Dataset generated successfully!")
    print(f"Total records: {n_total} (Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)})")
    print(f"Entities: {entity_counts}")
    print(f"Languages: {lang_counts}")

if __name__ == "__main__":
    split_and_save_dataset()
