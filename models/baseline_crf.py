"""
CRF (Conditional Random Field) Baseline Model for Multilingual Code-Mixed Clinical NER.
Extracts 5 clinical entities:
- SYMPTOM
- RISK_INDICATOR
- SEVERITY
- DURATION
- BODY_PART
Using pycrfsuite with specialized multilingual & code-mixed features.
"""

import os
import re
import unicodedata
import pycrfsuite
from typing import List, Dict, Any, Tuple

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(MODELS_DIR, "saved_models")
os.makedirs(SAVED_DIR, exist_ok=True)

# Gazetteer cues for Indian multilingual clinical domain
SYMPTOM_CUES = {
    "fever", "headache", "pain", "dard", "cough", "khansi", "vomiting", "ulti",
    "cramp", "swelling", "breathlessness", "infection", "tightness", "weakness",
    "dizziness", "chakkar", "novu", "vali", "cramping", "motion", "loose", "burning"
}

SEVERITY_CUES = {
    "severe", "mild", "acute", "extreme", "unbearable", "bahut", "bohot", "tez",
    "halka", "moderate", "continuous", "persistent", "thumba", "jaasthi", "swalpa",
    "romba", "high", "sudden", "heavy"
}

DURATION_CUES = {
    "days", "day", "din", "dina", "week", "weeks", "hafte", "month", "months", "years", "saal",
    "hours", "hour", "ghante", "since", "from", "for", "se", "inda", "irundhu", "yesterday",
    "kal", "today", "aaj", "morning", "subah", "night", "raat", "beligge", "pichle", "last"
}

RISK_CUES = {
    "diabetic", "diabetes", "bp", "hypertension", "hypertensive", "pressure",
    "smoker", "smoking", "asthma", "heart", "attack", "cardiac", "stroke", "kidney",
    "cancer", "pregnant", "thinners", "unconscious", "collapse", "alcoholic"
}

BODY_CUES = {
    "chest", "head", "headache", "throat", "gala", "stomach", "pet", "hotte", "leg",
    "kaalu", "arm", "back", "kamar", "neck", "nenju", "thala", "eyes", "eye", "kannu",
    "chaati", "sar", "sir"
}

class MultilingualCRFTagger:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.path.join(SAVED_DIR, "crf_model.crfsuite")
        self.tagger = None
        if os.path.exists(self.model_path):
            self.load()

    def _get_script(self, text: str) -> str:
        for char in text:
            name = unicodedata.name(char, "")
            if "DEVANAGARI" in name:
                return "Devanagari"
            elif "KANNADA" in name:
                return "Kannada"
            elif "TAMIL" in name:
                return "Tamil"
        return "Latin"

    def word_to_features(self, tokens: List[str], i: int) -> Dict[str, Any]:
        """Extracts dense linguistic and domain-specific features for token at index i."""
        word = tokens[i]
        w_lower = word.lower()
        script = self._get_script(word)

        features = {
            'bias': 1.0,
            'word.lower()': w_lower,
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'script': script,
            'length': len(word),
            # Morphology affixes
            'prefix-2': w_lower[:2] if len(w_lower) >= 2 else '',
            'prefix-3': w_lower[:3] if len(w_lower) >= 3 else '',
            'suffix-2': w_lower[-2:] if len(w_lower) >= 2 else '',
            'suffix-3': w_lower[-3:] if len(w_lower) >= 3 else '',
            # Gazetteer matches
            'is_symptom_cue': w_lower in SYMPTOM_CUES,
            'is_severity_cue': w_lower in SEVERITY_CUES,
            'is_duration_cue': w_lower in DURATION_CUES,
            'is_risk_cue': w_lower in RISK_CUES,
            'is_body_cue': w_lower in BODY_CUES,
        }

        # Previous word features
        if i > 0:
            w_prev = tokens[i-1]
            wp_lower = w_prev.lower()
            features.update({
                '-1:word.lower()': wp_lower,
                '-1:word.istitle()': w_prev.istitle(),
                '-1:word.isupper()': w_prev.isupper(),
                '-1:word.isdigit()': w_prev.isdigit(),
                '-1:is_severity_cue': wp_lower in SEVERITY_CUES,
                '-1:is_duration_cue': wp_lower in DURATION_CUES,
                '-1:is_risk_cue': wp_lower in RISK_CUES,
                '-1:is_body_cue': wp_lower in BODY_CUES,
            })
        else:
            features['BOS'] = True

        # Next word features
        if i < len(tokens) - 1:
            w_next = tokens[i+1]
            wn_lower = w_next.lower()
            features.update({
                '+1:word.lower()': wn_lower,
                '+1:word.istitle()': w_next.istitle(),
                '+1:word.isupper()': w_next.isupper(),
                '+1:word.isdigit()': w_next.isdigit(),
                '+1:is_symptom_cue': wn_lower in SYMPTOM_CUES,
                '+1:is_duration_cue': wn_lower in DURATION_CUES,
                '+1:is_risk_cue': wn_lower in RISK_CUES,
            })
        else:
            features['EOS'] = True

        return features

    def sent_to_features(self, tokens: List[str]) -> List[Dict[str, Any]]:
        return [self.word_to_features(tokens, i) for i in range(len(tokens))]

    def train(self, train_samples: List[Dict[str, Any]], val_samples: List[Dict[str, Any]] = None):
        """Trains the CRF model on annotated token sequences."""
        trainer = pycrfsuite.Trainer(verbose=False)
        trainer.set_params({
            'c1': 0.1,            # L1 penalty
            'c2': 0.05,           # L2 penalty
            'max_iterations': 80,
            'feature.possible_transitions': True
        })

        for sample in train_samples:
            x_feats = self.sent_to_features(sample["tokens"])
            y_tags = sample["bio_tags"]
            trainer.append(x_feats, y_tags)

        trainer.train(self.model_path)
        self.load()
        print(f"CRF model trained successfully and saved to {self.model_path}")

    def load(self):
        if os.path.exists(self.model_path):
            self.tagger = pycrfsuite.Tagger()
            self.tagger.open(self.model_path)

    def predict(self, tokens: List[str]) -> List[str]:
        """Returns BIO tag sequence for given tokens."""
        if not self.tagger:
            self.load()
        if not self.tagger:
            raise RuntimeError("CRF model has not been trained or loaded.")
        
        feats = self.sent_to_features(tokens)
        return self.tagger.tag(feats)

    def extract_entities(self, text: str, tokens: List[str] = None) -> List[Dict[str, Any]]:
        """Extracts chunked entity spans with offsets and confidence."""
        if tokens is None:
            tokens = re.findall(r'\w+|[^\w\s]', text)
        
        bio_tags = self.predict(tokens)
        entities = []
        current_entity = None

        # Reconstruct token spans in text
        cursor = 0
        for token, tag in zip(tokens, bio_tags):
            # find token location in original text
            t_idx = text.find(token, cursor)
            if t_idx == -1:
                t_idx = cursor
            t_start = t_idx
            t_end = t_start + len(token)
            cursor = t_end

            if tag.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                label = tag[2:]
                current_entity = {
                    "label": label,
                    "text": token,
                    "start": t_start,
                    "end": t_end,
                    "tokens": [token]
                }
            elif tag.startswith("I-") and current_entity and tag[2:] == current_entity["label"]:
                current_entity["text"] += " " + token
                current_entity["end"] = t_end
                current_entity["tokens"].append(token)
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        return entities

if __name__ == "__main__":
    import json
    data_file = os.path.join(MODELS_DIR, "..", "data", "train.json")
    if os.path.exists(data_file):
        with open(data_file, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        model = MultilingualCRFTagger()
        model.train(train_data)
        sample = "Mujhe pichle 3 days se severe chest pain ho raha hai and I am a diabetic patient."
        ents = model.extract_entities(sample)
        print("Sample:", sample)
        print("Extracted Entities:", ents)
