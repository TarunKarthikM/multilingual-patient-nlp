"""
Language and Code-Mixing Detector for Multilingual Patient Text.
Calculates:
- Script distribution (Latin, Devanagari, Kannada, Tamil, etc.)
- Token-level language identification
- Code-Mixing Index (CMI)
- Matrix language and embedded language detection
"""

import re
import unicodedata
from typing import Dict, List, Any

# Marker vocabulary for Romanized Indian languages vs English
HINDI_ROMAN_MARKERS = {
    "hai", "ho", "raha", "rahi", "halke", "bahut", "bohot", "bohot", "dard", "sir", "sar",
    "pet", "gala", "kharab", "ulti", "chakkar", "se", "mein", "me", "ko", "ki", "ka",
    "ke", "aur", "bhi", "kal", "subah", "raat", "hafte", "saab", "meri", "mera", "mujhe",
    "unko", "khansi", "saas", "dikkat", "pehle", "pichle", "din"
}

KANNADA_ROMAN_MARKERS = {
    "ide", "aagide", "aagthide", "barthide", "novu", "novoo", "hotte", "thale", "kaalu",
    "nanage", "nanna", "ge", "inda", "mathu", "mattu", "kuda", "jaasthi", "swalpa", "thumba",
    "ninne", "beligge", "eradu", "dina", "varada", "appange", "amma", "irbeku"
}

TAMIL_ROMAN_MARKERS = {
    "irukku", "iruku", "aagitaaru", "aagidhu", "romba", "enaku", "enakku", "appa", "ku",
    "valikuthu", "vali", "nethu", "kalaila", "irundhu", "adhigama", "chinna", "varuthu",
    "odambu", "thala", "nenju"
}

ENGLISH_COMMON = {
    "the", "is", "and", "in", "to", "of", "a", "with", "for", "patient", "severe", "fever",
    "pain", "chest", "cough", "since", "days", "headache", "vomiting", "weak", "history"
}

class LanguageDetector:
    def __init__(self):
        pass

    def detect_scripts(self, text: str) -> Dict[str, float]:
        """Detects the percentage breakdown of Unicode scripts in the text."""
        script_counts = {}
        total_chars = 0

        for char in text:
            if not char.isalnum():
                continue
            total_chars += 1
            name = unicodedata.name(char, "")
            if "DEVANAGARI" in name:
                script = "Devanagari"
            elif "KANNADA" in name:
                script = "Kannada"
            elif "TAMIL" in name:
                script = "Tamil"
            elif "LATIN" in name:
                script = "Latin"
            else:
                script = "Other"
            
            script_counts[script] = script_counts.get(script, 0) + 1

        if total_chars == 0:
            return {"Latin": 1.0}

        return {k: round(v / total_chars, 3) for k, v in script_counts.items()}

    def identify_token_language(self, token: str) -> str:
        """Heuristically tags token as EN, HI, KN, TA, or OTHER."""
        t_lower = token.lower()
        if re.search(r'[\u0900-\u097F]', token):
            return "HI_NATIVE"
        if re.search(r'[\u0C80-\u0CFF]', token):
            return "KN_NATIVE"
        if re.search(r'[\u0B80-\u0BFF]', token):
            return "TA_NATIVE"

        if t_lower in HINDI_ROMAN_MARKERS:
            return "HI_ROMAN"
        if t_lower in KANNADA_ROMAN_MARKERS:
            return "KN_ROMAN"
        if t_lower in TAMIL_ROMAN_MARKERS:
            return "TA_ROMAN"
        if t_lower in ENGLISH_COMMON:
            return "EN"
        
        # Punctuation / Numbers
        if not token.isalnum():
            return "PUNCT"
        if token.isdigit():
            return "NUM"

        return "UNKNOWN"

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Calculates Code-Mixing Index (CMI) and linguistic metadata:
        CMI = (N - max(w_i) + e) / N  where w_i are language word counts, e is language switches.
        """
        scripts = self.detect_scripts(text)
        tokens = re.findall(r'\w+|[^\w\s]', text)

        lang_tokens = []
        token_tags = []

        for t in tokens:
            tag = self.identify_token_language(t)
            token_tags.append({"token": t, "lang_tag": tag})
            if tag in ["HI_ROMAN", "KN_ROMAN", "TA_ROMAN", "HI_NATIVE", "KN_NATIVE", "TA_NATIVE", "EN"]:
                lang_tokens.append(tag.split("_")[0])

        n_lang = len(lang_tokens)
        if n_lang == 0:
            return {
                "scripts": scripts,
                "primary_language": "English",
                "code_mixed": False,
                "code_mixing_index": 0.0,
                "language_distribution": {},
                "token_tags": token_tags
            }

        # Count frequencies
        freqs = {}
        for l in lang_tokens:
            freqs[l] = freqs.get(l, 0) + 1

        # Language switches
        switches = 0
        for i in range(1, len(lang_tokens)):
            if lang_tokens[i] != lang_tokens[i - 1]:
                switches += 1

        max_lang_count = max(freqs.values()) if freqs else n_lang
        
        # Code Mixing Index: Gamback & Das formulation
        if n_lang > 0 and len(freqs) > 1:
            cmi = (n_lang - max_lang_count + switches) / (2 * n_lang)
            cmi = min(round(cmi, 3), 1.0)
        else:
            cmi = 0.0

        # Determine dominant language
        sorted_langs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
        primary_lang = sorted_langs[0][0]

        lang_label_map = {
            "EN": "English",
            "HI": "Hinglish" if "Latin" in scripts else "Hindi",
            "KN": "Kanglish" if "Latin" in scripts else "Kannada",
            "TA": "Tanglish" if "Latin" in scripts else "Tamil"
        }

        # If mixed, format label
        if len(freqs) > 1:
            secondary_lang = sorted_langs[1][0]
            display_lang = f"{lang_label_map.get(primary_lang, primary_lang)}-{lang_label_map.get(secondary_lang, secondary_lang)}"
        else:
            display_lang = lang_label_map.get(primary_lang, primary_lang)

        return {
            "scripts": scripts,
            "primary_language": display_lang,
            "code_mixed": cmi > 0.05 or len(freqs) > 1,
            "code_mixing_index": cmi,
            "language_distribution": {k: round(v / n_lang, 3) for k, v in freqs.items()},
            "token_tags": token_tags
        }

if __name__ == "__main__":
    detector = LanguageDetector()
    samples = [
        "Mujhe pichle 3 days se severe headache and gala pain ho raha hai",
        "Nanage last 2 days inda severe fever ide and breathing problem barthide",
        "Patient has high blood pressure since 5 years and severe chest pain",
        "मरीज को 3 दिन से तेज बुखार है और छाती में pain hai"
    ]
    for s in samples:
        res = detector.analyze_text(s)
        print("\nInput:", s)
        print("Lang:", res["primary_language"], "| Mixed:", res["code_mixed"], "| CMI:", res["code_mixing_index"])
