"""
Transliteration and Colloquial Terminology Mapper for Indian Patient Text.
Maps informal Romanized Indian words (Hindi/Kannada/Tamil) to standardized clinical concepts.
"""

# Lexicon of Romanized regional terms mapped to standard English clinical categories
TRANSLITERATION_LEXICON = {
    # Hindi / Hinglish terms
    "sir dard": {"standard": "headache", "category": "SYMPTOM", "body_part": "head"},
    "sar dard": {"standard": "headache", "category": "SYMPTOM", "body_part": "head"},
    "matha dard": {"standard": "headache", "category": "SYMPTOM", "body_part": "head"},
    "khansi": {"standard": "cough", "category": "SYMPTOM", "body_part": "throat"},
    "gala kharab": {"standard": "sore throat", "category": "SYMPTOM", "body_part": "throat"},
    "pet dard": {"standard": "abdominal pain", "category": "SYMPTOM", "body_part": "abdomen"},
    "pet kharab": {"standard": "stomach upset / diarrhea", "category": "SYMPTOM", "body_part": "abdomen"},
    "ulti": {"standard": "vomiting", "category": "SYMPTOM", "body_part": "abdomen"},
    "saas lene me dikkat": {"standard": "dyspnea / breathlessness", "category": "SYMPTOM", "body_part": "chest"},
    "saans fulna": {"standard": "shortness of breath", "category": "SYMPTOM", "body_part": "chest"},
    "chakkar": {"standard": "vertigo / dizziness", "category": "SYMPTOM", "body_part": "head"},
    "kampa": {"standard": "tremor / shivering", "category": "SYMPTOM", "body_part": "general"},
    "thand lagna": {"standard": "chills / rigors", "category": "SYMPTOM", "body_part": "general"},
    "kamar dard": {"standard": "lumbar pain / backache", "category": "SYMPTOM", "body_part": "back"},

    # Kannada / Kanglish terms
    "thale novu": {"standard": "headache", "category": "SYMPTOM", "body_part": "head"},
    "tale novu": {"standard": "headache", "category": "SYMPTOM", "body_part": "head"},
    "hotte novu": {"standard": "abdominal pain", "category": "SYMPTOM", "body_part": "abdomen"},
    "kaalu novu": {"standard": "leg pain", "category": "SYMPTOM", "body_part": "legs"},
    "kannu uri": {"standard": "eye burning sensation", "category": "SYMPTOM", "body_part": "eyes"},
    "kasa": {"standard": "cough", "category": "SYMPTOM", "body_part": "throat"},
    "chali": {"standard": "chills", "category": "SYMPTOM", "body_part": "general"},
    "usiraata thondare": {"standard": "breathing difficulty", "category": "SYMPTOM", "body_part": "chest"},

    # Tamil / Tanglish terms
    "thala vali": {"standard": "headache", "category": "SYMPTOM", "body_part": "head"},
    "nenju vali": {"standard": "chest pain", "category": "SYMPTOM", "body_part": "chest"},
    "vayithu vali": {"standard": "stomach ache", "category": "SYMPTOM", "body_part": "abdomen"},
    "irumal": {"standard": "cough", "category": "SYMPTOM", "body_part": "throat"},
    "moochu thinaral": {"standard": "dyspnea / breathlessness", "category": "SYMPTOM", "body_part": "chest"},
    "mayakkam": {"standard": "dizziness / syncope", "category": "SYMPTOM", "body_part": "head"},

    # Body parts
    "gala": {"standard": "throat", "category": "BODY_PART"},
    "pet": {"standard": "abdomen", "category": "BODY_PART"},
    "sir": {"standard": "head", "category": "BODY_PART"},
    "sar": {"standard": "head", "category": "BODY_PART"},
    "chaati": {"standard": "chest", "category": "BODY_PART"},
    "kamar": {"standard": "lower back", "category": "BODY_PART"},
    "thale": {"standard": "head", "category": "BODY_PART"},
    "hotte": {"standard": "abdomen", "category": "BODY_PART"},
    "kaalu": {"standard": "leg", "category": "BODY_PART"},
    "nenju": {"standard": "chest", "category": "BODY_PART"},
    "thala": {"standard": "head", "category": "BODY_PART"},

    # Severity terms
    "bahut tez": {"standard": "severe / acute", "category": "SEVERITY"},
    "bohot zyada": {"standard": "severe / intense", "category": "SEVERITY"},
    "halka": {"standard": "mild", "category": "SEVERITY"},
    "thumba jaasthi": {"standard": "very high / severe", "category": "SEVERITY"},
    "swalpa": {"standard": "mild", "category": "SEVERITY"},
    "romba": {"standard": "severe / extreme", "category": "SEVERITY"},
    "chinna": {"standard": "mild", "category": "SEVERITY"}
}

class ClinicalTermStandardizer:
    def __init__(self):
        self.lexicon = TRANSLITERATION_LEXICON

    def resolve_term(self, phrase: str):
        """Resolves an informal or transliterated term to clinical standard."""
        p_clean = phrase.lower().strip()
        if p_clean in self.lexicon:
            return self.lexicon[p_clean]
        
        # Check sub-words
        for k, v in self.lexicon.items():
            if k in p_clean:
                return v

        return {"standard": phrase, "category": "UNKNOWN"}

if __name__ == "__main__":
    standardizer = ClinicalTermStandardizer()
    for term in ["sir dard", "hotte novu", "nenju vali", "bahut tez"]:
        print(f"{term} -> {standardizer.resolve_term(term)}")
