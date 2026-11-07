"""
End-to-End Clinical Information Extraction Pipeline.
Integrates:
1. Normalization
2. Language and Code-Mixing Detection (CMI)
3. Multilingual NER (mBERT, XLM-R, IndicBERT, BiLSTM-CRF, CRF)
4. Transliteration and Colloquial Term Standardization
5. Clinical Risk Assessment Engine
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Dict, Any, List
from preprocessing.normalizer import TextNormalizer
from preprocessing.language_detector import LanguageDetector
from preprocessing.transliteration import ClinicalTermStandardizer
from models.hybrid_pipeline import ClinicalNERPipeline
from extraction.risk_engine import ClinicalRiskEngine

class PatientInformationExtractor:
    def __init__(self):
        self.normalizer = TextNormalizer()
        self.lang_detector = LanguageDetector()
        self.term_standardizer = ClinicalTermStandardizer()
        self.ner_pipeline = ClinicalNERPipeline()
        self.risk_engine = ClinicalRiskEngine()

    def process(self, raw_text: str, model_name: str = "Ensemble (Consensus)") -> Dict[str, Any]:
        """
        Executes end-to-end information extraction on unstructured patient text.
        """
        # 1. Normalization
        cleaned_text = self.normalizer.clean(raw_text)

        # 2. Language and Code-Mixing Analysis
        lang_meta = self.lang_detector.analyze_text(cleaned_text)

        # 3. Entity Extraction via Selected Model
        ner_output = self.ner_pipeline.extract_with_model(cleaned_text, model_name=model_name)
        extracted_entities = ner_output["entities"]

        # 4. Standardize and enrich entities with clinical mappings
        enriched_entities = []
        for ent in extracted_entities:
            resolved = self.term_standardizer.resolve_term(ent["text"])
            ent_copy = dict(ent)
            ent_copy["standardized_concept"] = resolved.get("standard", ent["text"])
            enriched_entities.append(ent_copy)

        # 5. Group by entity category
        grouped = {
            "symptoms": [e for e in enriched_entities if e["label"] == "SYMPTOM"],
            "risk_indicators": [e for e in enriched_entities if e["label"] == "RISK_INDICATOR"],
            "severity": [e for e in enriched_entities if e["label"] == "SEVERITY"],
            "duration": [e for e in enriched_entities if e["label"] == "DURATION"],
            "body_parts": [e for e in enriched_entities if e["label"] == "BODY_PART"],
        }

        # 6. Evaluate Clinical Risk
        risk_profile = self.risk_engine.evaluate_risk(enriched_entities)

        return {
            "input_text": raw_text,
            "normalized_text": cleaned_text,
            "linguistic_analysis": {
                "detected_language": lang_meta["primary_language"],
                "code_mixing_index": lang_meta["code_mixing_index"],
                "is_code_mixed": lang_meta["code_mixed"],
                "scripts": lang_meta["scripts"],
                "language_distribution": lang_meta["language_distribution"]
            },
            "model_used": model_name,
            "entities": enriched_entities,
            "structured_findings": grouped,
            "clinical_risk_assessment": risk_profile
        }

if __name__ == "__main__":
    extractor = PatientInformationExtractor()
    test_query = "Mujhe pichle 3 days se severe chest pain ho raha hai and I am a diabetic patient."
    result = extractor.process(test_query)
    print("Detected Language:", result["linguistic_analysis"]["detected_language"])
    print("Risk Level:", result["clinical_risk_assessment"]["risk_level"])
    print("Entities Found:", len(result["entities"]))
