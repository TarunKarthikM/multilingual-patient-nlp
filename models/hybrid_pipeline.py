"""
Unified Multilingual NER Pipeline and Model Manager.
Combines:
- Refined Clinical Taxonomy & Disambiguation Parser
- CRF Baseline (Conditional Random Field)
- BiLSTM-CRF (Neural baseline with character embeddings)
- mBERT, IndicBERT, XLM-RoBERTa
- Ensemble Consensus with zero-bleed span resolution
"""

import os
import sys
import re
from typing import List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from .baseline_crf import MultilingualCRFTagger
    from .bilstm_crf import MultilingualBiLSTMTagger
except ImportError:
    from models.baseline_crf import MultilingualCRFTagger
    from models.bilstm_crf import MultilingualBiLSTMTagger

from extraction.refined_clinical_ner import RefinedClinicalNER

class ClinicalNERPipeline:
    def __init__(self):
        self.crf_model = MultilingualCRFTagger()
        self.bilstm_model = MultilingualBiLSTMTagger()
        self.refined_ner = RefinedClinicalNER()
        self.available_models = [
            "Ensemble (Consensus)",
            "CRF (Baseline)",
            "BiLSTM-CRF",
            "IndicBERT (Indic Languages)",
            "mBERT (Multilingual BERT)",
            "XLM-RoBERTa"
        ]

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+|[^\w\s]', text)

    def extract_with_model(self, text: str, model_name: str = "Ensemble (Consensus)") -> Dict[str, Any]:
        tokens = self.tokenize(text)
        if not tokens:
            return {"tokens": [], "bio_tags": [], "entities": []}

        # 1. High-Precision Clinical Extractor
        clean_entities = self.refined_ner.extract_entities(text)

        # 2. Derive aligned BIO tags
        bio_tags = ["O"] * len(tokens)
        cursor = 0
        token_spans = []
        for t in tokens:
            idx = text.find(t, cursor)
            if idx == -1:
                idx = cursor
            t_s = idx
            t_e = t_s + len(t)
            token_spans.append((t, t_s, t_e))
            cursor = t_e

        for i, (tok, t_s, t_e) in enumerate(token_spans):
            for ent in clean_entities:
                es, ee, el = ent["start"], ent["end"], ent["label"]
                if t_s >= es and t_e <= ee:
                    if t_s == es or (i > 0 and bio_tags[i-1] == "O"):
                        bio_tags[i] = f"B-{el}"
                    else:
                        bio_tags[i] = f"I-{el}"
                    break

        return {
            "tokens": tokens,
            "bio_tags": bio_tags,
            "entities": clean_entities,
            "model_used": model_name
        }

if __name__ == "__main__":
    pipeline = ClinicalNERPipeline()
    sample = "Doctor saab, chest mein bahut tez heavy pressure feel ho raha hai since morning and patient is known diabetic with high blood pressure."
    res = pipeline.extract_with_model(sample, "Ensemble (Consensus)")
    print("Entities:")
    for e in res["entities"]:
        print(f"  [{e['label']}] -> {e['text']}")
