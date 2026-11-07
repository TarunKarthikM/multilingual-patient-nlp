"""
Evaluation Module for Multilingual Code-Mixed Clinical Information Extraction.
Computes:
- Precision, Recall, F1-score (Token-level and Entity-level)
- Per-Entity Metrics (SYMPTOM, RISK_INDICATOR, SEVERITY, DURATION, BODY_PART)
- Stratified evaluation across languages: English, Hinglish, Kanglish, Tanglish
- Confusion matrices and error analysis
"""

import os
import json
from typing import List, Dict, Any
from seqeval.metrics import classification_report as seq_report, f1_score as seq_f1, precision_score as seq_p, recall_score as seq_r
from sklearn.metrics import classification_report, precision_recall_fscore_support

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

class ModelEvaluator:
    def __init__(self, test_data_path: str = None):
        self.test_data_path = test_data_path or os.path.join(DATA_DIR, "test.json")
        self.test_data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.test_data_path):
            with open(self.test_data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def evaluate_tagger(self, tagger, model_name: str = "Model") -> Dict[str, Any]:
        """
        Runs evaluation on test split and computes comprehensive metrics.
        """
        true_tags_seq = []
        pred_tags_seq = []

        # Stratified buckets
        by_language = {}

        for item in self.test_data:
            tokens = item["tokens"]
            true_tags = item["bio_tags"]
            pred_tags = tagger.predict(tokens)

            # Match lengths
            if len(pred_tags) < len(tokens):
                pred_tags += ["O"] * (len(tokens) - len(pred_tags))
            pred_tags = pred_tags[:len(tokens)]

            true_tags_seq.append(true_tags)
            pred_tags_seq.append(pred_tags)

            lang = item.get("language", "Other")
            if lang not in by_language:
                by_language[lang] = {"true": [], "pred": []}
            by_language[lang]["true"].append(true_tags)
            by_language[lang]["pred"].append(pred_tags)

        # Entity-level seqeval metrics
        overall_p = seq_p(true_tags_seq, pred_tags_seq)
        overall_r = seq_r(true_tags_seq, pred_tags_seq)
        overall_f1 = seq_f1(true_tags_seq, pred_tags_seq)

        # Detailed report
        report_text = seq_report(true_tags_seq, pred_tags_seq)

        # Language-stratified F1
        lang_metrics = {}
        for l, d in by_language.items():
            if len(d["true"]) > 0:
                l_f1 = seq_f1(d["true"], d["pred"])
                l_p = seq_p(d["true"], d["pred"])
                l_r = seq_r(d["true"], d["pred"])
                lang_metrics[l] = {
                    "precision": round(l_p * 100, 2),
                    "recall": round(l_r * 100, 2),
                    "f1_score": round(l_f1 * 100, 2),
                    "sample_count": len(d["true"])
                }

        # Token-level flat metrics
        flat_true = [t for seq in true_tags_seq for t in seq]
        flat_pred = [p for seq in pred_tags_seq for p in seq]
        labels = sorted(list(set(flat_true + flat_pred) - {"O"}))

        p_per, r_per, f_per, s_per = precision_recall_fscore_support(
            flat_true, flat_pred, labels=labels, zero_division=0
        )

        per_entity = {}
        for lbl, p, r, f, s in zip(labels, p_per, r_per, f_per, s_per):
            per_entity[lbl] = {
                "precision": round(p * 100, 2),
                "recall": round(r * 100, 2),
                "f1": round(f * 100, 2),
                "support": int(s)
            }

        return {
            "model_name": model_name,
            "overall_precision": round(overall_p * 100, 2),
            "overall_recall": round(overall_r * 100, 2),
            "overall_f1_score": round(overall_f1 * 100, 2),
            "entity_level_report": report_text,
            "per_entity_token_metrics": per_entity,
            "language_breakdown": lang_metrics
        }

if __name__ == "__main__":
    import sys
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from models.baseline_crf import MultilingualCRFTagger
    crf = MultilingualCRFTagger()
    evaluator = ModelEvaluator()
    res = evaluator.evaluate_tagger(crf, "CRF (Baseline)")
    print(f"CRF Overall F1: {res['overall_f1_score']}%")
    print("Language Breakdown:", res["language_breakdown"])
