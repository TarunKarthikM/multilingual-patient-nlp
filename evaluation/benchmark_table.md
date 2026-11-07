# Model Comparison: Multilingual Patient Symptom & Risk Indicator Extraction

| Model Architecture | Precision (%) | Recall (%) | F1-Score (%) | Hinglish F1 | Kanglish F1 | English F1 | Best Suited For |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CRF (Baseline)** | 98.29 | 97.30 | **97.79** | 98.56 | 94.0 | 98.61 | Ultra-fast CPU inference & local edge deployment |
| **BiLSTM-CRF** | 94.90 | 94.26 | **94.58** | 98.92 | 96.97 | 81.94 | Strong morphology & sub-character transliteration handling |
| **IndicBERT (ai4bharat/indic-bert)** | 88.42 | 87.15 | **87.78** | 88.64 | 87.14 | 89.14 | Optimal for Dravidian & Indo-Aryan vocabulary alignment |
| **mBERT (bert-base-multilingual-cased)** | 86.80 | 85.30 | **86.04** | 86.54 | 84.69 | 88.94 | Balanced cross-lingual transfer across mixed scripts |
| **XLM-RoBERTa (xlm-roberta-base)** | 89.15 | 88.50 | **88.82** | 89.74 | 88.24 | 90.0 | Highest overall token representation capacity |