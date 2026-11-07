# Multilingual Code-Mixed Clinical NLP Framework
### Patient Symptom & Risk-Indicator Extraction for Early Clinical Risk Assessment
**Department of Computer Science and Engineering (Artificial Intelligence and Machine Learning)**  
**Dayananda Sagar University (DSU) · School of Engineering, Bengaluru**  
**Course:** Natural Language Models (`24AM3503`) · Review-1 / CAT-1 Working Model  
**Supervisor:** Prof. Shivanandan Roy, Assistant Professor, Dept. of CSE (AI&ML)

---

## 👥 Team Details & Contributions
| Sl. No | Name of Student | USN | Contribution Area |
| :---: | :--- | :---: | :--- |
| 1 | **Amal Abraham** | `ENG24AM0138` | Problem Definition & Clinical Schema Design |
| 2 | **Tarun Karthik M** | `ENG24AM0303` | Methodology, Pipeline Engineering & References |
| 3 | **M Shrihari** | `ENG24AM0344` | Objectives & Comparative Benchmark Evaluation |
| 4 | **Sherin Joe** | `ENG24AM0286` | Introduction & Literature Review Analysis |

---

## 🎯 Project Overview & SDG Alignment
Healthcare communication in Indian clinical settings is intrinsically multilingual and code-mixed. Patients routinely articulate symptoms by toggling between English and regional languages (Hinglish, Kanglish, Tanglish) using informal spellings and transliterations (e.g., *"Doctor saab, pet mein bahut tez pain ho raha hai since 2 days, and known diabetic"*).

Conventional clinical NLP models trained exclusively on monolingual, formal English Electronic Health Records (EHR) fail on such noisy, code-mixed sequences. This framework provides an end-to-end, multi-model pipeline to extract:
1. **Symptoms**
2. **Risk Indicators**
3. **Severity Modifiers**
4. **Durations**
5. **Anatomical Body Parts**

And executes **Early Clinical Triage Risk Stratification**, directly contributing to:
- **UN SDG 3:** Good Health & Well-being (Early risk detection and triaging)
- **UN SDG 9:** Industry, Innovation & Infrastructure (Reusable low-resource NLP toolchain)
- **UN SDG 10:** Reduced Inequalities (Inclusion of Romanized & regional language speakers)

---

## 🏗️ System Architecture
```
Unstructured Multilingual Patient Query (Hinglish, Kanglish, Tanglish, English)
                          │
                          ▼
            [ Text Normalization & Cleaning ]
        (Elongation reduction, punctuation spacing, abbrev)
                          │
                          ▼
         [ Language & Code-Mixing Detection ]
     (Script analysis, Token Lang-ID, Code-Mixing Index CMI)
                          │
                          ▼
        [ Multilingual Sequence Tagging Models ]
   ┌───────────────┬─────────────────┬──────────────────────┐
   ▼               ▼                 ▼                      ▼
  CRF         BiLSTM-CRF          IndicBERT              mBERT / XLM-R
Baseline   (Char+Word Embs)    (ai4bharat)        (Multilingual Transformers)
   └───────────────┴─────────────────┴──────────────────────┘
                          │
                          ▼
            [ Structured Entity Chunking ]
   (SYMPTOM, RISK_INDICATOR, SEVERITY, DURATION, BODY_PART)
                          │
                          ▼
       [ Transliteration & Clinical Normalization ]
     (Maps 'sir dard' -> 'headache', 'hotte novu' -> 'abdominal pain')
                          │
                          ▼
        [ Clinical Decision Support Triage Engine ]
     (Calculates Risk Score 0-100 & Triage Tier: CRITICAL/HIGH/MOD/LOW)
                          │
                          ▼
       Structured Clinical Record & Interactive UI Dashboard
```

---

## 🏷️ Annotation Schema (5 Target Entity Categories)
| Entity Label | BIO Tag | Example Expressions | Clinical Purpose |
| :--- | :--- | :--- | :--- |
| `SYMPTOM` | `B-SYMPTOM`, `I-SYMPTOM` | *headache, fever, sir dard, ulti, novu, breathlessness* | Chief clinical complaints |
| `RISK_INDICATOR` | `B-RISK_INDICATOR`, `I-RISK_INDICATOR` | *diabetic patient, high blood pressure, smoker, heart attack* | Comorbidities & acute red flags |
| `SEVERITY` | `B-SEVERITY`, `I-SEVERITY` | *severe, acute, bahut tez, mild, unbearable, thumba jaasthi* | Symptom intensity quantification |
| `DURATION` | `B-DURATION`, `I-DURATION` | *pichle 3 days se, since yesterday, last 2 days inda* | Progression & chronicity tracking |
| `BODY_PART` | `B-BODY_PART`, `I-BODY_PART` | *chest, pet, thale, gala, left arm, hotte, nenju* | Anatomical localization |

---

## 📊 Comparative Performance Benchmark (Review-1 Evaluation)
Evaluated across test split under identical stratified code-mixed distributions:

| Model Architecture | Precision (%) | Recall (%) | Overall F1 (%) | Hinglish F1 | Kanglish F1 | English F1 | Primary Suitability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CRF (Baseline)** | 98.29 | 97.30 | **97.79** | 98.56 | 94.00 | 98.61 | Ultra-fast CPU inference (< 5ms) for edge clinics |
| **BiLSTM-CRF** | 94.90 | 94.26 | **94.58** | 98.92 | 96.97 | 81.94 | Sub-character transliteration & noisy typos |
| **IndicBERT (ai4bharat)** | 88.42 | 87.15 | **87.78** | 88.64 | 87.14 | 89.14 | Dravidian & Indo-Aryan vocabulary alignment |
| **mBERT (Multilingual)** | 86.80 | 85.30 | **86.04** | 86.54 | 84.69 | 88.94 | Cross-script multi-language transfer |
| **XLM-RoBERTa** | 89.15 | 88.50 | **88.82** | 89.74 | 88.24 | 90.00 | Deep contextual representation capacity |

---

## 🚀 Quickstart & How to Run

### 1. Prerequisites & Virtual Environment
Ensure Python 3.10+ (or Python 3.14) is available. All required packages are listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Launch the Interactive Web Dashboard
Run the Flask server:
```bash
python web_app/app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5050
```

### 3. Run the CLI Review Demonstration
Execute the automated multi-case evaluation demonstration:
```bash
python scripts/run_demo.py
```

### 4. Retrain Models and Regenerate Benchmarks
Run the complete pipeline from scratch:
```bash
python scripts/train_and_eval.py
```

### 5. Google Colab GPU Notebook
Open `notebooks/multilingual_nlp_colab.ipynb` in [Google Colab](https://colab.research.google.com/) to inspect GPU fine-tuning of IndicBERT and XLM-RoBERTa.

---

## 📂 Project Directory Structure
```
multilingual-patient-nlp/
├── data/
│   ├── data_generator.py          # Synthetic dataset generator & BIO tokenizer
│   ├── annotated_dataset.json     # Complete annotated corpus (450 records)
│   ├── train.json, val.json, test.json
│   ├── train.conll, val.conll, test.conll  # CoNLL BIO sequence files
│   └── dataset_summary.json       # Corpus statistics
├── preprocessing/
│   ├── normalizer.py              # Elongated words & punctuation normalization
│   ├── language_detector.py       # Script detection, Lang-ID & CMI calculator
│   └── transliteration.py         # Standardized clinical term resolver
├── models/
│   ├── baseline_crf.py            # Feature-engineered CRF tagger
│   ├── bilstm_crf.py              # PyTorch BiLSTM with Char+Word embeddings
│   ├── transformer_ner.py         # IndicBERT / mBERT / XLM-R HuggingFace module
│   ├── hybrid_pipeline.py         # Unified inference & model switcher
│   └── saved_models/              # Trained weights & metadata
├── extraction/
│   ├── extractor.py               # End-to-end information extractor
│   └── risk_engine.py             # Early triage risk stratification (0-100 score)
├── evaluation/
│   ├── evaluator.py               # Seqeval & Scikit-learn evaluation metrics
│   ├── benchmark_report.py        # Comparative benchmark table generator
│   └── benchmark_results.json     # Serialized benchmark metrics
├── web_app/
│   ├── app.py                     # Flask web server & REST API
│   ├── templates/index.html       # DSU branded responsive clinical UI
│   └── static/                    # CSS stylesheets & client JavaScript
├── notebooks/
│   └── multilingual_nlp_colab.ipynb  # GPU ready Colab notebook
├── scripts/
│   ├── run_demo.py                # Command-line demonstration
│   └── train_and_eval.py          # Unified training & evaluation runner
├── requirements.txt               # Python package dependencies
└── README.md                      # Comprehensive project documentation
```

---

## 📚 References
1. Conneau, A., et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale (XLM-R)*. ACL.
2. Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.
3. Kakwani, D., et al. (2020). *IndicNLPSuite: Pre-trained Multilingual Language Models for Indian Languages*. EMNLP.
4. Raithel, L., et al. (2025). *Cross- & Multi-lingual Medication Detection: A Transformer-Based Analysis*. BMC Med. Inform.
5. Applied Sciences (2025). *Weakly-Supervised Multilingual Medical NER for Symptom Extraction for Low-Resource Languages*. MDPI.
6. Keloth, V., et al. (2025). *Social Determinants of Health Extraction from Clinical Notes Across Institutions Using LLMs*. npj Digital Medicine.
7. Dey, S., et al. (2026). *Understanding User Intent in Code-Mixed Sexual and Reproductive Health Queries in Urban India*. JMIR.
8. Aurpa, T. T., et al. (2026). *Bangla MedER: Multi-BERT Ensemble Approach for Recognition of Bangla Medical Entity*. PLOS ONE.
