"""
Streamlit Web Application for Multilingual Patient NLP Framework
Dayananda Sagar University (DSU) · Department of CSE (AI&ML)
Comprehensive Academic Presentation UI (Restored Full Layout with Accurate Predictions)
"""

import os
import sys
import json
import html
import streamlit as st
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from extraction.extractor import PatientInformationExtractor

# Page configuration
st.set_page_config(
    page_title="Multilingual Clinical NLP | DSU AI&ML",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# High-contrast, clean typography and spacing
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #0f172a;
    }
    
    /* Top Header Banner */
    .header-box {
        background: #0f172a;
        color: #ffffff;
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border-left: 8px solid #0284c7;
    }
    .header-box h1 {
        font-size: 1.75rem;
        font-weight: 800;
        color: #ffffff !important;
        margin: 0;
        padding-bottom: 0.3rem;
    }
    .header-box p {
        font-size: 1.05rem;
        color: #94a3b8 !important;
        margin: 0;
    }
    .header-box .meta-pill {
        display: inline-block;
        background: #0284c7;
        color: #ffffff;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        margin-top: 0.6rem;
    }

    /* Distinct, Large High-Contrast Entity Pills */
    .pill-symptom {
        background-color: #dbeafe;
        color: #1e3a8a;
        border: 1.5px solid #3b82f6;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        margin: 2px 4px;
    }
    .pill-risk {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1.5px solid #ef4444;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        margin: 2px 4px;
    }
    .pill-severity {
        background-color: #ffedd5;
        color: #9a3412;
        border: 1.5px solid #f97316;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        margin: 2px 4px;
    }
    .pill-duration {
        background-color: #f3e8ff;
        color: #581c87;
        border: 1.5px solid #a855f7;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        margin: 2px 4px;
    }
    .pill-body {
        background-color: #ccfbf1;
        color: #134e4a;
        border: 1.5px solid #14b8a6;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        margin: 2px 4px;
    }

    /* Highlighted annotated text container */
    .text-display-box {
        background: #ffffff;
        border: 2px solid #cbd5e1;
        border-radius: 8px;
        padding: 1.5rem;
        font-size: 1.25rem;
        line-height: 2.3;
        color: #0f172a;
        margin-top: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Triage Alert Banners */
    .triage-banner {
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        border-left: 8px solid;
    }
    .triage-CRITICAL {
        background-color: #fef2f2;
        border-left-color: #dc2626;
        color: #991b1b;
    }
    .triage-HIGH {
        background-color: #fff7ed;
        border-left-color: #ea580c;
        color: #9a3412;
    }
    .triage-MODERATE {
        background-color: #eff6ff;
        border-left-color: #2563eb;
        color: #1e40af;
    }
    .triage-LOW {
        background-color: #f0fdf4;
        border-left-color: #16a34a;
        color: #166534;
    }

    /* Clean Findings Cards */
    .finding-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }

    .stButton>button {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 6px !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_extractor():
    return PatientInformationExtractor()

extractor = get_extractor()

# Sample Presets with clear descriptions
PRESET_OPTIONS = {
    "Select a pre-built example...": "",
    "🚨 Hinglish: Cardiac Emergency & Diabetic Comorbidity": "Doctor saab, chest mein bahut tez heavy pressure feel ho raha hai since morning and patient is known diabetic with high blood pressure.",
    "🤒 Kanglish: Severe Fever & Throat Infection": "Nanage last 2 days inda severe fever ide and throat full infection aagide, swallowing madoke thumba novu barthide.",
    "💔 Tanglish: Chronic Heart Patient on Blood Thinners": "Enaku past 4 days ah severe chest pain irukku, and I am cardiac patient on blood thinners since 2 years.",
    "🤢 Hinglish: Acute Stomach Cramping & Continuous Vomiting": "Sir mere pet mein unbearable cramping ho rahi hai since 2 days, and bohot zyada continuous ulti aur weakness hai.",
    "🫁 English: Chronic Smoker with Acute Breathlessness": "Patient is chronic smoker for 15 years, complaining of acute breathlessness and chest tightness for the last 6 hours."
}

# ----------------- TOP HEADER -----------------
st.markdown("""
<div class="header-box">
    <h1>Multilingual Code-Mixed Clinical NLP Framework</h1>
    <p>Patient Symptom & Risk-Indicator Extraction for Early Triage Assessment (UN SDG 3)</p>
    <div class="meta-pill">Dayananda Sagar University (DSU) · CSE (AI&ML) · Natural Language Models (24AM3503)</div>
</div>
""", unsafe_allow_html=True)

# ----------------- TABS AT THE TOP -----------------
tab_live, tab_models, tab_dataset, tab_team = st.tabs([
    "🩺 Live Clinical Extraction & Triage",
    "📊 Model Benchmark Comparison",
    "📁 Dataset & Annotation Schema",
    "👥 Academic Team & Project Info"
])

# =========================================================================
# TAB 1: LIVE EXTRACTION & TRIAGE
# =========================================================================
with tab_live:
    col_preset, col_model = st.columns([2.5, 1.5])
    with col_preset:
        preset_key = st.selectbox(
            "📋 Quick Sample Inputs (click to load):",
            list(PRESET_OPTIONS.keys()),
            index=1
        )
    with col_model:
        selected_model = st.selectbox(
            "⚙️ Active NLP Model:",
            [
                "Ensemble (Consensus - Best Accuracy)",
                "CRF (Baseline - Ultra-Fast CPU)",
                "BiLSTM-CRF (Sub-word Embeddings)",
                "IndicBERT (Indic Languages)",
                "mBERT (Multilingual BERT)",
                "XLM-RoBERTa"
            ],
            index=0
        )

    clean_model_name = selected_model.split(" (")[0]
    if "Ensemble" in selected_model:
        clean_model_name = "Ensemble (Consensus)"

    default_text = PRESET_OPTIONS[preset_key]
    user_input = st.text_area(
        "📝 Patient Symptom Description (Hinglish, Kanglish, Tanglish, or English):",
        value=default_text,
        height=100,
        placeholder="Enter patient statement here..."
    )

    btn_extract = st.button("🔍 Extract Symptoms & Assess Risk Level", type="primary", use_container_width=True)

    # Process extraction
    if btn_extract or user_input:
        if not user_input.strip():
            st.warning("Please type a patient symptom description or select an example above.")
        else:
            res = extractor.process(user_input, model_name=clean_model_name)
            ling = res["linguistic_analysis"]
            risk = res["clinical_risk_assessment"]
            findings = res["structured_findings"]

            st.write("")
            
            # 1. TRIAGE RISK BANNER
            t_level = risk["risk_level"]
            icons = {"CRITICAL": "🚨", "HIGH": "⚠️", "MODERATE": "ℹ️", "LOW": "✅"}
            icon = icons.get(t_level, "🩺")

            st.markdown(f"""
            <div class="triage-banner triage-{t_level}">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <div style="font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Early Clinical Triage Level</div>
                        <div style="font-size: 1.6rem; font-weight: 800; margin-top: 0.2rem;">{icon} {t_level} RISK (Score: {risk['risk_score']} / 100)</div>
                    </div>
                    <div style="font-size: 1.05rem; font-weight: 700; background: rgba(255,255,255,0.7); padding: 0.5rem 1rem; border-radius: 6px;">
                        👉 Action: {risk['recommended_urgency']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 2. LINGUISTIC STATS ROW
            stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
            stat_c1.metric("Detected Language", ling["detected_language"])
            stat_c2.metric("Code-Mixing Index (CMI)", f"{ling['code_mixing_index']*100:.1f}%")
            stat_c3.metric("Clinical Entities Found", len(res["entities"]))
            stat_c4.metric("Model Used", clean_model_name)

            # 3. HIGHLIGHTED TEXT SECTION
            st.markdown("### 🏷️ Annotated Patient Statement")
            st.markdown("""
            **Visual Legend:** 
            <span class="pill-symptom">SYMPTOM</span>
            <span class="pill-risk">RISK_INDICATOR</span>
            <span class="pill-severity">SEVERITY</span>
            <span class="pill-duration">DURATION</span>
            <span class="pill-body">BODY_PART</span>
            """, unsafe_allow_html=True)

            # Build highlighted HTML
            def build_highlighted_box(text, entities):
                sorted_ents = sorted(entities, key=lambda x: x["start"])
                out = ""
                cursor = 0
                css_classes = {
                    "SYMPTOM": "pill-symptom",
                    "RISK_INDICATOR": "pill-risk",
                    "SEVERITY": "pill-severity",
                    "DURATION": "pill-duration",
                    "BODY_PART": "pill-body"
                }

                for e in sorted_ents:
                    if e["start"] > cursor:
                        out += html.escape(text[cursor:e["start"]])
                    c_name = css_classes.get(e["label"], "pill-symptom")
                    out += f'<span class="{c_name}">{html.escape(e["text"])} <span style="font-size:0.75rem; opacity:0.85;">[{e["label"]}]</span></span>'
                    cursor = max(cursor, e["end"])

                if cursor < len(text):
                    out += html.escape(text[cursor:])
                return out

            highlighted_html = build_highlighted_box(res["normalized_text"], res["entities"])
            st.markdown(f'<div class="text-display-box">{highlighted_html}</div>', unsafe_allow_html=True)

            st.write("")
            # 4. STRUCTURED FINDINGS COLUMNS (5 categories)
            st.markdown("### 📋 Extracted Clinical Findings")
            
            c_sym, c_risk, c_sev, c_dur, c_body = st.columns(5)

            with c_sym:
                st.markdown("#### 🩺 Symptoms")
                items = findings.get("symptoms", [])
                if items:
                    for it in items:
                        std = it.get("standardized_concept")
                        if std and std.lower() != it["text"].lower():
                            st.info(f"**{it['text']}**\n\n*(Concept: {std})*")
                        else:
                            st.info(f"**{it['text']}**")
                else:
                    st.caption("None detected")

            with c_risk:
                st.markdown("#### ⚠️ Risk Indicators")
                items = findings.get("risk_indicators", [])
                if items:
                    for it in items:
                        st.error(f"**{it['text']}**")
                else:
                    st.caption("None detected")

            with c_sev:
                st.markdown("#### ⚡ Severity")
                items = findings.get("severity", [])
                if items:
                    for it in items:
                        st.warning(f"**{it['text']}**")
                else:
                    st.caption("None detected")

            with c_dur:
                st.markdown("#### ⏱️ Duration")
                items = findings.get("duration", [])
                if items:
                    for it in items:
                        st.success(f"**{it['text']}**")
                else:
                    st.caption("None detected")

            with c_body:
                st.markdown("#### 📍 Body Location")
                items = findings.get("body_parts", [])
                if items:
                    for it in items:
                        st.markdown(f'<div class="finding-card" style="border-left: 3px solid #14b8a6;"><strong>{it["text"]}</strong></div>', unsafe_allow_html=True)
                else:
                    st.caption("None detected")

            # 5. Clinical Decision Support Rationale Expander
            with st.expander("🩺 View Clinical Decision Support Rationale", expanded=True):
                for r in risk["clinical_rationale"]:
                    st.markdown(f"• **{r}**")

            # 6. JSON Download
            with st.expander("📄 View / Download Clinical JSON"):
                st.json(res)
                st.download_button(
                    "💾 Download JSON Result",
                    data=json.dumps(res, indent=2),
                    file_name="patient_extraction.json",
                    mime="application/json"
                )

# =========================================================================
# TAB 2: MODEL BENCHMARK
# =========================================================================
with tab_models:
    st.subheader("📊 Model Performance Comparison (Review-1 Evaluation)")
    st.markdown("Evaluated on the held-out test split under identical multilingual code-mixed conditions:")

    bench_path = os.path.join(PROJECT_ROOT, "evaluation", "benchmark_results.json")
    if os.path.exists(bench_path):
        with open(bench_path, "r", encoding="utf-8") as f:
            benchmarks = json.load(f)

        rows = []
        for b in benchmarks:
            rows.append({
                "Model Architecture": b["model_name"],
                "Precision (%)": f"{b['overall_precision']:.2f}%",
                "Recall (%)": f"{b['overall_recall']:.2f}%",
                "Overall F1 (%)": f"{b['overall_f1_score']:.2f}%",
                "Hinglish F1": f"{b['language_breakdown'].get('Hinglish', {}).get('f1_score', 'N/A')}%",
                "Kanglish F1": f"{b['language_breakdown'].get('Kanglish', {}).get('f1_score', 'N/A')}%",
                "English F1": f"{b['language_breakdown'].get('English', {}).get('f1_score', 'N/A')}%"
            })

        df_bench = pd.DataFrame(rows)
        st.dataframe(df_bench, use_container_width=True)

        st.markdown("#### 📈 Overall F1-Score (%) by Model Architecture")
        raw_f1 = [b["overall_f1_score"] for b in benchmarks]
        model_names = [b["model_name"] for b in benchmarks]
        chart_df = pd.DataFrame({"F1-Score (%)": raw_f1}, index=model_names)
        st.bar_chart(chart_df)

    st.markdown("""
    ### 💡 Key Takeaways for Review Presentation:
    - **Baseline 1 (CRF):** Achieves **97.79% F1** with sub-5ms CPU latency — optimal for resource-constrained edge clinics.
    - **Baseline 2 (BiLSTM-CRF):** Achieves **94.58% F1** using character-level embeddings, effectively resisting spelling distortions in Romanized Hindi/Kannada (*"dardddd"*, *"novoo"*).
    - **Transformers (IndicBERT / XLM-R / mBERT):** Excel in cross-lingual semantic transfer and handling sentences with dense code-switches.
    """)

# =========================================================================
# TAB 3: DATASET EXPLORER
# =========================================================================
with tab_dataset:
    st.subheader("📁 Multilingual Code-Mixed Clinical Dataset Explorer")
    
    summary_path = os.path.join(PROJECT_ROOT, "data", "dataset_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Records", summary["total_patient_records"])
        d2.metric("Train Split", summary["train_records"])
        d3.metric("Val / Test Splits", f"{summary['val_records']} / {summary['test_records']}")
        d4.metric("Annotated Tokens", summary["total_tokens"])

        st.write("")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### 🏷️ Entity Distribution in Corpus")
            ent_df = pd.DataFrame(list(summary["entity_distribution"].items()), columns=["Entity Category", "Count"])
            st.dataframe(ent_df, use_container_width=True)

        with col_c2:
            st.markdown("#### 🌐 Language Distribution")
            lang_df = pd.DataFrame(list(summary["language_distribution"].items()), columns=["Language Variety", "Count"])
            st.dataframe(lang_df, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔍 Browse Individual Annotated Test Records")
    test_file = os.path.join(PROJECT_ROOT, "data", "test.json")
    if os.path.exists(test_file):
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        
        idx = st.slider("Select Record Number:", 1, len(test_data), 1)
        rec = test_data[idx - 1]
        st.markdown(f"**Patient Text:** *\"{rec['text']}\"*")
        st.markdown(f"**Language:** `{rec['language']}` | **Code-Mixed:** `{rec['code_mixing_detected']}`")
        
        tokens_df = pd.DataFrame({"Token": rec["tokens"], "BIO Tag": rec["bio_tags"]})
        st.dataframe(tokens_df.T, use_container_width=True)

# =========================================================================
# TAB 4: TEAM & PROJECT INFO
# =========================================================================
with tab_team:
    st.subheader("🏛️ Academic Project Information")
    st.markdown("""
    **Dayananda Sagar University (DSU) · School of Engineering**  
    Department of Computer Science and Engineering (Artificial Intelligence and Machine Learning)  
    **Course:** Natural Language Models (`24AM3503`) · Review-1  
    **Project Guide:** Prof. Shivanandan Roy, Assistant Professor  
    **Academic Year:** 2026-2027  
    """)

    st.markdown("### 👥 Student Contributors & Project Roles")
    st.markdown("""
    | Sl. No | Name of Student | USN | Contribution Area |
    | :---: | :--- | :---: | :--- |
    | 1 | **Amal Abraham** | `ENG24AM0138` | Problem Definition & Clinical Schema Design |
    | 2 | **Tarun Karthik M** | `ENG24AM0303` | Methodology, Pipeline Engineering & References |
    | 3 | **M Shrihari** | `ENG24AM0344` | Objectives & Comparative Benchmark Evaluation |
    | 4 | **Sherin Joe** | `ENG24AM0286` | Introduction & Literature Review Analysis |
    """)

    st.markdown("---")
    st.markdown("### 🎯 UN Sustainable Development Goals (SDG) Alignment")
    st.markdown("""
    - **SDG 3 (Good Health & Well-being):** Early symptom extraction and triage risk stratification.
    - **SDG 9 (Industry, Innovation & Infrastructure):** Reusable NLP framework for low-resource Indian languages.
    - **SDG 10 (Reduced Inequalities):** Bringing Romanized and regional-language speakers into digital healthcare tools.
    """)
