// Multilingual Clinical NLP Dashboard Client Logic

const PRESETS = {
    hinglish_cardiac: "Doctor saab, chest mein bahut tez heavy pressure feel ho raha hai since morning and patient is known diabetic with high blood pressure.",
    kanglish_fever: "Nanage last 2 days inda severe fever ide and throat full infection aagide, swallowing madoke thumba novu barthide.",
    tanglish_chest: "Enaku past 4 days ah severe chest pain irukku, and I am cardiac patient on blood thinners since 2 years.",
    hinglish_stomach: "Sir mere pet mein unbearable cramping ho rahi hai since 2 days, and bohot zyada continuous ulti aur weakness hai.",
    english_resp: "Patient is chronic smoker for 15 years, complaining of acute breathlessness and chest tightness for the last 6 hours."
};

let currentResultJson = null;

function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-tab').forEach(el => el.classList.remove('active'));

    const targetTab = document.getElementById(tabId);
    if (targetTab) {
        targetTab.classList.add('active');
    }

    // Highlight active nav tab button
    const tabs = document.querySelectorAll('.nav-tab');
    if (tabId === 'extraction-tab') tabs[0].classList.add('active');
    else if (tabId === 'benchmark-tab') tabs[1].classList.add('active');
    else if (tabId === 'dataset-tab') tabs[2].classList.add('active');
    else if (tabId === 'project-info-tab') tabs[3].classList.add('active');
}

function loadPreset(key) {
    const text = PRESETS[key];
    if (text) {
        document.getElementById('patient-text').value = text;
        runExtraction();
    }
}

async function runExtraction() {
    const textarea = document.getElementById('patient-text');
    const modelSelect = document.getElementById('model-select');
    const text = textarea.value.trim();

    if (!text) {
        alert("Please enter a patient symptom statement or select a sample preset.");
        return;
    }

    const btn = document.getElementById('extract-btn');
    const spinner = document.getElementById('spinner');
    btn.disabled = true;
    if (spinner) spinner.style.display = 'inline-block';

    try {
        const response = await fetch('/api/extract', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                model: modelSelect.value
            })
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const data = await response.json();
        currentResultJson = data;
        renderResults(data);

    } catch (err) {
        console.error("Extraction error:", err);
        alert("Extraction failed: " + err.message);
    } finally {
        btn.disabled = false;
        if (spinner) spinner.style.display = 'none';
    }
}

function renderResults(data) {
    document.getElementById('welcome-state').style.display = 'none';
    document.getElementById('results-panel').style.display = 'block';
    document.getElementById('entity-findings-card').style.display = 'block';

    // Model Badge
    const modelBadge = document.getElementById('model-badge');
    modelBadge.textContent = data.model_used;

    // Linguistic Stats
    const ling = data.linguistic_analysis;
    document.getElementById('res-language').textContent = ling.detected_language;
    document.getElementById('res-cmi').textContent = ling.code_mixing_index > 0 ? (ling.code_mixing_index * 100).toFixed(1) + '%' : '0.0% (Monolingual)';
    document.getElementById('res-entity-count').textContent = data.entities.length;

    // Triage / Risk Profile
    const risk = data.clinical_risk_assessment;
    const riskBox = document.getElementById('risk-box');
    const riskTier = document.getElementById('risk-tier');
    const riskScore = document.getElementById('risk-score-num');
    const riskAction = document.getElementById('risk-action');
    const rationaleList = document.getElementById('rationale-list');

    riskBox.className = 'risk-meter-box tier-' + risk.risk_level;
    riskTier.textContent = risk.risk_level;
    riskScore.textContent = risk.risk_score;
    riskAction.textContent = risk.recommended_urgency;

    rationaleList.innerHTML = '';
    risk.clinical_rationale.forEach(r => {
        const li = document.createElement('li');
        li.textContent = r;
        rationaleList.appendChild(li);
    });

    // Highlighted Text Box
    renderHighlightedText(data.normalized_text, data.entities);

    // Grouped Findings
    renderFindingsGroup('list-symptoms', data.structured_findings.symptoms, 'tag-symptom');
    renderFindingsGroup('list-risk', data.structured_findings.risk_indicators, 'tag-risk');
    renderFindingsGroup('list-severity', data.structured_findings.severity, 'tag-severity');
    renderFindingsGroup('list-duration', data.structured_findings.duration, 'tag-duration');
    renderFindingsGroup('list-body', data.structured_findings.body_parts, 'tag-body');
}

function renderHighlightedText(text, entities) {
    const container = document.getElementById('highlighted-text');
    if (!entities || entities.length === 0) {
        container.textContent = text;
        return;
    }

    // Sort non-overlapping entities by start offset
    const sorted = [...entities].sort((a, b) => a.start - b.start);
    let html = '';
    let cursor = 0;

    sorted.forEach(ent => {
        if (ent.start > cursor) {
            html += escapeHtml(text.substring(cursor, ent.start));
        }

        const tagClass = 'tag-' + ent.label.toLowerCase().replace('_indicator', '').replace('_part', '');
        html += `<span class="entity-span ${tagClass}" data-tag="${ent.label}">${escapeHtml(ent.text)}</span>`;
        cursor = Math.max(cursor, ent.end);
    });

    if (cursor < text.length) {
        html += escapeHtml(text.substring(cursor));
    }

    container.innerHTML = html;
}

function renderFindingsGroup(containerId, list, pillClass) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    if (!list || list.length === 0) {
        container.innerHTML = '<span style="color: #94a3b8; font-size: 0.8rem; font-style: italic;">None detected</span>';
        return;
    }

    list.forEach(item => {
        const div = document.createElement('div');
        div.className = 'finding-item';
        div.innerHTML = `<strong>${escapeHtml(item.text)}</strong>` +
            (item.standardized_concept && item.standardized_concept !== item.text ? 
                `<span class="concept">→ Concept: <em>${escapeHtml(item.standardized_concept)}</em></span>` : '');
        container.appendChild(div);
    });
}

function copyJson() {
    if (!currentResultJson) return;
    navigator.clipboard.writeText(JSON.stringify(currentResultJson, null, 2))
        .then(() => alert("Structured clinical JSON copied to clipboard!"))
        .catch(err => alert("Copy failed: " + err));
}

function escapeHtml(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
