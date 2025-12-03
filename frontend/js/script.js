/* script.js — Merged, cleaned, A2 version (duplicate theme block removed)
   Features:
   - Theme toggle (no flicker)
   - Mode switch (single / batch / api)
   - Verification form handling + fetch
   - Display single/batch results, expanded views
   - Download JSON/CSV
   - Utilities & validation
*/

/* =========================
   EARLY THEME APPLY (prevent flicker)
   This runs immediately to avoid visible theme flash.
   If you already have an inline <script> in <head> doing this,
   this is safe — it's idempotent.
   ========================= */
(function applySavedThemeEarly() {
    try {
        const saved = localStorage.getItem('theme');
        if (saved === 'dark') document.documentElement.classList.add('dark');
        else document.documentElement.classList.remove('dark');
    } catch (e) {
        console.warn('Theme early apply failed', e);
    }
})();

/* =========================
   Global config + state
   ========================= */
const API_BASE_URL = window.location.origin;
let currentMode = 'single';

/* =========================
   DOMContentLoaded initialization
   ========================= */
document.addEventListener('DOMContentLoaded', () => {
    console.log('script.js loaded');

    initThemeToggle();
    initServiceUI();
    initVerificationForm();
    initModalCloseOnOutside();

    const uploadSection = document.getElementById('uploadSection');
    if (uploadSection) uploadSection.style.display = 'none';
});

/* =========================
   THEME TOGGLE SYSTEM
   ========================= */
function initThemeToggle() {
    const themeBtn = document.getElementById('themeBtn');
    const themeMenu = document.getElementById('themeMenu');

    // ✔ FIX: universal selectors (works on all your HTML pages)
    const lightModeEl =
        document.querySelector('.theme-option[data-theme="light"]') ||
        document.getElementById('lightMode');

    const darkModeEl =
        document.querySelector('.theme-option[data-theme="dark"]') ||
        document.getElementById('darkMode');

    if (!themeBtn || !themeMenu) {
        updateThemeIcon();
        return;
    }

    updateThemeIcon();

    themeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        themeMenu.classList.toggle('hidden');
    });

    document.addEventListener('click', () => {
        themeMenu.classList.add('hidden');
    });

    themeMenu.addEventListener('click', (e) => e.stopPropagation());

    if (lightModeEl) {
        lightModeEl.addEventListener('click', () => {
            setTheme('light');
            themeMenu.classList.add('hidden');
        });
    }

    if (darkModeEl) {
        darkModeEl.addEventListener('click', () => {
            setTheme('dark');
            themeMenu.classList.add('hidden');
        });
    }
}

function setTheme(mode) {
    try {
        if (mode === 'dark') document.documentElement.classList.add('dark');
        else document.documentElement.classList.remove('dark');
        localStorage.setItem('theme', mode);
    } catch (e) {
        console.warn('setTheme failed', e);
    }
    updateThemeIcon();
}

function updateThemeIcon() {
    const themeBtn = document.getElementById('themeBtn');
    const current = localStorage.getItem('theme') === 'dark' ? 'dark' : 'light';
    if (!themeBtn) return;

    themeBtn.textContent = current === 'dark' ? '🌙' : '☀️';
}

/* =========================
   SERVICE CARDS + MODE SELECT
   ========================= */
function initServiceUI() {
    const serviceCards = document.querySelectorAll('.service-card[data-service]');
    if (serviceCards && serviceCards.length) {
        serviceCards.forEach(card => {
            card.addEventListener('click', () => {
                serviceCards.forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');

                const uploadSection = document.getElementById('uploadSection');
                if (uploadSection) {
                    uploadSection.style.display = 'block';
                    uploadSection.scrollIntoView({ behavior: 'smooth' });
                }

                const svc = card.getAttribute('data-service');
                if (svc === 'single' || svc === 'batch') setMode(svc);
                else if (svc === 'api') setMode('api');
            });
        });
    }

    const singleBtn = document.getElementById('singleBtn');
    const batchBtn = document.getElementById('batchBtn');
    if (singleBtn) singleBtn.addEventListener('click', () => setMode('single'));
    if (batchBtn) batchBtn.addEventListener('click', () => setMode('batch'));
}

/* Unified setMode */
function setMode(mode) {
    currentMode = mode;
    console.log('setMode ->', mode);

    const uploadTitle = document.getElementById('uploadTitle');
    const uploadDescription = document.getElementById('uploadDescription');
    const verifyForm = document.getElementById('verifyForm');
    const singleUpload = document.getElementById('singleUpload');
    const batchUpload = document.getElementById('batchUpload');
    const qrCheckboxContainer = document.getElementById('qrCheckboxContainer');
    const verificationResults = document.getElementById('verificationResults');

    if (uploadTitle && uploadDescription) {
        if (mode === 'single') {
            uploadTitle.textContent = 'Single Aadhaar Verification';
            uploadDescription.textContent = 'Upload Aadhaar card images for verification. Our system will auto-detect Aadhaar cards.';
        } else if (mode === 'batch') {
            uploadTitle.textContent = 'Batch Aadhaar Verification';
            uploadDescription.textContent = 'Upload a ZIP file containing multiple Aadhaar images.';
        } else if (mode === 'api') {
            uploadTitle.textContent = 'API Integration';
            uploadDescription.textContent = 'Contact us for API access.';
        }
    }

    if (singleUpload) singleUpload.style.display = mode === 'single' ? 'block' : 'none';
    if (batchUpload) batchUpload.style.display = mode === 'batch' ? 'block' : 'none';

    if (qrCheckboxContainer) qrCheckboxContainer.style.display = mode === 'single' ? 'block' : 'none';
    if (verifyForm) verifyForm.style.display = mode === 'api' ? 'none' : 'block';

    if (verificationResults)
        verificationResults.innerHTML = '<p>Results will appear here after verification.</p>';
}

/* =========================
   Verification form handling
   ========================= */
function initVerificationForm() {
    const verifyForm = document.getElementById('verifyForm');
    if (!verifyForm) return;

    verifyForm.removeEventListener('submit', handleVerificationSubmit);
    verifyForm.addEventListener('submit', handleVerificationSubmit);

    const zipInput = document.getElementById('zip');
    if (zipInput && zipInput.offsetParent === null) {
        zipInput.removeAttribute('required');
    }
}

/* show/hide spinner */
function showLoading(show) {
    const spinner = document.getElementById('loadingSpinner');
    if (!spinner) return;
    spinner.style.display = show ? 'block' : 'none';
}
/* Main submit handler */
async function handleVerificationSubmit(e) {
    e.preventDefault();
    console.log('handleVerificationSubmit — mode:', currentMode);

    const verificationResults = document.getElementById('verificationResults');
    if (verificationResults) verificationResults.innerHTML = '<p>Processing... please wait.</p>';

    showLoading(true);

    const formData = new FormData();
    try {
        if (currentMode === 'single') {
            const front = document.getElementById('front');
            if (!front || !front.files || !front.files[0]) {
                alert('Please select front image file.');
                showLoading(false);
                return;
            }
            formData.append('front', front.files[0]);

            const back = document.getElementById('back');
            if (back && back.files && back.files[0]) formData.append('back', back.files[0]);

            const qrCheckbox = document.querySelector('input[name="qr"]');
            formData.append('qr', qrCheckbox?.checked ? 'true' : 'false');

        } else if (currentMode === 'batch') {
            const zip = document.getElementById('zip');
            if (!zip || !zip.files || !zip.files[0]) {
                alert('Please select a ZIP file.');
                showLoading(false);
                return;
            }
            formData.append('zip', zip.files[0]);
        } else {
            showLoading(false);
            if (verificationResults) {
                verificationResults.innerHTML = '<p>API mode selected — contact sales.</p>';
            }
            return;
        }

        const endpoint = currentMode === 'single'
            ? `${API_BASE_URL}/api/verify_single`
            : `${API_BASE_URL}/api/verify_batch`;

        console.log('Sending to', endpoint);

        const resp = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        console.log('Response status', resp.status);

        if (!resp.ok) {
            let errMsg = `HTTP ${resp.status}`;
            try {
                const errJson = await resp.json();
                errMsg = errJson.error || JSON.stringify(errJson);
            } catch (_) {
                try {
                    const txt = await resp.text();
                    if (txt) errMsg = txt;
                } catch {}
            }
            throw new Error(errMsg);
        }

        const data = await resp.json();
        showLoading(false);

        if (currentMode === 'single') {
            if (data.success && data.result) displaySingleResult(data.result);
            else if (data.error === 'NOT_AADHAAR') displayNonAadhaarResult(data.result || data);
            else if (data.result) displaySingleResult(data.result);
            else {
                const msg = data.error || 'Unexpected response';
                verificationResults.innerHTML = `<p style="color:red">${msg}</p>`;
            }
        } else {
            if (data.results || data.summary) {
                const results = data.results || data;
                displayBatchResults(results, data.summary || null);
            } else {
                displayBatchResults(data);
            }
        }

    } catch (err) {
        console.error('Verification error:', err);
        showLoading(false);
        if (verificationResults) {
            verificationResults.innerHTML = `<p style="color:red">Error: ${err.message}</p>`;
        }
    }
}

/* =========================
   RESULT DISPLAY FUNCTIONS
   ========================= */

function displaySingleResult(result) {
    const out = document.getElementById('verificationResults');
    if (!out) return;

    if (!result || result.error === 'NOT_AADHAAR') {
        displayNonAadhaarResult(result);
        return;
    }

    const riskLevel = result.assessment || 'UNKNOWN';
    const fraudScore = result.fraud_score ?? 0;

    const riskClass =
        riskLevel === 'HIGH' ? 'risk-high' :
        riskLevel === 'MODERATE' ? 'risk-medium' :
        'risk-low';

    const riskTagClass =
        riskLevel === 'HIGH' ? 'risk-high-tag' :
        riskLevel === 'MODERATE' ? 'risk-medium-tag' :
        'risk-low-tag';

    let html = `
        <div class="result-card success" id="singleResultCard">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <h4>✅ Valid Aadhaar Card Detected</h4>
                <span class="risk-tag ${riskTagClass}">${riskLevel} RISK</span>
            </div>

            <div class="detail-item" style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span><strong>Fraud Score:</strong></span>
                <span class="${riskClass}">${fraudScore}</span>
            </div>

            <div class="detail-item" style="display:flex; justify-content:space-between; margin-bottom:12px;">
                <span><strong>Aadhaar Verification Confidence:</strong></span>
                <span>${result.aadhaar_verification?.confidence_score ?? 'N/A'}%</span>
            </div>

            ${addDownloadButtonToSingle(result)}

            <div style="text-align:center; margin-top:10px;">
                <button onclick="showSingleFullDetails(window.__lastResultPlaceholder || {})"
                    class="btn" style="background:#6c757d;">View Full Details</button>
            </div>
        </div>
    `;

    window.__lastResultPlaceholder = result;

    out.innerHTML = html;
    out.scrollIntoView({ behavior: 'smooth' });
}

function showSingleFullDetails(result) {
    result = result || window.__lastResultPlaceholder;
    if (!result) return;

    const out = document.getElementById('verificationResults');
    if (!out) return;

    const riskLevel = result.assessment || 'UNKNOWN';
    const fraudScore = result.fraud_score ?? 0;
    const riskTagClass =
        riskLevel === 'HIGH' ? 'risk-high-tag' :
        riskLevel === 'MODERATE' ? 'risk-medium-tag' :
        'risk-low-tag';

    let html = `
        <div class="result-card success expanded">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h4>✅ Valid Aadhaar Card Detected - Full Details</h4>
                <span class="risk-tag ${riskTagClass}">${riskLevel} RISK</span>
            </div>

            <div style="margin-top:10px;">
                <div class="detail-item" style="display:flex; justify-content:space-between;">
                    <span><strong>Fraud Score:</strong></span><span>${fraudScore}</span>
                </div>
                <div class="detail-item" style="display:flex; justify-content:space-between;">
                    <span><strong>Verification Confidence:</strong></span><span>${result.aadhaar_verification?.confidence_score ?? 'N/A'}%</span>
                </div>
            </div>
    `;

    if (result.extracted) {
        html += `
            <h5 style="margin-top:20px;">Extracted Information:</h5>
            <div class="verification-details">
                <div class="detail-item"><span><strong>Name:</strong></span><span>${result.extracted.name || 'Not found'}</span></div>
                <div class="detail-item"><span><strong>DOB:</strong></span><span>${result.extracted.dob || 'Not found'}</span></div>
                <div class="detail-item"><span><strong>Gender:</strong></span><span>${result.extracted.gender || 'Not found'}</span></div>
                <div class="detail-item"><span><strong>Aadhaar:</strong></span><span>${result.extracted.aadhaar || 'Not found'}</span></div>
            </div>
        `;
    }

    if (result.indicators?.length) {
        html += `<h5 style="margin-top:18px;">Verification Indicators:</h5><div class="verification-details">`;
        result.indicators.forEach(ind => {
            html += `<div class="detail-item"><span>${ind}</span></div>`;
        });
        html += `</div>`;
    }

    if (result.annotated_b64) {
        html += `
            <h5 style="margin-top:18px;">Processed Image:</h5>
            <div style="text-align:center; margin:12px 0;">
                <img src="data:image/jpeg;base64,${result.annotated_b64}"
                    alt="Annotated" style="max-width:100%; border-radius:8px; border:1px solid #e6e6e6;" />
            </div>
        `;
    }

    html += `
        <div style="text-align:center; margin-top:16px;">
            <button onclick="collapseSingleDetails()" class="btn" style="background:#6c757d;">Show Less</button>
            ${addDownloadButtonInline(result)}
        </div>
    </div>
    `;

    window.__lastResultPlaceholder = result;
    out.innerHTML = html;
    out.scrollIntoView({ behavior: 'smooth' });
}

function collapseSingleDetails() {
    const result = window.__lastResultPlaceholder;
    if (!result) return;
    displaySingleResult(result);
}
/* =========================
   BATCH RESULT DISPLAY
   ========================= */

function displayBatchResults(results, summary = null) {
    const out = document.getElementById('verificationResults');
    if (!out) return;

    if (!Array.isArray(results) && results.results) {
        summary = results.summary || summary;
        results = results.results;
    }

    if (!Array.isArray(results) || results.length === 0) {
        out.innerHTML = `
            <div class="result-card error">
                <h4>❌ No Results</h4>
                <p>No files processed.</p>
            </div>`;
        return;
    }

    const total = results.length;
    const validCount = results.filter(r => !r.error).length;
    const nonAadhaarCount = results.filter(r => r.error === 'NOT_AADHAAR').length;
    const errorCount = results.filter(r => r.error && r.error !== 'NOT_AADHAAR').length;

    let html = `
        <div class="result-card info">
            <h4>📊 Batch Processing Complete</h4>
            <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; margin-top:12px;">
                <div class="batch-box"><strong>${total}</strong><div>Total</div></div>
                <div class="batch-box"><strong>${validCount}</strong><div>Processed</div></div>
                <div class="batch-box"><strong>${nonAadhaarCount}</strong><div>Non-Aadhaar</div></div>
                <div class="batch-box"><strong>${errorCount}</strong><div>Errors</div></div>
            </div>

            <div style="margin-top:16px; max-height:300px; overflow:auto; border-radius:8px;">
                <table class="batch-table">
                    <thead>
                        <tr><th>Filename</th><th>Status</th><th>Risk</th></tr>
                    </thead>
                    <tbody>
    `;

    const slice = results.slice(0, 30);

    slice.forEach((r, idx) => {
        const isNon = r.error === 'NOT_AADHAAR';
        const isErr = r.error && r.error !== 'NOT_AADHAAR';

        let status = 'Valid Aadhaar';
        let color = '#28a745';

        if (isNon) { status = 'Non-Aadhaar'; color = '#ff9800'; }
        if (isErr) { status = 'Error'; color = '#d32f2f'; }

        const risk = isNon || isErr ? 'N/A' : (r.assessment || 'UNKNOWN');

        html += `
            <tr>
                <td>${r.filename || `File ${idx+1}`}</td>
                <td style="color:${color};">${status}</td>
                <td>${risk}</td>
            </tr>
        `;
    });

    if (results.length > slice.length) {
        html += `
            <tr><td colspan="3" style="text-align:center;">
                ... ${results.length - slice.length} more files
            </td></tr>`;
    }

    html += `
                    </tbody>
                </table>
            </div>

            <div style="text-align:center; margin-top:16px;">
                <button class="btn" style="background:#28a745;" onclick="downloadBatchResultsFromUI()">📥 Download Full Report</button>
                <button class="btn" style="background:#0078d4;" onclick="showBatchFullDetailsFromUI()">View All Details</button>
            </div>
        </div>
    `;

    window.__lastBatchResults = results;

    out.innerHTML = html;
    out.scrollIntoView({ behavior: 'smooth' });
}

function showBatchFullDetailsFromUI() {
    const results = window.__lastBatchResults || [];
    showBatchFullDetails(results);
}

function showBatchFullDetails(results) {
    const out = document.getElementById('verificationResults');
    if (!out) return;

    if (!Array.isArray(results) || results.length === 0) {
        out.innerHTML = `
            <div class="result-card error">
                <h4>No batch results available</h4>
            </div>`;
        return;
    }

    let html = `
        <div class="result-card info expanded">
            <div style="display:flex; justify-content:space-between;">
                <h4>📊 Batch Processing - Full Details</h4>
                <button class="btn" style="background:#6c757d;" onclick="displayBatchResults(window.__lastBatchResults)">Show Summary</button>
            </div>

            <div style="overflow-x:auto; margin-top:12px;">
                <table class="batch-table">
                    <thead>
                        <tr>
                            <th>Filename</th>
                            <th>Status</th>
                            <th>Risk</th>
                            <th>Fraud</th>
                            <th>Aadhaar</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
    `;

    results.forEach((r, i) => {
        const isNon = r.error === 'NOT_AADHAAR';
        const isErr = r.error && r.error !== 'NOT_AADHAAR';

        const filename = r.filename || `File ${i+1}`;
        const status = isNon ? 'Non-Aadhaar' : isErr ? 'Error' : 'Valid Aadhaar';
        const risk = (isNon || isErr) ? 'N/A' : (r.assessment || 'UNKNOWN');
        const fraud = r.fraud_score ?? 'N/A';
        const aadhaar = r.extracted?.aadhaar || 'N/A';
        const confidence = isNon
            ? 'N/A'
            : r.aadhaar_verification?.confidence_score
                ? `${r.aadhaar_verification.confidence_score}%`
                : 'N/A';

        html += `
            <tr>
                <td>${filename}</td>
                <td>${status}</td>
                <td>${risk}</td>
                <td>${fraud}</td>
                <td>${aadhaar}</td>
                <td>${confidence}</td>
            </tr>
        `;
    });

    html += `
                    </tbody>
                </table>
            </div>

            <div style="text-align:center; margin-top:16px;">
                ${addDownloadButtonToBatchInline(results)}
            </div>
        </div>
    `;

    out.innerHTML = html;
    out.scrollIntoView({ behavior: 'smooth' });

    window.__lastBatchResults = results;
}

/* =========================
   NON-AADHAAR UI
   ========================= */

function displayNonAadhaarResult(result) {
    const out = document.getElementById('verificationResults');
    if (!out) return;

    const confidence =
        result?.confidence_score ??
        result?.aadhaar_verification?.confidence_score ??
        0;

    const details = result?.aadhaar_verification_details || {};

    let barColor = '#d32f2f';
    if (confidence >= 50) barColor = '#ff9800';
    if (confidence >= 70) barColor = '#4caf50';

    let html = `
        <div class="non-aadhaar-alert">
            <h4>⚠️ Not an Aadhaar Card</h4>
            <p><strong>Message:</strong> ${result?.message || 'This image is not detected as an Aadhaar card.'}</p>

            <div class="conf-box">
                <strong>Confidence: ${confidence}%</strong>
                <div class="conf-bar">
                    <div class="conf-fill" style="width:${confidence}%; background:${barColor};"></div>
                </div>
            </div>

            <h5>Verification Details:</h5>
            <div class="verification-details">
                <div class="detail-item"><span>Keywords Found:</span><span>${details.keywords_found || 0}</span></div>
                <div class="detail-item"><span>Aadhaar Patterns:</span><span>${details.aadhaar_numbers_found || 0}</span></div>
                <div class="detail-item"><span>Aspect Ratio:</span><span>${details.aspect_ratio_valid ? 'Valid' : 'Invalid'}</span></div>
                <div class="detail-item"><span>Image Size:</span><span>${details.size_valid ? 'Adequate' : 'Too Small'}</span></div>
            </div>

            <h5>Recommendation</h5>
            <ul>
                <li>Upload clear Aadhaar card image</li>
                <li>Show the 12-digit Aadhaar number</li>
                <li>Proper rectangular aspect ratio</li>
                <li>Readable text with “Aadhaar / UIDAI” keywords</li>
            </ul>
        </div>
    `;

    out.innerHTML = html;
    out.scrollIntoView({ behavior: 'smooth' });
}

/* =========================
   DOWNLOAD HELPERS
   ========================= */

function downloadSingleResult(result) {
    if (!result) result = window.__lastResultPlaceholder;
    if (!result) return alert('No result to download');

    const filename = `aadhaar_single_${new Date().toISOString().replace(/[:.]/g, '-')}`;

    const payload = {
        verification_type: 'single',
        timestamp: new Date().toISOString(),
        overall_assessment: result.assessment || 'UNKNOWN',
        fraud_score: result.fraud_score || 0,
        confidence: result.aadhaar_verification?.confidence_score || 'N/A',
        extracted: result.extracted || {},
        indicators: result.indicators || []
    };

    const json = JSON.stringify(payload, null, 2);
    const csv = convertSingleToCSV(payload);

    showDownloadOptions(filename, json, csv);
}

function downloadBatchResultsFromUI() {
    const results = window.__lastBatchResults || [];
    if (!results.length) return alert('No batch results');

    downloadBatchResults(results);
}

function downloadBatchResults(results) {
    const filename = `aadhaar_batch_${new Date().toISOString().replace(/[:.]/g, '-')}`;

    const data = {
        verification_type: 'batch',
        timestamp: new Date().toISOString(),
        total_files: results.length,
        results: results.map(r => ({
            filename: r.filename,
            is_aadhaar: !r.error || r.error !== 'NOT_AADHAAR',
            error: r.error || null,
            assessment: r.assessment || 'UNKNOWN',
            fraud: r.fraud_score || 0,
            confidence: r.aadhaar_verification?.confidence_score || 'N/A',
            extracted: r.extracted || {}
        }))
    };

    const json = JSON.stringify(data, null, 2);
    const csv = convertBatchToCSV(data);

    showDownloadOptions(filename, json, csv);
}

function showDownloadOptions(filename, jsonData, csvData) {
    const old = document.getElementById('__download_modal');
    if (old) old.remove();

    const modal = document.createElement('div');
    modal.id = '__download_modal';
    modal.className = 'download-modal';

    modal.innerHTML = `
        <div class="download-box">
            <h3>Download Result</h3>
            <button class="btn" id="dl_json">JSON</button>
            <button class="btn" id="dl_csv">CSV</button>
            <button class="btn" id="dl_close" style="background:#555;">Cancel</button>
        </div>
    `;

    document.body.appendChild(modal);

    document.getElementById('dl_close').onclick = () => modal.remove();
    document.getElementById('dl_json').onclick = () => {
        downloadFile(`${filename}.json`, btoa(unescape(encodeURIComponent(jsonData))), 'application/json');
        modal.remove();
    };
    document.getElementById('dl_csv').onclick = () => {
        downloadFile(`${filename}.csv`, btoa(unescape(encodeURIComponent(csvData))), 'text/csv');
        modal.remove();
    };
}

function downloadFile(filename, base64, type) {
    const a = document.createElement('a');
    a.href = `data:${type};base64,${base64}`;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
}

/* =========================
   CSV HELPERS
   ========================= */
function convertSingleToCSV(data) {
    const rows = [
        ['Field','Value'],
        ['Assessment', data.overall_assessment],
        ['Fraud Score', data.fraud_score],
        ['Confidence', data.confidence],
        ['Aadhaar Number', data.extracted.aadhaar || ''],
        ['Name', data.extracted.name || ''],
        ['DOB', data.extracted.dob || ''],
        ['Gender', data.extracted.gender || '']
    ];

    return rows.map(r => r.map(v => `"${v}"`).join(',')).join('\n');
}

function convertBatchToCSV(batch) {
    const headers = [
        'Filename','Is Aadhaar','Error','Assessment','Fraud Score','Confidence',
        'Aadhaar','Name','DOB','Gender'
    ];
    const lines = [headers.join(',')];

    batch.results.forEach(r => {
        lines.push([
            `"${r.filename || ''}"`,
            `"${r.is_aadhaar ? 'Yes' : 'No'}"`,
            `"${r.error || ''}"`,
            `"${r.assessment}"`,
            `"${r.fraud}"`,
            `"${r.confidence}"`,
            `"${r.extracted?.aadhaar || ''}"`,
            `"${r.extracted?.name || ''}"`,
            `"${r.extracted?.dob || ''}"`,
            `"${r.extracted?.gender || ''}"`
        ].join(','));
    });

    return lines.join('\n');
}

/* =========================
   GLOBAL WINDOW EXPORTS
   ========================= */
window.setTheme = setTheme;
window.setMode = setMode;
window.showSingleFullDetails = showSingleFullDetails;
window.collapseSingleDetails = collapseSingleDetails;
window.showBatchFullDetails = showBatchFullDetails;
window.downloadSingleResult = downloadSingleResult;
window.downloadBatchResults = downloadBatchResults;
window.downloadFile = downloadFile;
/* =========================
   MISSING FUNCTION FIXES
   ========================= */

// Fix for modal close
function initModalCloseOnOutside() {
    document.addEventListener("click", (e) => {
        const modal = document.querySelector(".download-modal");
        if (!modal) return;

        // If clicked outside modal, close it
        if (e.target === modal) modal.remove();
    });
}

// Fix for download buttons
function addDownloadButtonToSingle(result) {
    return `
        <div style="text-align:center; margin-top:12px;">
            <button class="btn" style="background:#28a745;"
                onclick="downloadSingleResult(window.__lastResultPlaceholder)">
                📥 Download JSON/CSV
            </button>
        </div>
    `;
}

function addDownloadButtonInline(result) {
    return `
        <button class="btn" style="background:#28a745; margin-left:10px;"
            onclick="downloadSingleResult(window.__lastResultPlaceholder)">
            📥 Download JSON/CSV
        </button>
    `;
}

function addDownloadButtonToBatchInline(results) {
    return `
        <button class="btn" style="background:#28a745;"
            onclick="downloadBatchResults(window.__lastBatchResults)">
            📥 Download Full Report
        </button>
    `;
}
