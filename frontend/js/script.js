/* ============================================================
   GLOBAL STATE
============================================================ */
const AppState = {
    lastResult: null,
    batchResults: null
};

/* ============================================================
   UTILITY HELPERS
============================================================ */

/** Try safe JSON parse; fallback to text */
async function safeParseResponse(res) {
    try { return await res.json(); }
    catch {
        const text = await res.text();
        return { success: false, error: "Invalid JSON", raw: text };
    }
}

/** Show toast or fallback */
function showToast(msg, type = "info") {
    try {
        if (window.Swal) Swal.fire(type.toUpperCase(), msg, type);
        else alert(msg);
    } catch {
        alert(msg);
    }
}

/** Save history in unified format */
function saveHistory(entry) {
    let history = JSON.parse(localStorage.getItem("aadhaar_history") || "[]");
    history.push({
        status: entry.status || entry.overall_assessment || "Unknown",
        fraud_score: entry.fraud_score ?? 0,
        verification_type: entry.verification_type || "single",
        time: new Date().toISOString(),
        extracted: entry.extracted || null
    });
    localStorage.setItem("aadhaar_history", JSON.stringify(history));
}

/* ============================================================
   SINGLE VERIFICATION (used in verify-enhanced, services, etc.)
============================================================ */
async function runSingleVerification(formData) {
    const url = `${API_BASE_URL}/api/verify_single`;

    const response = await fetch(url, {
        method: "POST",
        body: formData
    });

    const json = await safeParseResponse(response);
    if (!json.success) return { success: false, error: json.error || "Processing failed" };

    return { success: true, result: json.result };
}

/* ============================================================
   BATCH VERIFICATION
============================================================ */
async function runBatchVerification(formData) {
    const url = `${API_BASE_URL}/api/verify_batch`;

    const response = await fetch(url, {
        method: "POST",
        body: formData
    });

    const json = await safeParseResponse(response);
    if (!json.success) return { success: false, error: json.error || "Processing failed" };

    return { success: true, results: json.results };
}

/* ============================================================
   HOOK FOR SERVICES.HTML UPLOAD
============================================================ */
window.handleVerificationSubmitForm = async function (e) {
    e.preventDefault();

    const form = document.getElementById("verifyForm");
    const front = form.front?.files[0];
    const back = form.back?.files[0];
    const zipFile = form.zip?.files[0];
    const qrFlag = form.qr?.checked ? "1" : "0";

    // SINGLE
    if (front) {
        const fd = new FormData();
        fd.append("front", front);
        if (back) fd.append("back", back);
        fd.append("qr", qrFlag);

        showToast("Processing single verification…");

        const result = await runSingleVerification(fd);

        if (!result.success) {
            showToast(result.error, "error");
            return;
        }

        AppState.lastResult = result.result;
        saveHistory({
            ...result.result,
            verification_type: "single"
        });

        renderSingleResult(result.result);
        return;
    }

    // BATCH
    if (zipFile) {
        const fd = new FormData();
        fd.append("zip", zipFile);

        showToast("Processing batch verification…");

        const result = await runBatchVerification(fd);

        if (!result.success) {
            showToast(result.error, "error");
            return;
        }

        AppState.batchResults = result.results;

        result.results.forEach(item => {
            saveHistory({
                ...item,
                verification_type: "batch"
            });
        });

        renderBatchResults(result.results);
        return;
    }

    showToast("Please upload at least one file.", "warning");
};

/* ============================================================
   RENDER RESULT (SINGLE)
============================================================ */
function renderSingleResult(result) {
    const box = document.getElementById("verificationResults");
    if (!box) return;

    box.innerHTML = `
        <div class="result-card">
            <h3>Verification Result</h3>
            <p>Status: <strong>${result.status || result.overall_assessment}</strong></p>
            <p>Fraud Score: <strong>${result.fraud_score}</strong></p>
        </div>
    `;
}

/* ============================================================
   RENDER BATCH RESULTS
============================================================ */
function renderBatchResults(list) {
    const box = document.getElementById("verificationResults");
    if (!box) return;

    box.innerHTML = list.map(res => `
        <div class="result-card">
            <p><strong>${res.file || "File"}</strong></p>
            <p>Status: ${res.status || res.overall_assessment}</p>
            <p>Fraud Score: ${res.fraud_score}</p>
        </div>
    `).join("");
}

/* ============================================================
   DOWNLOAD HANDLERS (CSV / JSON)
============================================================ */
window.downloadJSON = function () {
    if (!AppState.lastResult && !AppState.batchResults) {
        showToast("Nothing to download!", "warning");
        return;
    }

    const data = AppState.lastResult || AppState.batchResults;

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "aadhaar_verification.json";
    a.click();
};

/* ============================================================
   LOGIN HANDLING (USED ACROSS PAGES)
============================================================ */
window.handleLogin = function (username, role) {
    localStorage.setItem("user_name", username);
    localStorage.setItem("user_role", role);
};

/* ============================================================
   LOGOUT HANDLER
============================================================ */
window.performLogout = function () {
    localStorage.removeItem("user_name");
    localStorage.removeItem("user_role");
    window.location.href = "login.html";
};
