const API_BASE = "";
let currentUser = null;
let currentTab = "auth";
let authMode = "login";

let historicalChart = null;
let predictionChart = null;

document.addEventListener("DOMContentLoaded", () => {
    lucide.createIcons();
    const saved = localStorage.getItem("nepse_user");
    if (saved) {
        currentUser = JSON.parse(saved);
        updateUIForRole(currentUser.role);
        switchTab("dashboard");
        loadDashboardData();
    } else {
        switchTab("auth");
    }
});

function toggleAuthMode(mode) {
    authMode = mode;
    const btnLogin = document.getElementById("btnTabLogin");
    const btnRegister = document.getElementById("btnTabRegister");
    const fieldName = document.getElementById("fieldName");
    const authRoleField = document.getElementById("authRoleField");
    const authTitle = document.getElementById("authTitle");
    const authSubmitBtn = document.getElementById("authSubmitBtn");

    if (mode === "register") {
        btnLogin.className = "py-2 rounded-md text-slate-500 hover:text-slate-700 transition-colors";
        btnRegister.className = "py-2 rounded-md bg-white text-slate-800 shadow-sm transition-colors";
        fieldName.classList.remove("hidden");
        authRoleField.classList.remove("hidden");
        authTitle.innerText = "Create Your Account";
        authSubmitBtn.innerText = "Register Account";
    } else {
        btnRegister.className = "py-2 rounded-md text-slate-500 hover:text-slate-700 transition-colors";
        btnLogin.className = "py-2 rounded-md bg-white text-slate-800 shadow-sm transition-colors";
        fieldName.classList.add("hidden");
        authRoleField.classList.add("hidden");
        authTitle.innerText = "Welcome Back";
        authSubmitBtn.innerText = "Sign In";
    }
    hideAuthError();
}

async function handleAuthSubmit(e) {
    e.preventDefault();
    const email = document.getElementById("authEmail").value;
    const password = document.getElementById("authPassword").value;
    const errorEl = document.getElementById("authError");
    const btn = document.getElementById("authSubmitBtn");

    btn.classList.add("btn-loading");
    hideAuthError();

    try {
        let res;
        if (authMode === "register") {
            const name = document.getElementById("regName").value;
            const role = document.getElementById("authRole").value;
            res = await fetch(`${API_BASE}/api/auth/register`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name, email, password, role }),
            });
        } else {
            res = await fetch(`${API_BASE}/api/auth/login`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email, password }),
            });
        }

        if (!res.ok) {
            const data = await res.json();
            throw new Error(data.detail || "Authentication failed");
        }

        const data = await res.json();
        currentUser = data;
        localStorage.setItem("nepse_user", JSON.stringify(data));
        updateUIForRole(data.role);
        switchTab("dashboard");
        await loadDashboardData();
    } catch (err) {
        showAuthError(err.message);
    } finally {
        btn.classList.remove("btn-loading");
    }
}

function logout() {
    currentUser = null;
    localStorage.removeItem("nepse_user");
    document.getElementById("authEmail").value = "";
    document.getElementById("authPassword").value = "";
    switchTab("auth");
}

function updateUIForRole(role) {
    const navAdmin = document.getElementById("nav-admin");
    const roleBadge = document.getElementById("roleBadge");
    const currentUserLabel = document.getElementById("currentUserLabel");

    if (role === "admin") {
        navAdmin.classList.remove("hidden");
        roleBadge.innerText = "Admin Access";
        roleBadge.className = "text-xs text-amber-500 font-semibold uppercase";
        currentUserLabel.innerText = "Active: Admin Mode";
    } else {
        navAdmin.classList.add("hidden");
        roleBadge.innerText = "User Access";
        roleBadge.className = "text-xs text-brand-500 font-semibold uppercase";
        currentUserLabel.innerText = "Active: User Mode";
    }
}

function showAuthError(msg) {
    const el = document.getElementById("authError");
    el.innerText = msg;
    el.classList.remove("hidden");
}

function hideAuthError() {
    const el = document.getElementById("authError");
    el.classList.add("hidden");
}

function switchTab(tabId) {
    currentTab = tabId;
    const sidebar = document.getElementById("appSidebar");

    if (tabId === "auth") {
        sidebar.classList.add("hidden");
    } else {
        sidebar.classList.remove("hidden");
    }

    ["auth", "dashboard", "prediction", "admin"].forEach((id) => {
        const el = document.getElementById(`page-${id}`);
        if (el) el.classList.add("hidden");

        const navBtn = document.getElementById(`nav-${id}`);
        if (navBtn) {
            navBtn.classList.remove("bg-brand-600", "text-white");
            navBtn.classList.add("text-slate-400");
        }
    });

    document.getElementById(`page-${tabId}`).classList.remove("hidden");
    const activeBtn = document.getElementById(`nav-${tabId}`);
    if (activeBtn) {
        activeBtn.classList.add("bg-brand-600", "text-white");
        activeBtn.classList.remove("text-slate-400");
    }

    setTimeout(() => lucide.createIcons(), 50);
}

async function loadDashboardData() {
    const symbol = document.getElementById("dashSymbol").value;
    try {
        const res = await fetch(`${API_BASE}/api/dashboard/data?symbol=${symbol}`);
        if (!res.ok) throw new Error("Failed to load data");
        const data = await res.json();

        document.getElementById("dashClose").innerText = data.latest.close.toFixed(2);
        const changeEl = document.getElementById("dashChange");
        const changePctEl = document.getElementById("dashChangePct");
        changeEl.innerText = `${data.latest.change >= 0 ? "+" : ""}${data.latest.change}`;
        changeEl.className = `text-2xl font-bold mt-1 ${data.latest.change >= 0 ? "text-brand-600" : "text-red-600"}`;
        changePctEl.innerText = `${data.latest.change_pct >= 0 ? "+" : ""}${data.latest.change_pct}%`;
        changePctEl.className = `text-2xl font-bold mt-1 ${data.latest.change_pct >= 0 ? "text-brand-600" : "text-red-600"}`;
        document.getElementById("dashVolume").innerText = data.latest.volume.toLocaleString();

        renderHistoricalChart(data);
    } catch (err) {
        showToast(err.message, "error");
    }
}

function renderHistoricalChart(data) {
    const ctx = document.getElementById("historicalChart").getContext("2d");
    if (historicalChart) historicalChart.destroy();

    historicalChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: "Close Price",
                    data: data.close,
                    borderColor: "#16a34a",
                    backgroundColor: "rgba(22, 163, 74, 0.05)",
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "#1e293b",
                    titleColor: "#f8fafc",
                    bodyColor: "#cbd5e1",
                    borderColor: "#334155",
                    borderWidth: 1,
                    padding: 10,
                    displayColors: false,
                },
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { maxTicksLimit: 8, color: "#64748b", font: { size: 10 } },
                },
                y: {
                    grid: { color: "#f1f5f9" },
                    ticks: { color: "#64748b", font: { size: 10 } },
                },
            },
        },
    });
}

async function loadAvailableSymbols() {
    try {
        const res = await fetch(`${API_BASE}/api/dashboard/symbols`);
        const data = await res.json();
        const dashSelect = document.getElementById("dashSymbol");
        const predSelect = document.getElementById("predSymbol");
        const currentDash = dashSelect.value;
        const currentPred = predSelect.value;

        dashSelect.innerHTML = "";
        predSelect.innerHTML = "";

        data.symbols.forEach((sym) => {
            const opt1 = document.createElement("option");
            opt1.value = sym;
            opt1.textContent = sym;
            dashSelect.appendChild(opt1);

            const opt2 = document.createElement("option");
            opt2.value = sym;
            opt2.textContent = sym;
            predSelect.appendChild(opt2);
        });

        if (data.symbols.includes(currentDash)) dashSelect.value = currentDash;
        if (data.symbols.includes(currentPred)) predSelect.value = currentPred;
    } catch (err) {
        console.error("Failed to load symbols", err);
    }
}

async function runPrediction() {
    const symbol = document.getElementById("predSymbol").value;
    const model = document.getElementById("predModel").value;
    const resultDiv = document.getElementById("predictionResult");

    try {
        const res = await fetch(`${API_BASE}/api/prediction/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ symbol, model }),
        });

        if (!res.ok) {
            const data = await res.json();
            throw new Error(data.detail || "Prediction failed");
        }

        const data = await res.json();
        resultDiv.classList.remove("hidden");
        resultDiv.classList.add("animate-fade-in");

        document.getElementById("predLastClose").innerText = data.last_close.toFixed(2);
        const returnEl = document.getElementById("predReturn");
        returnEl.innerText = `${data.predicted_return >= 0 ? "+" : ""}${(data.predicted_return * 100).toFixed(2)}%`;
        returnEl.className = `text-xl font-bold mt-1 ${data.predicted_return >= 0 ? "text-brand-600" : "text-red-600"}`;
        document.getElementById("predPrice").innerText = data.predicted_price.toFixed(2);

        showToast("Prediction completed", "success");
        loadChartData();
    } catch (err) {
        showToast(err.message, "error");
    }
}

async function loadChartData() {
    const symbol = document.getElementById("predSymbol").value;
    const model = document.getElementById("predModel").value;
    const ctx = document.getElementById("predictionChart").getContext("2d");

    try {
        const res = await fetch(`${API_BASE}/api/prediction/chart?symbol=${symbol}&model=${model}`);
        if (!res.ok) {
            const data = await res.json();
            throw new Error(data.detail || "Failed to load chart");
        }
        const data = await res.json();

        if (predictionChart) predictionChart.destroy();

        predictionChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: data.dates,
                datasets: [
                    {
                        label: "Actual Price",
                        data: data.actual,
                        borderColor: "#0f172a",
                        backgroundColor: "rgba(15, 23, 42, 0.05)",
                        fill: true,
                        tension: 0.1,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                    },
                    {
                        label: `${model.toUpperCase()} Prediction`,
                        data: data.predicted,
                        borderColor: "#22c55e",
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.1,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: "index", intersect: false },
                plugins: {
                    legend: {
                        position: "top",
                        labels: { usePointStyle: true, pointStyle: "line", padding: 20, color: "#475569" },
                    },
                    tooltip: {
                        backgroundColor: "#1e293b",
                        titleColor: "#f8fafc",
                        bodyColor: "#cbd5e1",
                        borderColor: "#334155",
                        borderWidth: 1,
                        padding: 10,
                    },
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { maxTicksLimit: 8, color: "#64748b", font: { size: 10 } },
                    },
                    y: {
                        grid: { color: "#f1f5f9" },
                        ticks: { color: "#64748b", font: { size: 10 } },
                    },
                },
            },
        });
    } catch (err) {
        showToast(err.message, "error");
    }
}

function downloadPlot() {
    const canvas = document.getElementById("predictionChart");
    const link = document.createElement("a");
    link.download = `prediction_${document.getElementById("predSymbol").value}_${Date.now()}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
    showToast("Plot downloaded", "success");
}

function sharePlot() {
    const canvas = document.getElementById("predictionChart");
    canvas.toBlob((blob) => {
        const file = new File([blob], "prediction_plot.png", { type: "image/png" });
        if (navigator.share) {
            navigator.share({
                title: "NEPSE Prediction Plot",
                text: `Stock prediction for ${document.getElementById("predSymbol").value}`,
                files: [file],
            }).catch(() => {});
        } else {
            navigator.clipboard.writeText("Check out this prediction plot from NEPSE Neural Analytics!");
            showToast("Link copied to clipboard", "info");
        }
    });
}

async function runScrape() {
    const btn = document.getElementById("btnScrape");
    const statusEl = document.getElementById("scrapeStatus");
    btn.classList.add("btn-loading");
    btn.disabled = true;
    statusEl.classList.remove("hidden");
    statusEl.innerHTML = '<span class="badge badge-running">Running</span> Scraping data... this may take a few minutes.';

    try {
        const res = await fetch(`${API_BASE}/api/admin/scrape`, { method: "POST" });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Scrape failed");

        pollTask(data.task_id, "scrape", statusEl);
    } catch (err) {
        statusEl.innerHTML = `<span class="badge badge-failed">Failed</span> ${err.message}`;
        btn.classList.remove("btn-loading");
        btn.disabled = false;
    }
}

async function runTrain(model) {
    const statusEl = document.getElementById("trainStatus");
    statusEl.classList.remove("hidden");
    statusEl.innerHTML = `<span class="badge badge-running">Running</span> Training ${model.toUpperCase()}... this may take several minutes.`;

    try {
        const res = await fetch(`${API_BASE}/api/admin/train?model=${model}`, { method: "POST" });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Training failed");

        pollTask(data.task_id, "train", statusEl, model);
    } catch (err) {
        statusEl.innerHTML = `<span class="badge badge-failed">Failed</span> ${err.message}`;
    }
}

async function pollTask(taskId, type, statusEl, model) {
    const interval = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/api/admin/tasks/${taskId}`);
            const task = await res.json();
            const status = task.status;

            if (status === "completed") {
                clearInterval(interval);
                statusEl.innerHTML = `<span class="badge badge-success">Completed</span> ${type === "scrape" ? "Data synced successfully" : model.toUpperCase() + " trained successfully"}`;
                showToast(`${type === "scrape" ? "Scrape" : "Training"} completed`, "success");
                refreshAdminPanel();
            } else if (status === "failed") {
                clearInterval(interval);
                statusEl.innerHTML = `<span class="badge badge-failed">Failed</span> ${task.error || "Unknown error"}`;
                showToast(`${type} failed`, "error");
            }
        } catch (err) {
            clearInterval(interval);
            statusEl.innerHTML = `<span class="badge badge-failed">Error</span> ${err.message}`;
        }
    }, 2000);
}

async function refreshAdminPanel() {
    await loadModelsList();
    await loadComparisonPlot();
    await loadAvailableSymbols();
}

async function loadModelsList() {
    try {
        const res = await fetch(`${API_BASE}/api/admin/models`);
        const data = await res.json();
        const container = document.getElementById("modelsList");

        if (data.models.length === 0) {
            container.innerHTML = '<p class="text-sm text-slate-400 col-span-full">No trained models found. Train a model to see it here.</p>';
            return;
        }

        container.innerHTML = data.models
            .map(
                (m) => `
            <div class="bg-slate-50 p-4 rounded-lg border border-slate-200 flex items-center space-x-3">
                <div class="p-2 bg-brand-100 text-brand-700 rounded-lg">
                    <i data-lucide="cpu" class="w-5 h-5"></i>
                </div>
                <div>
                    <p class="text-sm font-semibold text-slate-800">${m}</p>
                    <p class="text-xs text-slate-500">Ready for inference</p>
                </div>
            </div>
        `
            )
            .join("");
        lucide.createIcons();
    } catch (err) {
        console.error("Failed to load models", err);
    }
}

async function loadComparisonPlot() {
    const img = document.getElementById("comparisonPlot");
    const placeholder = document.getElementById("plotPlaceholder");
    try {
        const res = await fetch(`${API_BASE}/api/prediction/plot`);
        if (res.ok) {
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            img.src = url;
            img.classList.remove("hidden");
            placeholder.classList.add("hidden");
        } else {
            img.classList.add("hidden");
            placeholder.classList.remove("hidden");
        }
    } catch (err) {
        img.classList.add("hidden");
        placeholder.classList.remove("hidden");
    }
}

function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.innerText = message;
    document.body.appendChild(toast);

    requestAnimationFrame(() => {
        toast.classList.add("show");
    });

    setTimeout(() => {
        toast.classList.remove("show");
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

setTimeout(loadAvailableSymbols, 100);
