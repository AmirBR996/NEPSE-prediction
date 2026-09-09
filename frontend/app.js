(() => {
  const state = {
    token: localStorage.getItem('jwt_token') || null,
    baseUrl: localStorage.getItem('api_base_url') || 'http://localhost:8000',
    user: null,
    charts: {},
    pollTimers: {},
  };

  document.addEventListener('DOMContentLoaded', () => {
    if (window.lucide) lucide.createIcons();
    ensureToastHost();
    bindBaseUrl();
    bindAuthForms();
    bindAuthTabs();
    checkCurrentUser().then(() => {
      applyNavGuards();
      bootPage();
    });
  });

  function ensureToastHost() {
    if (!document.getElementById('toastContainer')) {
      const el = document.createElement('div');
      el.id = 'toastContainer';
      el.className = 'toast-container';
      document.body.appendChild(el);
    }
  }

  function toast(message, type = 'info') {
    const host = document.getElementById('toastContainer');
    if (!host) return alert(message);
    const el = document.createElement('div');
    el.className = `toast ${type === 'error' ? 'error' : type === 'success' ? 'success' : ''}`;
    el.textContent = message;
    host.appendChild(el);
    setTimeout(() => el.remove(), 3500);
  }

  function bindBaseUrl() {
    const input = document.getElementById('baseUrlInput');
    if (!input) return;
    input.value = state.baseUrl;
    input.addEventListener('change', (e) => {
      state.baseUrl = e.target.value.replace(/\/$/, '');
      localStorage.setItem('api_base_url', state.baseUrl);
      toast('API base URL saved', 'success');
    });
  }

  async function apiFetch(endpoint, options = {}) {
    const url = `${state.baseUrl}${endpoint}`;
    const headers = {
      'Content-Type': 'application/json',
      ...(state.token ? { Authorization: `Bearer ${state.token}` } : {}),
      ...(options.headers || {}),
    };
    const response = await fetch(url, { ...options, headers });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map((d) => d.msg || JSON.stringify(d)).join(', ')
        : data.detail || data.message || `Error ${response.status}`;
      const err = new Error(detail);
      err.status = response.status;
      throw err;
    }
    return data;
  }

  function destroyChart(key) {
    if (state.charts[key]) {
      state.charts[key].destroy();
      delete state.charts[key];
    }
  }

  function makeChart(key, canvas, config) {
    if (!window.Chart || !canvas) return null;
    destroyChart(key);
    state.charts[key] = new Chart(canvas.getContext('2d'), config);
    return state.charts[key];
  }

  function formatRs(n) {
    if (n == null || Number.isNaN(n)) return '—';
    return `Rs. ${Number(n).toFixed(2)}`;
  }

  function formatPct(n) {
    if (n == null || Number.isNaN(n)) return '—';
    const sign = n > 0 ? '+' : '';
    return `${sign}${Number(n).toFixed(2)}%`;
  }

  function statusBadge(status) {
    const s = (status || '').toLowerCase();
    let cls = 'badge-gray';
    if (s.includes('ready') || s.includes('evaluated') || s === 'trained' || s === 'completed') cls = 'badge-green';
    else if (s.includes('training') || s.includes('pending') || s === 'queued') cls = 'badge-yellow';
    else if (s.includes('fail')) cls = 'badge-red';
    else if (s.includes('partial')) cls = 'badge-blue';
    return `<span class="badge ${cls}">${status || 'Unknown'}</span>`;
  }

  async function checkCurrentUser() {
    const nameEl = document.getElementById('userName');
    const roleEl = document.getElementById('userRole');
    const avatarEl = document.getElementById('userAvatar');
    const statusTextEl = document.getElementById('authStatusText');
    const authLink = document.getElementById('authPageLink');

    if (!state.token) {
      state.user = null;
      if (statusTextEl) statusTextEl.innerText = 'Unauthenticated';
      return null;
    }

    try {
      const user = await apiFetch('/auth/me');
      state.user = user;
      if (nameEl) nameEl.innerText = user.username;
      if (roleEl) roleEl.innerText = user.role + (user.is_authorized ? '' : ' · Pending');
      if (avatarEl) avatarEl.innerText = (user.username || 'U')[0].toUpperCase();
      if (statusTextEl) statusTextEl.innerText = 'Authenticated';
      if (authLink) {
        authLink.title = 'Logout';
        authLink.innerHTML = '<i data-lucide="log-out"></i>';
        authLink.href = '#';
        authLink.onclick = async (e) => {
          e.preventDefault();
          try { await apiFetch('/auth/logout', { method: 'POST' }); } catch (_) {}
          localStorage.removeItem('jwt_token');
          state.token = null;
          state.user = null;
          window.location.href = 'index.html';
        };
        if (window.lucide) lucide.createIcons();
      }
      return user;
    } catch (_) {
      localStorage.removeItem('jwt_token');
      state.token = null;
      state.user = null;
      if (statusTextEl) statusTextEl.innerText = 'Session Expired';
      return null;
    }
  }

  function applyNavGuards() {
    const page = document.body.dataset.page;
    const user = state.user;

    document.querySelectorAll('[data-auth="user"]').forEach((el) => {
      el.classList.toggle('hidden', !user || user.role === 'ADMIN');
    });
    document.querySelectorAll('[data-auth="guest"]').forEach((el) => {
      el.classList.toggle('hidden', !!user);
    });
    document.querySelectorAll('[data-auth="admin"]').forEach((el) => {
      el.classList.toggle('hidden', !(user && user.role === 'ADMIN'));
    });
    document.querySelectorAll('[data-auth="authed"]').forEach((el) => {
      el.classList.toggle('hidden', !user);
    });

    const predictLink = document.getElementById('navPredict');
    if (predictLink && user && user.role === 'USER' && !user.is_authorized) {
      predictLink.innerHTML = '<i data-lucide="sparkles"></i> Predict <span class="lock-badge">🔒</span>';
      if (window.lucide) lucide.createIcons();
    }

    if (page === 'admin' && (!user || user.role !== 'ADMIN')) {
      toast('Admin access required', 'error');
      window.location.href = user ? 'index.html' : 'auth.html';
      return;
    }
    if (page === 'predict' && (!user || user.role !== 'USER')) {
      if (user && user.role === 'ADMIN') {
        window.location.href = 'admin.html';
        return;
      }
      toast('Please log in as a user to access predictions', 'error');
      window.location.href = 'auth.html';
    }
  }

  function bindAuthTabs() {
    const tabBtnLogin = document.getElementById('tabBtnLogin');
    const tabBtnRegister = document.getElementById('tabBtnRegister');
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');
    if (!tabBtnLogin || !tabBtnRegister) return;
    tabBtnLogin.addEventListener('click', () => {
      tabBtnLogin.classList.add('active');
      tabBtnRegister.classList.remove('active');
      loginForm.classList.add('active-form');
      registerForm.classList.remove('active-form');
    });
    tabBtnRegister.addEventListener('click', () => {
      tabBtnRegister.classList.add('active');
      tabBtnLogin.classList.remove('active');
      registerForm.classList.add('active-form');
      loginForm.classList.remove('active-form');
    });
  }

  function bindAuthForms() {
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');

    if (loginForm) {
      loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = loginForm.querySelector('button[type="submit"]');
        btn.disabled = true;
        try {
          const data = await apiFetch('/auth/login', {
            method: 'POST',
            body: JSON.stringify({
              username: document.getElementById('loginUsername').value,
              password: document.getElementById('loginPassword').value,
            }),
          });
          state.token = data.access_token;
          localStorage.setItem('jwt_token', state.token);
          toast('Login successful', 'success');
          window.location.href = data.role === 'ADMIN' ? 'admin.html' : 'index.html';
        } catch (err) {
          toast(`Login failed: ${err.message}`, 'error');
        } finally {
          btn.disabled = false;
        }
      });
    }

    if (registerForm) {
      registerForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const password = document.getElementById('regPassword').value;
        const confirm = document.getElementById('regConfirmPassword')?.value;
        if (confirm != null && password !== confirm) {
          toast('Passwords do not match', 'error');
          return;
        }
        const btn = registerForm.querySelector('button[type="submit"]');
        btn.disabled = true;
        try {
          await apiFetch('/auth/register', {
            method: 'POST',
            body: JSON.stringify({
              email: document.getElementById('regEmail').value,
              username: document.getElementById('regUsername').value,
              password,
              confirm_password: confirm,
            }),
          });
          toast('Registration successful. Please log in.', 'success');
          document.getElementById('tabBtnLogin')?.click();
        } catch (err) {
          toast(`Registration failed: ${err.message}`, 'error');
        } finally {
          btn.disabled = false;
        }
      });
    }
  }

  async function bootPage() {
    const page = document.body.dataset.page;
    if (page === 'dashboard') await initDashboard();
    if (page === 'predict') await initPredict();
    if (page === 'admin') await initAdmin();
  }

  // -------------------- Dashboard --------------------
  async function initDashboard() {
    const banner = document.getElementById('authPendingBanner');
    if (banner && state.user && state.user.role === 'USER' && !state.user.is_authorized) {
      banner.classList.remove('hidden');
    }

    const select = document.getElementById('companySelect');
    if (!select) return;

    try {
      const res = await apiFetch('/companies');
      const companies = res.companies || [];
      select.innerHTML = companies
        .map((c) => `<option value="${c.symbol}">${c.symbol} — ${c.name}</option>`)
        .join('');
      if (companies.length) {
        select.value = companies.find((c) => c.symbol === 'NABIL')?.symbol || companies[0].symbol;
        await loadCompanyDashboard(select.value);
      }
    } catch (err) {
      toast(`Failed to load companies: ${err.message}`, 'error');
    }

    select.addEventListener('change', () => loadCompanyDashboard(select.value));

    document.querySelectorAll('.range-pill').forEach((btn) => {
      btn.addEventListener('click', async () => {
        document.querySelectorAll('.range-pill').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        await loadHistory(select.value, btn.dataset.range);
      });
    });

    // Probe API
    try {
      await apiFetch('/');
      const apiStatus = document.getElementById('apiStatusText');
      if (apiStatus) apiStatus.textContent = 'Connected';
      document.querySelector('.status-indicator')?.classList.add('online');
    } catch (_) {
      const apiStatus = document.getElementById('apiStatusText');
      if (apiStatus) apiStatus.textContent = 'Offline';
    }
  }

  async function loadCompanyDashboard(symbol) {
    const overviewEl = document.getElementById('companyOverview');
    const forecastEl = document.getElementById('forecastCards');
    if (overviewEl) overviewEl.innerHTML = '<div class="skeleton" style="height:80px"></div>';
    if (forecastEl) forecastEl.innerHTML = '<div class="empty-state">Loading forecasts…</div>';

    try {
      const [overview, forecast] = await Promise.all([
        apiFetch(`/companies/${encodeURIComponent(symbol)}`),
        apiFetch(`/companies/${encodeURIComponent(symbol)}/forecast`),
      ]);
      renderOverview(overview);
      renderForecastCards(forecast);
      const range = document.querySelector('.range-pill.active')?.dataset.range || '1Y';
      await loadHistory(symbol, range);
      renderComparisonChart(forecast);
    } catch (err) {
      toast(err.message, 'error');
    }
  }

  function renderOverview(c) {
    const el = document.getElementById('companyOverview');
    if (!el) return;
    const change = c.percentage_change;
    const changeCls = change > 0 ? 'up' : change < 0 ? 'down' : '';
    el.innerHTML = `
      <div class="overview-grid">
        <div class="stat-box"><div class="label">Company</div><div class="value">${c.name || c.symbol}</div></div>
        <div class="stat-box"><div class="label">Symbol</div><div class="value">${c.symbol}</div></div>
        <div class="stat-box"><div class="label">Latest Price</div><div class="value">${formatRs(c.latest_price)}</div></div>
        <div class="stat-box"><div class="label">Previous</div><div class="value">${formatRs(c.previous_price)}</div></div>
        <div class="stat-box"><div class="label">Change</div><div class="value ${changeCls}">${formatPct(change)}</div></div>
        <div class="stat-box"><div class="label">Volume</div><div class="value">${c.trading_volume != null ? Number(c.trading_volume).toLocaleString() : '—'}</div></div>
        <div class="stat-box"><div class="label">Last Updated</div><div class="value" style="font-size:0.95rem">${c.last_updated || '—'}</div></div>
        <div class="stat-box"><div class="label">Training Status</div><div class="value" style="font-size:0.95rem">${statusBadge(c.training_status)}</div></div>
      </div>`;
  }

  function renderForecastCards(forecast) {
    const el = document.getElementById('forecastCards');
    if (!el) return;
    const items = forecast.forecasts || [];
    const current = forecast.latest_price;
    if (!items.length) {
      el.innerHTML = '<div class="empty-state">No forecasting results available yet.</div>';
      return;
    }
      el.innerHTML = items.map((item) => {
      const m = item.metrics;
      const has = !!m;
      const prod = item.is_production ? '<span class="badge badge-green">Production</span>' : '';
      const prodReady = item.has_production_weights
        ? '<span class="badge badge-blue">Prod weights</span>'
        : '';
      if (!has) {
        return `<div class="model-card"><h4>${item.model.toUpperCase()} ${prod}</h4>
          <p class="section-desc">${item.has_evaluation_weights || item.has_weights ? 'Evaluation trained — run Evaluate' : 'Not trained (evaluation)'}</p>
          <p class="section-desc">Test split from ${item.test_split_date || '2025-01-01'}</p></div>`;
      }
      return `<div class="model-card ${item.is_production ? 'production' : ''}">
        <h4>${item.model.toUpperCase()} ${prod} ${prodReady}</h4>
        <p><strong>Evaluation metrics</strong> (test from ${item.test_split_date || '2025'})</p>
        <p>Current: ${formatRs(current)}</p>
        <p>MAE: ${m.mae?.toFixed(2)} · RMSE: ${m.rmse?.toFixed(2)}</p>
        <p>R²: ${m.r2?.toFixed(3)} · Dir. Acc: ${m.directional_accuracy?.toFixed(2)}%</p>
        ${m.mape != null ? `<p>MAPE: ${m.mape.toFixed(2)}%</p>` : ''}
      </div>`;
    }).join('');
  }

  async function loadHistory(symbol, rangeKey) {
    try {
      const data = await apiFetch(`/companies/${encodeURIComponent(symbol)}/data?range=${encodeURIComponent(rangeKey)}`);
      const points = data.points || [];
      const labels = points.map((p) => p.date);
      const closes = points.map((p) => p.close);
      const volumes = points.map((p) => p.volume);
      const opens = points.map((p) => p.open);
      const highs = points.map((p) => p.high);
      const lows = points.map((p) => p.low);

      makeChart('price', document.getElementById('priceChart'), {
        type: 'line',
        data: {
          labels,
          datasets: [
            { label: 'Close', data: closes, borderColor: '#4f46e5', tension: 0.15, pointRadius: 0, borderWidth: 2 },
            { label: 'Open', data: opens, borderColor: '#94a3b8', tension: 0.15, pointRadius: 0, borderWidth: 1, hidden: true },
            { label: 'High', data: highs, borderColor: '#059669', tension: 0.15, pointRadius: 0, borderWidth: 1, hidden: true },
            { label: 'Low', data: lows, borderColor: '#dc2626', tension: 0.15, pointRadius: 0, borderWidth: 1, hidden: true },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { position: 'bottom' } },
          scales: { x: { ticks: { maxTicksLimit: 8 } } },
        },
      });

      makeChart('volume', document.getElementById('volumeChart'), {
        type: 'bar',
        data: {
          labels,
          datasets: [{ label: 'Volume', data: volumes, backgroundColor: 'rgba(79,70,229,0.35)' }],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: { x: { ticks: { maxTicksLimit: 8 } } },
        },
      });
    } catch (err) {
      toast(`History: ${err.message}`, 'error');
    }
  }

  function renderComparisonChart(forecast) {
    const canvas = document.getElementById('comparisonChart');
    if (!canvas) return;
    const items = (forecast.forecasts || []).filter((f) => f.series && f.series.dates?.length);
    if (!items.length) {
      destroyChart('comparison');
      const empty = document.getElementById('comparisonEmpty');
      if (empty) empty.classList.remove('hidden');
      return;
    }
    document.getElementById('comparisonEmpty')?.classList.add('hidden');

    const base = items[0].series;
    const colors = { lstm: '#f97316', gru: '#16a34a', transformer: '#dc2626' };
    const styles = { lstm: [6, 4], gru: [2, 3], transformer: [8, 3, 2, 3] };
    const datasets = [
      {
        label: 'Actual Price',
        data: base.actual,
        borderColor: '#2563eb',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1,
      },
    ];
    items.forEach((item) => {
      datasets.push({
        label: `${item.model.toUpperCase()} Prediction`,
        data: item.series.predicted,
        borderColor: colors[item.model] || '#64748b',
        borderDash: styles[item.model] || [],
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1,
      });
    });

    makeChart('comparison', canvas, {
      type: 'line',
      data: { labels: base.dates, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: `${forecast.symbol} Stock Price: Actual vs LSTM vs GRU vs Transformer`,
          },
          legend: { position: 'top' },
        },
        scales: {
          x: { title: { display: true, text: 'Date' }, ticks: { maxTicksLimit: 10 } },
          y: { title: { display: true, text: 'Stock Price' } },
        },
      },
    });
  }

  // -------------------- Predict --------------------
  async function initPredict() {
    const locked = document.getElementById('predictLocked');
    const panel = document.getElementById('predictPanel');
    if (!state.user) return;

    if (!state.user.is_authorized) {
      locked?.classList.remove('hidden');
      panel?.classList.add('hidden');
      return;
    }
    locked?.classList.add('hidden');
    panel?.classList.remove('hidden');

    const select = document.getElementById('predictCompany');
    const res = await apiFetch('/companies');
    select.innerHTML = (res.companies || [])
      .map((c) => `<option value="${c.symbol}">${c.symbol}</option>`)
      .join('');

    document.getElementById('predictForm')?.addEventListener('submit', async (e) => {
      e.preventDefault();
      const btn = e.target.querySelector('button[type="submit"]');
      btn.disabled = true;
      try {
        const model = document.querySelector('input[name="predictModel"]:checked')?.value || null;
        const body = {
          company: select.value,
          horizon_days: Number(document.getElementById('predictHorizon').value),
        };
        if (model && model !== 'production') body.model = model;
        const result = await apiFetch('/predictions', { method: 'POST', body: JSON.stringify(body) });
        document.getElementById('predictResult').innerHTML = `
          <div class="overview-grid">
            <div class="stat-box"><div class="label">Model</div><div class="value">${result.model}</div></div>
            <div class="stat-box"><div class="label">Current</div><div class="value">${formatRs(result.current_price)}</div></div>
            <div class="stat-box"><div class="label">Predicted</div><div class="value">${formatRs(result.predicted_price)}</div></div>
            <div class="stat-box"><div class="label">Change</div><div class="value ${result.predicted_change_pct >= 0 ? 'up' : 'down'}">${formatPct(result.predicted_change_pct)}</div></div>
          </div>
          <div class="table-wrap" style="margin-top:1rem">
            <table class="data-table"><thead><tr><th>Day</th><th>Predicted Price</th><th>Return</th></tr></thead>
            <tbody>${(result.predictions || []).map((p) => `<tr><td>${p.day}</td><td>${formatRs(p.predicted_price)}</td><td>${formatPct(p.predicted_return * 100)}</td></tr>`).join('')}</tbody></table>
          </div>`;
        toast('Prediction completed', 'success');
        loadUserPredictions();
      } catch (err) {
        toast(err.message, 'error');
      } finally {
        btn.disabled = false;
      }
    });

    document.getElementById('userTrainForm')?.addEventListener('submit', async (e) => {
      e.preventDefault();
      const btn = e.target.querySelector('button[type="submit"]');
      btn.disabled = true;
      try {
        const model = document.querySelector('input[name="trainModel"]:checked')?.value;
        const res = await apiFetch('/training/user', {
          method: 'POST',
          body: JSON.stringify({
            company: document.getElementById('trainCompany').value,
            model,
            epochs: Number(document.getElementById('trainEpochs').value || 50),
          }),
        });
        toast('Training queued', 'success');
        pollJob(res.job.id, 'userTrainStatus');
      } catch (err) {
        toast(err.message, 'error');
      } finally {
        btn.disabled = false;
      }
    });

    // populate train company
    const trainCompany = document.getElementById('trainCompany');
    if (trainCompany) {
      trainCompany.innerHTML = select.innerHTML;
    }

    loadUserPredictions();
  }

  async function loadUserPredictions() {
    const el = document.getElementById('predictionHistory');
    if (!el) return;
    try {
      const res = await apiFetch('/predictions');
      const rows = res.predictions || [];
      if (!rows.length) {
        el.innerHTML = '<div class="empty-state">No predictions yet.</div>';
        return;
      }
      el.innerHTML = `<div class="table-wrap"><table class="data-table">
        <thead><tr><th>ID</th><th>Company</th><th>Model</th><th>Horizon</th><th>Predicted</th><th>Change</th><th>When</th></tr></thead>
        <tbody>${rows.map((p) => `<tr>
          <td>${p.id}</td><td>${p.company}</td><td>${p.model_type}</td><td>${p.horizon_days}d</td>
          <td>${formatRs(p.predicted_price)}</td><td>${formatPct(p.predicted_change_pct)}</td>
          <td>${p.created_at ? new Date(p.created_at).toLocaleString() : '—'}</td>
        </tr>`).join('')}</tbody></table></div>`;
    } catch (err) {
      el.innerHTML = `<div class="empty-state">${err.message}</div>`;
    }
  }

  function pollJob(jobId, statusElId) {
    const el = document.getElementById(statusElId);
    if (state.pollTimers[jobId]) clearInterval(state.pollTimers[jobId]);
    const tick = async () => {
      try {
        const job = await apiFetch(`/training/${jobId}`);
        if (el) {
          el.innerHTML = `${statusBadge(job.status)} Epoch ${job.current_epoch || 0}/${job.total_epochs || '?'}
            · loss ${job.train_loss != null ? job.train_loss.toFixed(6) : '—'}
            · val ${job.validation_loss != null ? job.validation_loss.toFixed(6) : '—'}
            ${job.error ? `<br><span style="color:#dc2626">${job.error}</span>` : ''}`;
        }
        if (['completed', 'failed'].includes(job.status)) {
          clearInterval(state.pollTimers[jobId]);
          toast(`Training ${job.status}`, job.status === 'completed' ? 'success' : 'error');
        }
      } catch (err) {
        clearInterval(state.pollTimers[jobId]);
        if (el) el.textContent = err.message;
      }
    };
    tick();
    state.pollTimers[jobId] = setInterval(tick, 2500);
  }

  // -------------------- Admin --------------------
  async function initAdmin() {
    document.querySelectorAll('[data-admin-tab]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const tab = btn.dataset.adminTab;
        document.querySelectorAll('[data-admin-tab]').forEach((b) => b.classList.remove('active'));
        document.querySelectorAll('.admin-tab').forEach((s) => s.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(`admin-${tab}`)?.classList.add('active');
        const title = document.getElementById('pageTitle');
        if (title) title.textContent = btn.textContent.trim();
        if (tab === 'dashboard') loadAdminStats();
        if (tab === 'users') loadAdminUsers();
        if (tab === 'companies') loadAdminCompanies();
        if (tab === 'models') loadAdminModels();
        if (tab === 'training') loadAdminTraining();
        if (tab === 'evaluation') initAdminEvaluation();
        if (tab === 'system') loadSystemStatus();
      });
    });
    await loadAdminStats();
    populateCompanySelects();
  }

  async function populateCompanySelects() {
    try {
      const res = await apiFetch('/companies');
      const opts = (res.companies || []).map((c) => `<option value="${c.symbol}">${c.symbol}</option>`).join('');
      document.querySelectorAll('[data-company-select]').forEach((el) => {
        el.innerHTML = opts;
      });
    } catch (_) {}
  }

  async function loadAdminStats() {
    try {
      const s = await apiFetch('/admin/stats');
      const el = document.getElementById('adminStats');
      if (!el) return;
      el.innerHTML = `
        <div class="card metric"><div><p class="metric-label">Users</p><p>${s.users.total}</p></div></div>
        <div class="card metric"><div><p class="metric-label">Authorized</p><p>${s.users.authorized}</p></div></div>
        <div class="card metric"><div><p class="metric-label">Pending</p><p>${s.users.pending}</p></div></div>
        <div class="card metric"><div><p class="metric-label">Companies</p><p>${s.companies.total}</p></div></div>
        <div class="card metric"><div><p class="metric-label">Trained</p><p>${s.companies.trained}</p></div></div>
        <div class="card metric"><div><p class="metric-label">Untrained</p><p>${s.companies.untrained}</p></div></div>
        <div class="card metric"><div><p class="metric-label">Active Models</p><p>${s.active_models}</p></div></div>
        <div class="card metric"><div><p class="metric-label">Failed Jobs</p><p>${s.failed_training_jobs}</p></div></div>`;
    } catch (err) {
      toast(err.message, 'error');
    }
  }

  async function loadAdminUsers() {
    const el = document.getElementById('usersTable');
    try {
      const res = await apiFetch('/admin/users');
      const users = res.users || [];
      el.innerHTML = `<div class="table-wrap"><table class="data-table">
        <thead><tr><th>ID</th><th>Username</th><th>Email</th><th>Role</th><th>Auth</th><th>Active</th><th>Created</th><th>Actions</th></tr></thead>
        <tbody>${users.map((u) => `<tr>
          <td>${u.id}</td><td>${u.username}</td><td>${u.email}</td><td>${u.role}</td>
          <td>${u.is_authorized ? statusBadge('Authorized') : statusBadge('Pending')}</td>
          <td>${u.is_active ? statusBadge('Active') : statusBadge('Inactive')}</td>
          <td>${u.created_at ? new Date(u.created_at).toLocaleDateString() : '—'}</td>
          <td class="action-row">
            ${u.role !== 'ADMIN' ? `<button class="btn btn-sm btn-accent" data-auth-user="${u.id}" data-val="${!u.is_authorized}">${u.is_authorized ? 'Revoke' : 'Authorize'}</button>` : ''}
            <button class="btn btn-sm btn-secondary" data-status-user="${u.id}" data-val="${!u.is_active}">${u.is_active ? 'Deactivate' : 'Activate'}</button>
            <button class="btn btn-sm btn-danger" data-del-user="${u.id}" data-role="${u.role}">Delete</button>
          </td>
        </tr>`).join('')}</tbody></table></div>`;

      el.querySelectorAll('[data-auth-user]').forEach((btn) => {
        btn.addEventListener('click', async () => {
          try {
            await apiFetch(`/admin/users/${btn.dataset.authUser}/authorize`, {
              method: 'PATCH',
              body: JSON.stringify({ is_authorized: btn.dataset.val === 'true' }),
            });
            toast('Authorization updated', 'success');
            loadAdminUsers();
            loadAdminStats();
          } catch (err) { toast(err.message, 'error'); }
        });
      });
      el.querySelectorAll('[data-status-user]').forEach((btn) => {
        btn.addEventListener('click', async () => {
          try {
            await apiFetch(`/admin/users/${btn.dataset.statusUser}/status`, {
              method: 'PATCH',
              body: JSON.stringify({ is_active: btn.dataset.val === 'true' }),
            });
            toast('Status updated', 'success');
            loadAdminUsers();
          } catch (err) { toast(err.message, 'error'); }
        });
      });
      el.querySelectorAll('[data-del-user]').forEach((btn) => {
        btn.addEventListener('click', async () => {
          const isAdmin = btn.dataset.role === 'ADMIN';
          if (!confirm(isAdmin ? 'Delete this ADMIN account? This requires confirmation.' : 'Delete this user?')) return;
          try {
            await apiFetch(`/admin/users/${btn.dataset.delUser}?confirm=${isAdmin}`, { method: 'DELETE' });
            toast('User deleted', 'success');
            loadAdminUsers();
            loadAdminStats();
          } catch (err) { toast(err.message, 'error'); }
        });
      });
    } catch (err) {
      el.innerHTML = `<div class="empty-state">${err.message}</div>`;
    }
  }

  async function loadAdminCompanies() {
    const el = document.getElementById('companiesTable');
    const filter = document.getElementById('companyFilter')?.value || 'all';
    const q = document.getElementById('companySearch')?.value || '';
    try {
      const res = await apiFetch(`/admin/companies?status=${encodeURIComponent(filter)}&q=${encodeURIComponent(q)}`);
      const rows = res.companies || [];
      el.innerHTML = `<div class="table-wrap"><table class="data-table">
        <thead><tr><th>Company</th><th>LSTM</th><th>GRU</th><th>Transformer</th><th>Production</th><th>Status</th><th>Actions</th></tr></thead>
        <tbody>${rows.map((c) => {
          const mark = (m) => {
            const ev = c.models?.[m]?.has_evaluation_weights ? 'E' : '·';
            const pr = c.models?.[m]?.has_production_weights ? 'P' : '·';
            return `${ev}/${pr}`;
          };
          return `<tr>
            <td><strong>${c.symbol}</strong><br><small>${c.name || ''}</small></td>
            <td>${mark('lstm')}</td><td>${mark('gru')}</td><td>${mark('transformer')}</td>
            <td>${c.production_model || 'None'}</td>
            <td>${statusBadge(c.training_status)}</td>
            <td class="action-row">
              <button class="btn btn-sm btn-primary" data-train-co="${c.symbol}">Train Eval LSTM</button>
              <button class="btn btn-sm btn-secondary" data-eval-co="${c.symbol}">Evaluate</button>
            </td>
          </tr>`;
        }).join('')}</tbody></table></div>
        <p class="section-desc" style="margin-top:0.75rem">E = evaluation weights (2025 hold-out) · P = production weights (all data)</p>`;

      el.querySelectorAll('[data-train-co]').forEach((btn) => {
        btn.addEventListener('click', async () => {
          btn.disabled = true;
          try {
            const res = await apiFetch('/admin/training', {
              method: 'POST',
              body: JSON.stringify({ company: btn.dataset.trainCo, model: 'lstm', epochs: 100 }),
            });
            toast('Training queued', 'success');
            pollJob(res.job.id, 'adminJobStatus');
          } catch (err) { toast(err.message, 'error'); }
          finally { btn.disabled = false; }
        });
      });
      el.querySelectorAll('[data-eval-co]').forEach((btn) => {
        btn.addEventListener('click', async () => {
          btn.disabled = true;
          try {
            await apiFetch(`/admin/companies/${btn.dataset.evalCo}/evaluate`, { method: 'POST' });
            toast('Evaluation completed', 'success');
            loadAdminCompanies();
          } catch (err) { toast(err.message, 'error'); }
          finally { btn.disabled = false; }
        });
      });
    } catch (err) {
      el.innerHTML = `<div class="empty-state">${err.message}</div>`;
    }
  }

  document.getElementById('companyFilter')?.addEventListener('change', () => loadAdminCompanies());
  document.getElementById('companySearch')?.addEventListener('input', () => {
    clearTimeout(state._searchTimer);
    state._searchTimer = setTimeout(loadAdminCompanies, 300);
  });

  async function loadAdminModels() {
    const el = document.getElementById('modelsTable');
    try {
      const res = await apiFetch('/admin/models');
      el.innerHTML = (res.models || []).map((row) => `
        <div class="card" style="margin-bottom:1rem">
          <h3>${row.company} <small style="color:var(--text-muted)">${row.name || ''}</small>
            ${statusBadge(row.status)} Production: <strong>${row.production_model || 'None'}</strong></h3>
          <div class="model-cards" style="margin-top:1rem">
            ${['lstm','gru','transformer'].map((m) => {
              const info = row.models?.[m] || {};
              return `<div class="model-card ${info.is_production ? 'production' : ''}">
                <h4>${m.toUpperCase()} ${info.is_production ? '<span class="badge badge-green">Selected</span>' : ''}</h4>
                <p>Evaluation weights: ${info.has_evaluation_weights ? 'Yes' : 'No'}</p>
                <p>Production weights (all data): ${info.has_production_weights ? 'Yes' : 'No'}</p>
                <p>Status: ${info.status || 'not_trained'}</p>
                ${info.has_evaluation_weights && !info.is_production ? `<button class="btn btn-sm btn-accent" data-set-prod="${row.company}" data-model="${m}">Set Production & Deploy</button>` : ''}
                ${info.is_production && !info.has_production_weights ? `<button class="btn btn-sm btn-primary" data-set-prod="${row.company}" data-model="${m}">Deploy Production Train</button>` : ''}
              </div>`;
            }).join('')}
          </div>
        </div>`).join('') || '<div class="empty-state">No models</div>';

      el.querySelectorAll('[data-set-prod]').forEach((btn) => {
        btn.addEventListener('click', async () => {
          try {
            await apiFetch(`/admin/companies/${btn.dataset.setProd}/production`, {
              method: 'POST',
              body: JSON.stringify({ model_type: btn.dataset.model }),
            });
            toast('Production selected; all-data deploy queued', 'success');
            loadAdminModels();
          } catch (err) { toast(err.message, 'error'); }
        });
      });
    } catch (err) {
      el.innerHTML = `<div class="empty-state">${err.message}</div>`;
    }
  }

  async function loadAdminTraining() {
    const form = document.getElementById('adminTrainForm');
    if (form && !form.dataset.bound) {
      form.dataset.bound = '1';
      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = form.querySelector('button[type="submit"]');
        btn.disabled = true;
        try {
          const res = await apiFetch('/admin/training', {
            method: 'POST',
            body: JSON.stringify({
              company: document.getElementById('adminTrainCompany').value,
              model: document.getElementById('adminTrainModel').value,
              epochs: Number(document.getElementById('adminTrainEpochs').value || 100),
            }),
          });
          toast('Training queued', 'success');
          pollJob(res.job.id, 'adminJobStatus');
          loadAdminJobs();
        } catch (err) { toast(err.message, 'error'); }
        finally { btn.disabled = false; }
      });
    }
    loadAdminJobs();
  }

  async function loadAdminJobs() {
    const el = document.getElementById('jobsTable');
    if (!el) return;
    try {
      const res = await apiFetch('/admin/training/jobs');
      const jobs = res.jobs || [];
      if (!jobs.length) {
        el.innerHTML = '<div class="empty-state">No training jobs yet.</div>';
        return;
      }
      el.innerHTML = `<div class="table-wrap"><table class="data-table">
        <thead><tr><th>ID</th><th>Company</th><th>Model</th><th>Purpose</th><th>Status</th><th>Epoch</th><th>Loss</th><th>Val/Test Loss</th><th>Started</th></tr></thead>
        <tbody>${jobs.map((j) => `<tr>
          <td>${j.id}</td><td>${j.company}</td><td>${j.model_type}</td>
          <td>${j.purpose || (j.include_test ? 'evaluation' : 'production')}</td>
          <td>${statusBadge(j.status)}</td>
          <td>${j.current_epoch}/${j.total_epochs}</td>
          <td>${j.train_loss != null ? j.train_loss.toFixed(5) : '—'}</td>
          <td>${j.validation_loss != null ? j.validation_loss.toFixed(5) : '—'}</td>
          <td>${j.started_at ? new Date(j.started_at).toLocaleString() : '—'}</td>
        </tr>`).join('')}</tbody></table></div>`;
    } catch (err) {
      el.innerHTML = `<div class="empty-state">${err.message}</div>`;
    }
  }

  function initAdminEvaluation() {
    const btn = document.getElementById('btnRunEvaluation');
    const select = document.getElementById('evalCompany');
    if (btn && !btn.dataset.bound) {
      btn.dataset.bound = '1';
      btn.addEventListener('click', async () => {
        btn.disabled = true;
        try {
          await apiFetch(`/admin/companies/${select.value}/evaluate`, { method: 'POST' });
          toast('Evaluation completed', 'success');
          await loadEvaluation(select.value);
        } catch (err) { toast(err.message, 'error'); }
        finally { btn.disabled = false; }
      });
      select?.addEventListener('change', () => loadEvaluation(select.value));
    }
    if (select?.value) loadEvaluation(select.value);
  }

  async function loadEvaluation(symbol) {
    const tableEl = document.getElementById('evalTable');
    const chartCanvas = document.getElementById('adminComparisonChart');
    try {
      const res = await apiFetch(`/admin/evaluations/${encodeURIComponent(symbol)}`);
      const evals = res.evaluations || [];
      const metrics = evals.filter((e) => e.metrics);
      const best = {};
      ['mae', 'rmse', 'mape', 'r2', 'directional_accuracy'].forEach((k) => {
        if (!metrics.length) return;
        const higherBetter = k === 'r2' || k === 'directional_accuracy';
        const sorted = [...metrics].sort((a, b) =>
          higherBetter ? (b.metrics[k] ?? -Infinity) - (a.metrics[k] ?? -Infinity)
            : (a.metrics[k] ?? Infinity) - (b.metrics[k] ?? Infinity)
        );
        best[k] = sorted[0]?.model;
      });

      tableEl.innerHTML = `<div class="table-wrap"><table class="data-table">
        <thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>R²</th><th>Dir. Acc</th><th>Production</th></tr></thead>
        <tbody>${evals.map((e) => {
          const m = e.metrics || {};
          const cell = (k, fmt) => {
            const v = m[k];
            if (v == null) return '<td>—</td>';
            const cls = best[k] === e.model ? 'best' : '';
            return `<td class="${cls}">${fmt(v)}</td>`;
          };
          return `<tr>
            <td><strong>${e.model.toUpperCase()}</strong></td>
            ${cell('mae', (v) => v.toFixed(3))}
            ${cell('rmse', (v) => v.toFixed(3))}
            ${cell('mape', (v) => v.toFixed(2) + '%')}
            ${cell('r2', (v) => v.toFixed(3))}
            ${cell('directional_accuracy', (v) => v.toFixed(2) + '%')}
            <td>${e.is_production ? statusBadge('YES') : '—'}
              ${e.has_evaluation_weights && !e.is_production ? `<button class="btn btn-sm btn-accent" data-prod="${symbol}" data-model="${e.model}">Set Prod & Deploy</button>` : ''}
              ${e.is_production && !e.has_production_weights ? `<span class="badge badge-yellow">Deploying/pending</span>` : ''}
              ${e.has_production_weights ? '<span class="badge badge-blue">Prod ready</span>' : ''}
            </td>
          </tr>`;
        }).join('')}</tbody></table></div>`;

      tableEl.querySelectorAll('[data-prod]').forEach((btn) => {
        btn.addEventListener('click', async () => {
          try {
            await apiFetch(`/admin/companies/${btn.dataset.prod}/production`, {
              method: 'POST',
              body: JSON.stringify({ model_type: btn.dataset.model }),
            });
            toast('Production selected; all-data training queued', 'success');
            loadEvaluation(symbol);
          } catch (err) { toast(err.message, 'error'); }
        });
      });

      const withSeries = evals.filter((e) => e.series?.dates?.length);
      if (withSeries.length && chartCanvas) {
        document.getElementById('adminComparisonEmpty')?.classList.add('hidden');
        const base = withSeries[0].series;
        const colors = { lstm: '#f97316', gru: '#16a34a', transformer: '#dc2626' };
        const styles = { lstm: [6, 4], gru: [2, 3], transformer: [8, 3, 2, 3] };
        makeChart('adminComparison', chartCanvas, {
          type: 'line',
          data: {
            labels: base.dates,
            datasets: [
              { label: 'Actual Price', data: base.actual, borderColor: '#2563eb', borderWidth: 2, pointRadius: 0 },
              ...withSeries.map((e) => ({
                label: `${e.model.toUpperCase()} Prediction`,
                data: e.series.predicted,
                borderColor: colors[e.model],
                borderDash: styles[e.model],
                borderWidth: 2,
                pointRadius: 0,
              })),
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              title: { display: true, text: `${symbol} Stock Price: Actual vs LSTM vs GRU vs Transformer` },
              legend: { position: 'top' },
            },
            scales: {
              x: { title: { display: true, text: 'Date' }, ticks: { maxTicksLimit: 10 } },
              y: { title: { display: true, text: 'Stock Price' } },
            },
          },
        });

        // Metrics bar chart
        makeChart('adminMetrics', document.getElementById('metricsChart'), {
          type: 'bar',
          data: {
            labels: metrics.map((e) => e.model.toUpperCase()),
            datasets: [
              { label: 'MAE', data: metrics.map((e) => e.metrics.mae), backgroundColor: '#6366f1' },
              { label: 'RMSE', data: metrics.map((e) => e.metrics.rmse), backgroundColor: '#f59e0b' },
            ],
          },
          options: { responsive: true, maintainAspectRatio: false },
        });
      } else {
        document.getElementById('adminComparisonEmpty')?.classList.remove('hidden');
        destroyChart('adminComparison');
        destroyChart('adminMetrics');
      }
    } catch (err) {
      tableEl.innerHTML = `<div class="empty-state">${err.message}</div>`;
    }
  }

  async function loadSystemStatus() {
    const el = document.getElementById('systemStatus');
    try {
      const s = await apiFetch('/admin/system');
      el.innerHTML = `<div class="overview-grid">
        <div class="stat-box"><div class="label">API</div><div class="value">${s.api}</div></div>
        <div class="stat-box"><div class="label">Database</div><div class="value">${s.database}</div></div>
        <div class="stat-box"><div class="label">Active Jobs</div><div class="value">${s.active_training_jobs}</div></div>
        <div class="stat-box"><div class="label">Companies</div><div class="value">${s.companies}</div></div>
        <div class="stat-box"><div class="label">Models Tracked</div><div class="value">${s.models_tracked}</div></div>
        <div class="stat-box"><div class="label">Evaluations</div><div class="value">${s.evaluations}</div></div>
      </div>`;
    } catch (err) {
      el.innerHTML = `<div class="empty-state">${err.message}</div>`;
    }
  }

  // expose for inline handlers if needed
  window.StockPulse = { apiFetch, toast, loadAdminCompanies };
})();
