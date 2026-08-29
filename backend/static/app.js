let currentSamples = [];
let reviewQueueSamples = [];
let consoleLogs = [];
let selectedCardIndex = 0;
let activeTab = "tab-review";
let selectedSampleIds = new Set();

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initEventListeners();
  refreshAllData();

  // Polling interval to refresh metrics, review queue, and console logs every 2 seconds
  setInterval(() => {
    fetchConsoleLogs();
    if (activeTab === "tab-review") {
      fetchMetrics();
    }
  }, 2000);
});

// ----------------------------------------------------------------------------
// Tab Navigation
// ----------------------------------------------------------------------------

function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));

      tab.classList.add("active");
      activeTab = tab.dataset.tab;
      const targetPanel = document.getElementById(activeTab);
      if (targetPanel) {
        targetPanel.classList.add("active");
      }

      if (activeTab === "tab-review") {
        fetchReviewQueue();
      } else if (activeTab === "tab-samples") {
        fetchDatasetSamples();
      } else if (activeTab === "tab-model") {
        fetchMetrics();
      } else if (activeTab === "tab-console") {
        fetchConsoleLogs();
      }
    });
  });
}

// ----------------------------------------------------------------------------
// Event Listeners & Keyboard Controls
// ----------------------------------------------------------------------------

function initEventListeners() {
  // Review Queue Filter & Actions
  const reviewFilter = document.getElementById("filter-review-pred");
  if (reviewFilter) {
    reviewFilter.addEventListener("change", () => renderReviewGrid());
  }

  const confirmAllBtn = document.getElementById("btn-confirm-all");
  if (confirmAllBtn) {
    confirmAllBtn.addEventListener("click", confirmAllReviewItems);
  }

  // Dataset Inspector Filters
  const sampleModeFilter = document.getElementById("filter-sample-mode");
  const sampleLabelFilter = document.getElementById("filter-sample-label");
  const refreshSamplesBtn = document.getElementById("btn-refresh-samples");

  if (sampleModeFilter) sampleModeFilter.addEventListener("change", fetchDatasetSamples);
  if (sampleLabelFilter) sampleLabelFilter.addEventListener("change", fetchDatasetSamples);
  if (refreshSamplesBtn) refreshSamplesBtn.addEventListener("click", fetchDatasetSamples);

  // Batch Selection Controls
  const selectAllBtn = document.getElementById("btn-select-all-samples");
  const deselectAllBtn = document.getElementById("btn-deselect-all-samples");
  const deleteSelectedBtn = document.getElementById("btn-delete-selected-samples");

  if (selectAllBtn) {
    selectAllBtn.addEventListener("click", handleSelectAllSamples);
  }
  if (deselectAllBtn) {
    deselectAllBtn.addEventListener("click", handleDeselectAllSamples);
  }
  if (deleteSelectedBtn) {
    deleteSelectedBtn.addEventListener("click", handleBatchDeleteSamples);
  }

  // Retrain Model Button
  const retrainBtn = document.getElementById("btn-retrain");
  if (retrainBtn) {
    retrainBtn.addEventListener("click", handleRetrainModel);
  }

  // Threshold Slider
  const thresholdSlider = document.getElementById("slider-threshold");
  let thresholdDebounceTimer = null;
  if (thresholdSlider) {
    thresholdSlider.addEventListener("input", (e) => {
      const val = parseFloat(e.target.value).toFixed(2);
      document.getElementById("display-threshold-val").textContent = val;

      if (thresholdDebounceTimer) clearTimeout(thresholdDebounceTimer);
      thresholdDebounceTimer = setTimeout(async () => {
        try {
          const res = await fetch("/api/threshold", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ threshold: parseFloat(val) }),
          });
          if (!res.ok) throw new Error("Failed to update threshold");
          const data = await res.json();
          if (data.metrics) {
            updateMetricsDisplay(data.metrics);
          }
          showToast(`Active threshold updated to θ=${val}`);
        } catch (err) {
          console.error("Error updating threshold:", err);
        }
      }, 250);
    });
  }

  // Console Filters & Actions
  const logLevelFilter = document.getElementById("filter-log-level");
  const logModeFilter = document.getElementById("filter-log-mode");
  const clearLogsBtn = document.getElementById("btn-clear-logs");
  const refreshLogsBtn = document.getElementById("btn-refresh-logs");

  if (logLevelFilter) logLevelFilter.addEventListener("change", fetchConsoleLogs);
  if (logModeFilter) logModeFilter.addEventListener("change", fetchConsoleLogs);
  if (clearLogsBtn) clearLogsBtn.addEventListener("click", handleClearLogs);
  if (refreshLogsBtn) refreshLogsBtn.addEventListener("click", fetchConsoleLogs);

  // Keyboard Navigation: '1' for Like, '0' for Dislike, Arrow keys to navigate
  window.addEventListener("keydown", handleGlobalKeydown);
}

function handleGlobalKeydown(e) {
  // Ignore if user is typing in an input
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) {
    return;
  }

  if (activeTab !== "tab-review" || reviewQueueSamples.length === 0) {
    return;
  }

  if (e.key === "1") {
    e.preventDefault();
    setSampleLabel(selectedCardIndex, 1);
  } else if (e.key === "0") {
    e.preventDefault();
    setSampleLabel(selectedCardIndex, 0);
  } else if (e.key === "ArrowRight" || e.key === "ArrowDown") {
    e.preventDefault();
    if (selectedCardIndex < reviewQueueSamples.length - 1) {
      selectedCardIndex++;
      updateSelectedCardHighlight();
    }
  } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
    e.preventDefault();
    if (selectedCardIndex > 0) {
      selectedCardIndex--;
      updateSelectedCardHighlight();
    }
  }
}

// ----------------------------------------------------------------------------
// API Data Fetching
// ----------------------------------------------------------------------------

async function refreshAllData() {
  await fetchMetrics();
  await fetchReviewQueue();
}

async function fetchMetrics() {
  try {
    const res = await fetch("/api/metrics");
    if (!res.ok) throw new Error("Failed to fetch metrics");
    const data = await res.json();

    const stats = data.statistics || {};
    const model = data.model_status || {};

    // Header Statistics
    document.getElementById("stat-total-samples").textContent = stats.total_samples || 0;
    const ratioPercent = ((stats.positive_ratio || 0) * 100).toFixed(1);
    document.getElementById("stat-like-ratio").textContent = `${ratioPercent}%`;
    document.getElementById("stat-pending-review").textContent = stats.pending_auto_review_count || 0;
    document.getElementById("counter-review").textContent = stats.pending_auto_review_count || 0;
    document.getElementById("counter-samples").textContent = stats.total_samples || 0;

    // Model Status Indicator
    const statusText = document.getElementById("model-status-text");
    const indicator = document.getElementById("status-indicator");

    if (model.model_loaded) {
      statusText.textContent = "Model Active";
      indicator.className = "indicator-dot status-active";
    } else {
      statusText.textContent = "Model Not Trained";
      indicator.className = "indicator-dot status-idle";
    }

    // Model Metrics Tab Display
    renderModelTab(model, stats);
  } catch (err) {
    console.error("Error loading metrics:", err);
  }
}

async function fetchReviewQueue() {
  try {
    const res = await fetch("/api/samples?mode=auto&reviewed=0&limit=100");
    if (!res.ok) throw new Error("Failed to fetch review queue");
    reviewQueueSamples = await res.json();

    selectedCardIndex = 0;
    renderReviewGrid();
  } catch (err) {
    console.error("Error loading review queue:", err);
  }
}

async function fetchDatasetSamples() {
  const mode = document.getElementById("filter-sample-mode").value;
  const label = document.getElementById("filter-sample-label").value;

  const params = new URLSearchParams();
  if (mode) params.append("mode", mode);
  if (label) params.append("label", label);
  params.append("limit", "100");

  try {
    const res = await fetch(`/api/samples?${params.toString()}`);
    if (!res.ok) throw new Error("Failed to fetch samples");
    currentSamples = await res.json();
    renderSamplesGrid();
    updateBatchSelectionUI();
  } catch (err) {
    console.error("Error loading samples:", err);
  }
}

async function fetchConsoleLogs() {
  const level = document.getElementById("filter-log-level") ? document.getElementById("filter-log-level").value : "";
  const mode = document.getElementById("filter-log-mode") ? document.getElementById("filter-log-mode").value : "";

  const params = new URLSearchParams();
  if (level) params.append("level", level);
  if (mode) params.append("mode", mode);
  params.append("limit", "150");

  try {
    const res = await fetch(`/api/logs?${params.toString()}`);
    if (!res.ok) throw new Error("Failed to fetch activity logs");
    consoleLogs = await res.json();
    renderConsoleLogs();

    const counter = document.getElementById("counter-console");
    if (counter) counter.textContent = consoleLogs.length;
  } catch (err) {
    console.error("Error loading activity logs:", err);
  }
}

function renderConsoleLogs() {
  const container = document.getElementById("console-logs-container");
  if (!container) return;

  container.innerHTML = "";

  if (consoleLogs.length === 0) {
    container.innerHTML = `
      <div style="text-align: center; color: var(--text-muted); padding: 3rem 1rem;">
        No activity logs recorded yet. Rate artworks in the library to see live interactions.
      </div>
    `;
    return;
  }

  consoleLogs.forEach((log) => {
    const entry = document.createElement("div");
    entry.className = "log-entry";

    const modeBadge = log.mode ? `<span class="log-mode-tag">${log.mode}</span>` : "";

    entry.innerHTML = `
      <span class="log-time">[${log.timestamp}]</span>
      <span class="log-pill ${log.level}">${log.level}</span>
      ${modeBadge}
      <span class="log-msg">${escapeHtml(log.message)}</span>
    `;
    container.appendChild(entry);
  });
}

async function handleClearLogs() {
  try {
    const res = await fetch("/api/logs/clear", { method: "POST" });
    if (!res.ok) throw new Error("Failed to clear logs");
    showToast("Activity console cleared");
    await fetchConsoleLogs();
  } catch (err) {
    console.error("Error clearing logs:", err);
    showToast("Failed to clear logs", true);
  }
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ----------------------------------------------------------------------------
// Rendering
// ----------------------------------------------------------------------------

function renderReviewGrid() {
  const grid = document.getElementById("review-grid");
  const emptyState = document.getElementById("review-empty-state");
  const filterVal = document.getElementById("filter-review-pred").value;

  grid.innerHTML = "";

  const filtered = reviewQueueSamples.filter((sample) => {
    if (filterVal === "all") return true;
    const score = sample.prediction_score || 0;
    const isLike = score >= 0.35;
    return filterVal === "1" ? isLike : !isLike;
  });

  if (filtered.length === 0) {
    emptyState.classList.remove("hidden");
    return;
  }

  emptyState.classList.add("hidden");

  filtered.forEach((sample, idx) => {
    const card = document.createElement("div");
    card.className = `artwork-card ${idx === selectedCardIndex ? "selected" : ""}`;
    card.dataset.index = idx;
    card.dataset.id = sample.id;

    const isLike = sample.label === 1;
    const scoreVal = sample.prediction_score !== null ? (sample.prediction_score * 100).toFixed(0) : "N/A";

    card.innerHTML = `
      <div class="card-media">
        <button class="card-delete-btn" data-id="${sample.id}" title="Delete sample #${sample.id}" aria-label="Delete sample">✕</button>
        <img src="${sample.image_base64 || ''}" alt="Artwork" loading="lazy">
        <div class="badge-overlay ${sample.prediction_score >= 0.35 ? 'badge-like' : 'badge-dislike'}">
          ${scoreVal}% Conf
        </div>
      </div>
      <div class="card-body">
        <div class="card-meta-row">
          <span>ID: #${sample.id}</span>
          <span>Mode: ${sample.mode}</span>
        </div>
        <button class="label-toggle-btn ${isLike ? 'is-like' : 'is-dislike'}" data-id="${sample.id}">
          <span>${isLike ? '★ Liked' : '✕ Disliked'}</span>
          <span style="font-size: 0.72rem; opacity: 0.8;">Click to Toggle</span>
        </button>
      </div>
    `;

    // Click on card selects it and toggles label
    card.addEventListener("click", (e) => {
      selectedCardIndex = idx;
      updateSelectedCardHighlight();
      toggleSampleLabel(sample.id);
    });

    // Delete single sample directly from review queue
    const deleteBtn = card.querySelector(".card-delete-btn");
    deleteBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      handleDeleteSingleSample(sample.id, true);
    });

    grid.appendChild(card);
  });
}

function renderSamplesGrid() {
  const grid = document.getElementById("samples-grid");
  grid.innerHTML = "";

  if (currentSamples.length === 0) {
    grid.innerHTML = `<div class="empty-state"><h3>No Samples Found</h3><p>Try clearing filters or recording samples in Manual Mode.</p></div>`;
    return;
  }

  currentSamples.forEach((sample) => {
    const isSelected = selectedSampleIds.has(sample.id);
    const card = document.createElement("div");
    card.className = `artwork-card ${isSelected ? "is-selected" : ""}`;
    card.dataset.id = sample.id;
    const isLike = sample.label === 1;

    card.innerHTML = `
      <div class="card-media">
        <label class="card-select-wrap" title="Select sample #${sample.id}">
          <input type="checkbox" class="card-checkbox" data-id="${sample.id}" ${isSelected ? "checked" : ""}>
          <span class="custom-checkbox"></span>
        </label>
        <button class="card-delete-btn" data-id="${sample.id}" title="Delete sample #${sample.id}" aria-label="Delete sample">✕</button>
        <img src="${sample.image_base64 || ''}" alt="Artwork" loading="lazy">
        <div class="badge-overlay ${isLike ? 'badge-like' : 'badge-dislike'}">
          ${isLike ? 'LIKE' : 'DISLIKE'}
        </div>
      </div>
      <div class="card-body">
        <div class="card-meta-row">
          <span>ID: #${sample.id}</span>
          <span>Mode: ${sample.mode}</span>
        </div>
        <div class="card-meta-row">
          <span>Reviewed: ${sample.reviewed === 1 ? 'Yes' : 'Pending'}</span>
          <span>Score: ${sample.prediction_score !== null ? (sample.prediction_score * 100).toFixed(0) + '%' : '-'}</span>
        </div>
      </div>
    `;

    // Checkbox interaction
    const checkbox = card.querySelector(".card-checkbox");
    checkbox.addEventListener("click", (e) => {
      e.stopPropagation();
    });
    checkbox.addEventListener("change", (e) => {
      toggleSampleSelection(sample.id, e.target.checked);
    });

    // Delete single sample button
    const deleteBtn = card.querySelector(".card-delete-btn");
    deleteBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      handleDeleteSingleSample(sample.id, false);
    });

    // Clicking card toggles selection
    card.addEventListener("click", () => {
      const willSelect = !selectedSampleIds.has(sample.id);
      toggleSampleSelection(sample.id, willSelect);
    });

    grid.appendChild(card);
  });
}

function toggleSampleSelection(sampleId, isSelected) {
  if (isSelected === undefined) {
    isSelected = !selectedSampleIds.has(sampleId);
  }
  if (isSelected) {
    selectedSampleIds.add(sampleId);
  } else {
    selectedSampleIds.delete(sampleId);
  }
  updateCardSelectionClasses();
  updateBatchSelectionUI();
}

function handleSelectAllSamples() {
  currentSamples.forEach((sample) => {
    selectedSampleIds.add(sample.id);
  });
  updateCardSelectionClasses();
  updateBatchSelectionUI();
}

function handleDeselectAllSamples() {
  selectedSampleIds.clear();
  updateCardSelectionClasses();
  updateBatchSelectionUI();
}

function updateCardSelectionClasses() {
  const cards = document.querySelectorAll("#samples-grid .artwork-card");
  cards.forEach((card) => {
    const id = parseInt(card.dataset.id, 10);
    const checkbox = card.querySelector(".card-checkbox");
    if (selectedSampleIds.has(id)) {
      card.classList.add("is-selected");
      if (checkbox) checkbox.checked = true;
    } else {
      card.classList.remove("is-selected");
      if (checkbox) checkbox.checked = false;
    }
  });
}

function updateBatchSelectionUI() {
  const count = selectedSampleIds.size;
  const countEl = document.getElementById("selected-samples-count");
  if (countEl) countEl.textContent = count;

  const deleteBtn = document.getElementById("btn-delete-selected-samples");
  const deselectBtn = document.getElementById("btn-deselect-all-samples");

  if (deleteBtn) {
    if (count > 0) {
      deleteBtn.classList.remove("hidden");
    } else {
      deleteBtn.classList.add("hidden");
    }
  }

  if (deselectBtn) {
    if (count > 0) {
      deselectBtn.classList.remove("hidden");
    } else {
      deselectBtn.classList.add("hidden");
    }
  }
}

async function handleDeleteSingleSample(sampleId, fromReviewQueue = false) {
  try {
    const res = await fetch(`/api/samples/${sampleId}`, { method: "DELETE" });
    if (!res.ok) throw new Error("Failed to delete sample");

    selectedSampleIds.delete(sampleId);
    showToast(`Sample #${sampleId} deleted`);

    if (fromReviewQueue) {
      await fetchReviewQueue();
      await fetchMetrics();
    } else {
      await fetchDatasetSamples();
      await fetchMetrics();
    }
  } catch (err) {
    console.error("Error deleting sample:", err);
    showToast(`Failed to delete sample #${sampleId}`, true);
  }
}

async function handleBatchDeleteSamples() {
  if (selectedSampleIds.size === 0) return;
  const ids = Array.from(selectedSampleIds);

  try {
    const res = await fetch("/api/samples/batch-delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids }),
    });

    if (!res.ok) throw new Error("Failed to delete selected samples");
    const data = await res.json();

    selectedSampleIds.clear();
    showToast(`Deleted ${data.deleted_count} sample(s)`);
    await fetchDatasetSamples();
    await fetchMetrics();
  } catch (err) {
    console.error("Error batch deleting samples:", err);
    showToast("Failed to delete selected samples", true);
  }
}

function updateMetricsDisplay(m) {
  const metricsDisplay = document.getElementById("model-metrics-display");
  if (!metricsDisplay || !m) return;

  metricsDisplay.innerHTML = `
    <div class="metrics-stat-grid">
      <div class="stat-box">
        <div class="box-val">${m.pr_auc || 0}</div>
        <div class="box-lbl">PR-AUC</div>
      </div>
      <div class="stat-box">
        <div class="box-val" style="color: #10b981;">${((m.recall || 0) * 100).toFixed(0)}%</div>
        <div class="box-lbl">Recall (Likes Caught)</div>
      </div>
      <div class="stat-box">
        <div class="box-val">${((m.precision || 0) * 100).toFixed(0)}%</div>
        <div class="box-lbl">Precision</div>
      </div>
      <div class="stat-box">
        <div class="box-val" style="color: #a855f7;">${m.f2_score || 0}</div>
        <div class="box-lbl">F₂ Score</div>
      </div>
    </div>
  `;

  // Update Confusion Matrix cells
  if (m.confusion_matrix) {
    const cm = m.confusion_matrix;
    const tpEl = document.getElementById("cell-tp");
    const fpEl = document.getElementById("cell-fp");
    const fnEl = document.getElementById("cell-fn");
    const tnEl = document.getElementById("cell-tn");
    if (tpEl) tpEl.textContent = cm.true_positives || 0;
    if (fpEl) fpEl.textContent = cm.false_positives || 0;
    if (fnEl) fnEl.textContent = cm.false_negatives || 0;
    if (tnEl) tnEl.textContent = cm.true_negatives || 0;
  }
}

function renderModelTab(model, stats) {
  const metricsDisplay = document.getElementById("model-metrics-display");

  if (!model.model_loaded || !model.metrics) {
    metricsDisplay.innerHTML = `
      <div class="empty-state" style="padding: 2rem 1rem;">
        <div class="empty-icon">📊</div>
        <h4>Model Not Trained Yet</h4>
        <p>Gather initial ratings in Manual Mode on the library site, then click "Retrain Model" below.</p>
      </div>
    `;
    return;
  }

  updateMetricsDisplay(model.metrics);

  // Update slider default value only if user is not actively adjusting it
  const slider = document.getElementById("slider-threshold");
  if (slider && document.activeElement !== slider && model.decision_threshold !== undefined && model.decision_threshold !== null) {
    const formatted = parseFloat(model.decision_threshold).toFixed(2);
    slider.value = formatted;
    document.getElementById("display-threshold-val").textContent = formatted;
  }
}

// ----------------------------------------------------------------------------
// Review Actions
// ----------------------------------------------------------------------------

function updateSelectedCardHighlight() {
  const cards = document.querySelectorAll("#review-grid .artwork-card");
  cards.forEach((card, idx) => {
    if (idx === selectedCardIndex) {
      card.classList.add("selected");
      card.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } else {
      card.classList.remove("selected");
    }
  });
}

function setSampleLabel(index, newLabel) {
  if (index < 0 || index >= reviewQueueSamples.length) return;
  const sample = reviewQueueSamples[index];
  sample.label = newLabel;
  renderReviewGrid();
  showToast(`Sample #${sample.id} marked as ${newLabel === 1 ? 'Like' : 'Dislike'}`);
}

function toggleSampleLabel(sampleId) {
  const sample = reviewQueueSamples.find((s) => s.id === sampleId);
  if (!sample) return;
  sample.label = sample.label === 1 ? 0 : 1;
  renderReviewGrid();
}

async function confirmAllReviewItems() {
  if (reviewQueueSamples.length === 0) {
    showToast("No samples in review queue");
    return;
  }

  const updates = reviewQueueSamples.map((s) => ({
    id: s.id,
    label: s.label !== null ? s.label : (s.prediction_score >= 0.35 ? 1 : 0),
    reviewed: 1,
  }));

  try {
    const res = await fetch("/api/review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ updates }),
    });

    if (!res.ok) throw new Error("Failed to save reviews");
    const data = await res.json();
    showToast(`Successfully confirmed ${data.updated_count} reviews!`);
    await refreshAllData();
  } catch (err) {
    console.error("Error confirming reviews:", err);
    showToast("Error saving reviews", true);
  }
}

// ----------------------------------------------------------------------------
// Retrain Model Trigger
// ----------------------------------------------------------------------------

async function handleRetrainModel() {
  const btn = document.getElementById("btn-retrain");
  const targetRecall = parseFloat(document.getElementById("input-target-recall").value) || 0.90;
  const thresholdSlider = document.getElementById("slider-threshold");
  const currentThreshold = thresholdSlider ? parseFloat(thresholdSlider.value) : null;

  btn.disabled = true;
  btn.textContent = "⏳ Training Model...";

  try {
    const res = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_recall: targetRecall,
        threshold: currentThreshold,
      }),
    });

    const data = await res.json();
    if (data.status === "insufficient_data") {
      showToast(data.message || "Need more labeled data to train", true);
    } else if (data.status === "trained") {
      showToast(`Model retrained successfully on ${data.sample_count} samples! (θ=${data.metrics?.decision_threshold})`);
      await fetchMetrics();
    } else {
      showToast("Training completed");
    }
  } catch (err) {
    console.error("Error training model:", err);
    showToast("Model training failed", true);
  } finally {
    btn.disabled = false;
    btn.textContent = "🚀 Retrain Model";
  }
}

// ----------------------------------------------------------------------------
// Toast Notifications
// ----------------------------------------------------------------------------

function showToast(message, isError = false) {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = "toast";
  if (isError) {
    toast.style.borderColor = "var(--color-dislike)";
    toast.style.color = "#fca5a5";
  }
  toast.textContent = message;

  container.appendChild(toast);
  setTimeout(() => {
    toast.remove();
  }, 3000);
}
