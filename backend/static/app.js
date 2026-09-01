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
      } else if (activeTab === "tab-visualize") {
        fetchScatterData();
        requestAnimationFrame(() => {
          resizeScatterCanvas();
          drawScatterPlot();
        });
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

  // Scatter Plot Visualizer Controls
  const scatterMethodSelect = document.getElementById("scatter-select-method");
  const scatterColorSelect = document.getElementById("scatter-select-color");
  const toggleLikes = document.getElementById("toggle-show-likes");
  const toggleDislikes = document.getElementById("toggle-show-dislikes");
  const toggleUnlabeled = document.getElementById("toggle-show-unlabeled");
  const resetScatterZoomBtn = document.getElementById("btn-reset-scatter-zoom");
  const refreshScatterBtn = document.getElementById("btn-refresh-scatter");
  const closeInspectorBtn = document.getElementById("btn-close-scatter-inspector");

  if (scatterMethodSelect) scatterMethodSelect.addEventListener("change", () => fetchScatterData());
  if (scatterColorSelect) scatterColorSelect.addEventListener("change", () => drawScatterPlot());
  if (toggleLikes) toggleLikes.addEventListener("change", () => drawScatterPlot());
  if (toggleDislikes) toggleDislikes.addEventListener("change", () => drawScatterPlot());
  if (toggleUnlabeled) toggleUnlabeled.addEventListener("change", () => drawScatterPlot());
  if (resetScatterZoomBtn) resetScatterZoomBtn.addEventListener("click", () => resetScatterView());
  if (refreshScatterBtn) refreshScatterBtn.addEventListener("click", () => fetchScatterData());
  if (closeInspectorBtn) closeInspectorBtn.addEventListener("click", () => closeScatterInspector());

  // Inspector Quick Actions
  const btnSetLike = document.getElementById("btn-scatter-set-like");
  const btnSetDislike = document.getElementById("btn-scatter-set-dislike");
  const btnDeleteSample = document.getElementById("btn-scatter-delete");

  if (btnSetLike) btnSetLike.addEventListener("click", () => handleInspectorSetLabel(1));
  if (btnSetDislike) btnSetDislike.addEventListener("click", () => handleInspectorSetLabel(0));
  if (btnDeleteSample) btnDeleteSample.addEventListener("click", () => handleInspectorDeleteSample());

  initScatterCanvas();

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

// ----------------------------------------------------------------------------
// Embedding Space Interactive Scatter Plot
// ----------------------------------------------------------------------------

let scatterPoints = [];
let scatterMetadata = { total_points: 0, method: "pca", variance_ratio: null };
let scatterTransform = {
  zoom: 1.0,
  panX: 0,
  panY: 0,
  isPanning: false,
  startX: 0,
  startY: 0,
};
let hoveredScatterPoint = null;
let selectedScatterPoint = null;

function initScatterCanvas() {
  const canvas = document.getElementById("scatter-canvas");
  if (!canvas) return;

  const card = canvas.parentElement;
  if (card && window.ResizeObserver) {
    const ro = new ResizeObserver(() => {
      if (activeTab === "tab-visualize") {
        resizeScatterCanvas();
        drawScatterPlot();
      }
    });
    ro.observe(card);
  }

  window.addEventListener("resize", () => {
    if (activeTab === "tab-visualize") {
      resizeScatterCanvas();
      drawScatterPlot();
    }
  });

  canvas.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    scatterTransform.isPanning = true;
    scatterTransform.startX = e.clientX - scatterTransform.panX;
    scatterTransform.startY = e.clientY - scatterTransform.panY;
    try {
      canvas.setPointerCapture(e.pointerId);
    } catch (_) {}
  });

  canvas.addEventListener("pointermove", (e) => {
    if (scatterTransform.isPanning) {
      scatterTransform.panX = e.clientX - scatterTransform.startX;
      scatterTransform.panY = e.clientY - scatterTransform.startY;
      drawScatterPlot();
    } else {
      handleScatterMouseMove(e);
    }
  });

  canvas.addEventListener("pointerup", (e) => {
    if (scatterTransform.isPanning) {
      scatterTransform.isPanning = false;
      try {
        canvas.releasePointerCapture(e.pointerId);
      } catch (_) {}
    }
  });

  canvas.addEventListener("pointercancel", () => {
    scatterTransform.isPanning = false;
  });

  canvas.addEventListener("pointerleave", () => {
    hideScatterTooltip();
    hoveredScatterPoint = null;
    drawScatterPlot();
  });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
    const newZoom = Math.max(0.2, Math.min(30, scatterTransform.zoom * zoomFactor));

    scatterTransform.panX = mouseX - (mouseX - scatterTransform.panX) * (newZoom / scatterTransform.zoom);
    scatterTransform.panY = mouseY - (mouseY - scatterTransform.panY) * (newZoom / scatterTransform.zoom);
    scatterTransform.zoom = newZoom;

    drawScatterPlot();
    handleScatterMouseMove(e);
  }, { passive: false });

  canvas.addEventListener("click", (e) => {
    if (hoveredScatterPoint) {
      selectedScatterPoint = hoveredScatterPoint;
      openScatterInspector(selectedScatterPoint);
      drawScatterPlot();
    }
  });
}

async function fetchScatterData() {
  const methodSelect = document.getElementById("scatter-select-method");
  const method = methodSelect ? methodSelect.value : "pca";

  try {
    const res = await fetch(`/api/embeddings/scatter?method=${method}`);
    if (!res.ok) throw new Error("Failed to fetch scatter data");
    const data = await res.json();

    scatterPoints = data.points || [];
    scatterMetadata = data;

    // Update UI counters and HUD
    const counterEl = document.getElementById("counter-visualize");
    if (counterEl) counterEl.textContent = scatterPoints.length;

    const hudPoints = document.getElementById("scatter-hud-points");
    if (hudPoints) hudPoints.textContent = `Points: ${scatterPoints.length}`;

    const hudVariance = document.getElementById("scatter-hud-variance");
    if (hudVariance) {
      if (data.variance_ratio && data.variance_ratio.length >= 2) {
        const v1 = (data.variance_ratio[0] * 100).toFixed(1);
        const v2 = (data.variance_ratio[1] * 100).toFixed(1);
        hudVariance.textContent = `PC1: ${v1}%, PC2: ${v2}%`;
      } else {
        hudVariance.textContent = `Method: ${method.toUpperCase()}`;
      }
    }

    // Update filter counts
    let likes = 0, dislikes = 0, unlabeled = 0;
    scatterPoints.forEach((pt) => {
      if (pt.label === 1) likes++;
      else if (pt.label === 0) dislikes++;
      else unlabeled++;
    });

    const lCount = document.getElementById("scatter-likes-count");
    const dCount = document.getElementById("scatter-dislikes-count");
    const uCount = document.getElementById("scatter-unlabeled-count");
    if (lCount) lCount.textContent = likes;
    if (dCount) dCount.textContent = dislikes;
    if (uCount) uCount.textContent = unlabeled;

    const emptyState = document.getElementById("scatter-empty-state");
    if (emptyState) {
      emptyState.classList.toggle("hidden", scatterPoints.length > 0);
    }

    requestAnimationFrame(() => {
      resizeScatterCanvas();
      resetScatterView();
    });
  } catch (err) {
    console.error("Error fetching scatter data:", err);
    showToast("Failed to load embedding scatter plot", true);
  }
}

function resizeScatterCanvas() {
  const canvas = document.getElementById("scatter-canvas");
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;

  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(rect.height * dpr);
}

function getScatterDataBounds() {
  if (scatterPoints.length === 0) {
    return { minX: -1, maxX: 1, minY: -1, maxY: 1, spanX: 2, spanY: 2, midX: 0, midY: 0 };
  }

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  scatterPoints.forEach((p) => {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  });

  if (minX === maxX) { minX -= 1; maxX += 1; }
  if (minY === maxY) { minY -= 1; maxY += 1; }

  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const midX = (minX + maxX) / 2;
  const midY = (minY + maxY) / 2;

  return { minX, maxX, minY, maxY, spanX, spanY, midX, midY };
}

function resetScatterView() {
  scatterTransform.zoom = 1.0;
  scatterTransform.panX = 0;
  scatterTransform.panY = 0;
  drawScatterPlot();
}

function worldToScreen(wx, wy, bounds, rect) {
  const padding = 60;
  const availW = Math.max(100, rect.width - padding * 2);
  const availH = Math.max(100, rect.height - padding * 2);

  const baseScale = Math.min(availW / bounds.spanX, availH / bounds.spanY);
  const scale = baseScale * scatterTransform.zoom;

  const cx = rect.width / 2 + scatterTransform.panX;
  const cy = rect.height / 2 + scatterTransform.panY;

  const sx = cx + (wx - bounds.midX) * scale;
  const sy = cy - (wy - bounds.midY) * scale; // Invert Y for Cartesian coordinates
  return { x: sx, y: sy };
}

function getPointColor(point, colorMode) {
  if (colorMode === "score") {
    const score = point.prediction_score;
    if (score === null || score === undefined) return "#94a3b8"; // slate
    // Interpolate: score 0 (blue #3b82f6) -> score 0.5 (purple #a855f7) -> score 1.0 (emerald #10b981)
    if (score < 0.5) {
      const t = score / 0.5;
      return interpolateColor("#3b82f6", "#a855f7", t);
    } else {
      const t = (score - 0.5) / 0.5;
      return interpolateColor("#a855f7", "#10b981", t);
    }
  }

  // Color by ground truth label
  if (point.label === 1) return "#10b981"; // Like (Emerald)
  if (point.label === 0) return "#ef4444"; // Dislike (Rose)
  return "#f59e0b"; // Unlabeled (Amber)
}

function interpolateColor(color1, color2, factor) {
  const c1 = hexToRgb(color1);
  const c2 = hexToRgb(color2);
  if (!c1 || !c2) return color1;
  const r = Math.round(c1.r + factor * (c2.r - c1.r));
  const g = Math.round(c1.g + factor * (c2.g - c1.g));
  const b = Math.round(c1.b + factor * (c2.b - c1.b));
  return `rgb(${r}, ${g}, ${b})`;
}

function hexToRgb(hex) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16),
  } : null;
}

function drawScatterPlot() {
  const canvas = document.getElementById("scatter-canvas");
  if (!canvas) return;

  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;

  const dpr = window.devicePixelRatio || 1;
  const targetW = Math.floor(rect.width * dpr);
  const targetH = Math.floor(rect.height * dpr);

  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW;
    canvas.height = targetH;
  }

  const ctx = canvas.getContext("2d");
  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, rect.width, rect.height);

  if (scatterPoints.length === 0) {
    ctx.restore();
    return;
  }

  const bounds = getScatterDataBounds();
  const colorMode = document.getElementById("scatter-select-color")?.value || "label";
  const showLikes = document.getElementById("toggle-show-likes")?.checked ?? true;
  const showDislikes = document.getElementById("toggle-show-dislikes")?.checked ?? true;
  const showUnlabeled = document.getElementById("toggle-show-unlabeled")?.checked ?? true;

  // Draw background grid lines & axes
  drawGridAndAxes(ctx, bounds, rect);

  // Render scatter points
  scatterPoints.forEach((point) => {
    if (point.label === 1 && !showLikes) return;
    if (point.label === 0 && !showDislikes) return;
    if (point.label === null && !showUnlabeled) return;

    const screenPos = worldToScreen(point.x, point.y, bounds, rect);
    const color = getPointColor(point, colorMode);
    const isHovered = hoveredScatterPoint && hoveredScatterPoint.id === point.id;
    const isSelected = selectedScatterPoint && selectedScatterPoint.id === point.id;

    const radius = isHovered ? 8 : (isSelected ? 7 : 5);

    // Glow halo
    if (isHovered || isSelected) {
      ctx.beginPath();
      ctx.arc(screenPos.x, screenPos.y, radius + 5, 0, Math.PI * 2);
      ctx.fillStyle = isHovered ? "rgba(99, 102, 241, 0.35)" : "rgba(255, 255, 255, 0.25)";
      ctx.fill();
    }

    // Main Point
    ctx.beginPath();
    ctx.arc(screenPos.x, screenPos.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = isHovered ? 2.5 : 1.5;
    ctx.strokeStyle = isHovered ? "#ffffff" : "rgba(255, 255, 255, 0.4)";
    ctx.stroke();
  });

  ctx.restore();
}

function drawGridAndAxes(ctx, bounds, rect) {
  const origin = worldToScreen(0, 0, bounds, rect);

  // Grid lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
  ctx.lineWidth = 1;

  const step = 50;
  for (let x = (origin.x % step); x < rect.width; x += step) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, rect.height);
    ctx.stroke();
  }
  for (let y = (origin.y % step); y < rect.height; y += step) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(rect.width, y);
    ctx.stroke();
  }

  // Axes crossing origin
  ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
  ctx.lineWidth = 1.5;

  if (origin.x >= 0 && origin.x <= rect.width) {
    ctx.beginPath();
    ctx.moveTo(origin.x, 0);
    ctx.lineTo(origin.x, rect.height);
    ctx.stroke();
  }

  if (origin.y >= 0 && origin.y <= rect.height) {
    ctx.beginPath();
    ctx.moveTo(0, origin.y);
    ctx.lineTo(rect.width, origin.y);
    ctx.stroke();
  }
}

function handleScatterMouseMove(e) {
  const canvas = document.getElementById("scatter-canvas");
  if (!canvas || scatterPoints.length === 0) return;

  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  const bounds = getScatterDataBounds();
  const showLikes = document.getElementById("toggle-show-likes")?.checked ?? true;
  const showDislikes = document.getElementById("toggle-show-dislikes")?.checked ?? true;
  const showUnlabeled = document.getElementById("toggle-show-unlabeled")?.checked ?? true;

  let closest = null;
  let minDistance = 14; // Hit distance threshold in px

  for (let i = 0; i < scatterPoints.length; i++) {
    const pt = scatterPoints[i];
    if (pt.label === 1 && !showLikes) continue;
    if (pt.label === 0 && !showDislikes) continue;
    if (pt.label === null && !showUnlabeled) continue;

    const screenPos = worldToScreen(pt.x, pt.y, bounds, rect);
    const dist = Math.hypot(screenPos.x - mouseX, screenPos.y - mouseY);

    if (dist < minDistance) {
      minDistance = dist;
      closest = pt;
    }
  }

  if (closest !== hoveredScatterPoint) {
    hoveredScatterPoint = closest;
    drawScatterPlot();
  }

  if (hoveredScatterPoint) {
    showScatterTooltip(hoveredScatterPoint, mouseX, mouseY, rect);
  } else {
    hideScatterTooltip();
  }
}

function showScatterTooltip(point, mouseX, mouseY, rect) {
  const tooltip = document.getElementById("scatter-tooltip");
  if (!tooltip) return;

  const imgEl = document.getElementById("scatter-tooltip-img");
  const idEl = document.getElementById("scatter-tooltip-id");
  const labelEl = document.getElementById("scatter-tooltip-label");
  const scoreEl = document.getElementById("scatter-tooltip-score");
  const modeEl = document.getElementById("scatter-tooltip-mode");

  if (idEl) idEl.textContent = `Sample #${point.id}`;
  if (imgEl) imgEl.src = point.image_url;

  if (labelEl) {
    if (point.label === 1) {
      labelEl.textContent = "Like (1)";
      labelEl.className = "badge badge-like";
    } else if (point.label === 0) {
      labelEl.textContent = "Dislike (0)";
      labelEl.className = "badge badge-dislike";
    } else {
      labelEl.textContent = "Unlabeled";
      labelEl.className = "badge badge-unlabeled";
    }
  }

  if (scoreEl) {
    scoreEl.textContent = point.prediction_score !== null ? point.prediction_score.toFixed(2) : "N/A";
  }
  if (modeEl) {
    modeEl.textContent = point.mode || "manual";
  }

  tooltip.classList.remove("hidden");

  // Keep tooltip on screen
  const tooltipWidth = 230;
  const tooltipHeight = 200;
  let posX = mouseX + 16;
  let posY = mouseY + 16;

  if (posX + tooltipWidth > rect.width) {
    posX = mouseX - tooltipWidth - 12;
  }
  if (posY + tooltipHeight > rect.height) {
    posY = mouseY - tooltipHeight - 12;
  }

  tooltip.style.left = `${posX}px`;
  tooltip.style.top = `${posY}px`;
}

function hideScatterTooltip() {
  const tooltip = document.getElementById("scatter-tooltip");
  if (tooltip) tooltip.classList.add("hidden");
}

function formatScatterDate(dateStr) {
  if (!dateStr) return "N/A";
  try {
    const d = new Date(dateStr.replace(" ", "T"));
    if (isNaN(d.getTime())) {
      const parts = dateStr.split(" ")[0].split("-");
      if (parts.length === 3) {
        return `${parts[2]}-${parts[1]}-${parts[0]}`;
      }
      return dateStr;
    }
    const day = String(d.getDate()).padStart(2, "0");
    const month = String(d.getMonth() + 1).padStart(2, "0");
    const year = d.getFullYear();
    return `${day}-${month}-${year}`;
  } catch (_) {
    return dateStr;
  }
}

function openScatterInspector(point) {
  const inspector = document.getElementById("scatter-inspector");
  if (!inspector) return;

  document.getElementById("scatter-insp-id").textContent = `#${point.id}`;
  const dateEl = document.getElementById("scatter-insp-date");
  if (dateEl) {
    dateEl.textContent = formatScatterDate(point.created_at);
  }
  document.getElementById("scatter-insp-mode").textContent = point.mode;
  document.getElementById("scatter-insp-reviewed").textContent = point.reviewed === 1 ? "Confirmed" : "Pending";
  document.getElementById("scatter-insp-score").textContent = point.prediction_score !== null ? point.prediction_score.toFixed(4) : "N/A";

  const labelEl = document.getElementById("scatter-insp-label");
  if (point.label === 1) {
    labelEl.textContent = "Like (1)";
    labelEl.style.color = "var(--color-like)";
  } else if (point.label === 0) {
    labelEl.textContent = "Dislike (0)";
    labelEl.style.color = "var(--color-dislike)";
  } else {
    labelEl.textContent = "Unlabeled";
    labelEl.style.color = "var(--color-warning)";
  }

  const imgEl = document.getElementById("scatter-inspector-img");
  if (imgEl) imgEl.src = point.image_url;

  inspector.classList.remove("hidden");
}

function closeScatterInspector() {
  const inspector = document.getElementById("scatter-inspector");
  if (inspector) inspector.classList.add("hidden");
  selectedScatterPoint = null;
  drawScatterPlot();
}

async function handleInspectorSetLabel(newLabel) {
  if (!selectedScatterPoint) return;

  try {
    const res = await fetch("/api/review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        updates: [{ id: selectedScatterPoint.id, label: newLabel, reviewed: 1 }],
      }),
    });

    if (!res.ok) throw new Error("Failed to update sample label");

    selectedScatterPoint.label = newLabel;
    selectedScatterPoint.reviewed = 1;

    // Update in scatterPoints list
    const pt = scatterPoints.find((p) => p.id === selectedScatterPoint.id);
    if (pt) {
      pt.label = newLabel;
      pt.reviewed = 1;
    }

    openScatterInspector(selectedScatterPoint);
    drawScatterPlot();
    showToast(`Sample #${selectedScatterPoint.id} updated to ${newLabel === 1 ? "Like" : "Dislike"}`);
    fetchMetrics();
  } catch (err) {
    console.error("Error setting label from inspector:", err);
    showToast("Failed to update label", true);
  }
}

async function handleInspectorDeleteSample() {
  if (!selectedScatterPoint) return;
  if (!confirm(`Delete sample #${selectedScatterPoint.id}?`)) return;

  try {
    const res = await fetch(`/api/samples/${selectedScatterPoint.id}`, {
      method: "DELETE",
    });

    if (!res.ok) throw new Error("Failed to delete sample");

    scatterPoints = scatterPoints.filter((p) => p.id !== selectedScatterPoint.id);
    closeScatterInspector();
    drawScatterPlot();
    showToast("Sample deleted successfully");
    fetchMetrics();
  } catch (err) {
    console.error("Error deleting sample from inspector:", err);
    showToast("Failed to delete sample", true);
  }
}

