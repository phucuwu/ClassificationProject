// ==UserScript==
// @name         Art Taste Collector & Classifier
// @namespace    http://localhost:8000/
// @version      1.1.0
// @description  Local art taste collector with Manual, Supervised, and Full Auto modes
// @author       Antigravity
// @match        https://*.tinder.com/*
// @match        https://tinder.com/*
// @match        http://*.tinder.com/*
// @match        http://tinder.com/*
// @grant        GM_xmlhttpRequest
// @grant        GM_setValue
// @grant        GM_getValue
// @connect      localhost
// @connect      127.0.0.1
// @run-at       document-idle
// ==/UserScript==

(function () {
  "use strict";

  // Restrict execution strictly to tinder.com
  const hostname = window.location.hostname.toLowerCase();
  if (hostname !== "tinder.com" && !hostname.endsWith(".tinder.com")) {
    return;
  }

  // ---------------------------------------------------------------------------
  // Configuration
  // ---------------------------------------------------------------------------
  const CONFIG = {
    apiBaseUrl: "http://127.0.0.1:8000",
    // Explicit active-card strategy only. This scope must never fall back to
    // global `body` or `main`: extraction runs strictly inside the active card.
    activeCardSelector: ".StretchedBox, [style*='background-image'], .artwork, .art-card, .image-container",
    imageContainerSelector: ".StretchedBox, [style*='background-image'], .artwork, .art-card, .image-container",
    primaryImageSelector: "img",
    minCardDimension: 120, // Min width/height in px for valid artwork elements
    autoModeDelayMs: 1000, // Delay between automated ratings in Full Auto mode
  };

  // State
  let currentMode = "manual"; // 'manual', 'supervised', 'auto'
  let isProcessing = false;
  let autoModeTimer = null;
  let lastAutoActionTime = 0;
  let supervisedPendingData = null;

  // ---------------------------------------------------------------------------
  // UI Overlay & HUD
  // ---------------------------------------------------------------------------
  function createHUD() {
    if (document.getElementById("taste-classifier-hud")) return;

    const hud = document.createElement("div");
    hud.id = "taste-classifier-hud";
    hud.innerHTML = `
      <div id="tc-status-pill" class="tc-pill tc-manual">
        <span class="tc-dot"></span>
        <span id="tc-mode-text">Mode: MANUAL</span>
      </div>
      <div id="tc-supervised-prompt" class="tc-card tc-hidden">
        <div class="tc-title">Model Prediction</div>
        <div id="tc-pred-result" class="tc-badge">-</div>
        <div class="tc-hotkeys">
          <span><kbd>Y</kbd> Accept</span>
          <span><kbd>N</kbd> Flip / Override</span>
        </div>
      </div>
      <div class="tc-help-tooltip">
        <span><kbd>←</kbd> Dislike</span>
        <span><kbd>→</kbd> Like</span>
        <span><kbd>S</kbd> Supervised</span>
        <span><kbd>A</kbd> Auto</span>
      </div>
    `;

    // Inject styles
    const style = document.createElement("style");
    style.textContent = `
      #taste-classifier-hud {
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 999999;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 8px;
        pointer-events: none;
      }
      .tc-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        background: #181b24;
        color: #f8fafc;
        border: 1px solid #2b3040;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        pointer-events: auto;
      }
      .tc-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
      }
      .tc-manual .tc-dot { background: #3b82f6; box-shadow: 0 0 6px #3b82f6; }
      .tc-supervised .tc-dot { background: #f59e0b; box-shadow: 0 0 6px #f59e0b; }
      .tc-auto .tc-dot { background: #10b981; box-shadow: 0 0 6px #10b981; }

      .tc-card {
        background: #181b24;
        border: 1px solid #3b4256;
        border-radius: 12px;
        padding: 12px 16px;
        color: #f8fafc;
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        display: flex;
        flex-direction: column;
        gap: 8px;
        min-width: 200px;
        pointer-events: auto;
        animation: tcSlideUp 0.2s ease-out;
      }
      .tc-card.tc-hidden { display: none; }
      .tc-title { font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600; }
      .tc-badge {
        font-size: 14px;
        font-weight: 700;
        padding: 6px 10px;
        border-radius: 6px;
        text-align: center;
      }
      .tc-badge-like { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
      .tc-badge-dislike { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
      .tc-hotkeys { display: flex; justify-content: space-between; font-size: 11px; color: #cbd5e1; gap: 8px; }
      .tc-help-tooltip {
        display: flex;
        gap: 8px;
        font-size: 10px;
        color: #64748b;
        background: rgba(15, 17, 23, 0.85);
        padding: 4px 8px;
        border-radius: 6px;
      }
      kbd {
        background: #2b3040;
        padding: 2px 5px;
        border-radius: 4px;
        color: #f8fafc;
        font-family: monospace;
      }
      @keyframes tcSlideUp {
        from { transform: translateY(10px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
      }
    `;

    document.head.appendChild(style);
    document.body.appendChild(hud);
  }

  function updateHUDMode(mode) {
    const pill = document.getElementById("tc-status-pill");
    const modeText = document.getElementById("tc-mode-text");
    if (!pill || !modeText) return;

    pill.className = `tc-pill tc-${mode}`;
    modeText.textContent = `Mode: ${mode.toUpperCase()}`;

    const promptCard = document.getElementById("tc-supervised-prompt");
    if (mode !== "supervised" && promptCard) {
      promptCard.classList.add("tc-hidden");
    }
  }

  // ---------------------------------------------------------------------------
  // Action Dispatcher (Arrow Key Signals)
  // ---------------------------------------------------------------------------
  function sendKeySignal(key, code, keyCode) {
    const eventOptions = {
      key: key,
      code: code,
      keyCode: keyCode,
      which: keyCode,
      bubbles: true,
      cancelable: true,
      view: window,
    };
    document.dispatchEvent(new KeyboardEvent("keydown", eventOptions));
    document.dispatchEvent(new KeyboardEvent("keyup", eventOptions));
  }

  function dispatchDislikeAction() {
    sendKeySignal("ArrowLeft", "ArrowLeft", 37);
  }

  function dispatchLikeAction() {
    sendKeySignal("ArrowRight", "ArrowRight", 39);
  }

  // ---------------------------------------------------------------------------
  // Image Extraction Waterfall (Active Slide / Multi-Image Aware)
  // ---------------------------------------------------------------------------
  function inspectAndLogImageSet(targetElement) {
    if (!targetElement) return 1;

    // Scope strictly to targetElement or its immediate card/slide container (never global body/main)
    const childImageCount = targetElement.querySelectorAll("[style*='background-image'], .StretchedBox, img").length;
    const scope = childImageCount > 1 ? targetElement : (targetElement.parentElement || targetElement);

    // 1. Check for bullet / tab indicators strictly within this element's local scope
    const bullets = Array.from(
      scope.querySelectorAll(
        "[role='tab'], .bullet, [class*='bullet'], [class*='indicator'], [aria-label*='photo' i], [aria-label*='slide' i]"
      )
    );

    // 2. Check for unique image URLs strictly within this element's local scope
    const childImageEls = Array.from(
      scope.querySelectorAll("[style*='background-image'], .StretchedBox, img")
    );

    const uniqueUrls = new Set();
    const selfUrl = getImageUrlFromElement(targetElement);
    if (selfUrl) uniqueUrls.add(selfUrl);

    for (const el of childImageEls) {
      const url = getImageUrlFromElement(el);
      if (url) uniqueUrls.add(url);
    }

    const count = Math.max(bullets.length, uniqueUrls.size, 1);

    if (count > 1) {
      console.info(`[Image Set Detected] Active card has ${count} images.`);
      sendLogToDashboard("INFO", `Image set detected: Active card has ${count} photos.`, currentMode, { image_count: count });
    } else {
      console.info(`[Single Image] Active card has 1 image.`);
      sendLogToDashboard("INFO", `Active card has 1 image.`, currentMode, { image_count: 1 });
    }

    return count;
  }

  function sendLogToDashboard(level, message, mode, details) {
    try {
      GM_xmlhttpRequest({
        method: "POST",
        url: `${CONFIG.apiBaseUrl}/api/log`,
        headers: { "Content-Type": "application/json" },
        data: JSON.stringify({
          level: level || "INFO",
          event: "image_set",
          message: message,
          mode: mode || currentMode,
          details: details || {},
        }),
        onerror: function (err) {
          console.warn("Failed to send log to dashboard API:", err);
        },
      });
    } catch (e) {
      console.warn("GM_xmlhttpRequest error in sendLogToDashboard:", e);
    }
  }

  function getImageUrlFromElement(el) {
    if (!el) return null;
    if (el.tagName === "IMG" && el.src && !el.src.startsWith("data:image/svg")) {
      return el.src;
    }
    const bg = window.getComputedStyle(el).backgroundImage;
    if (bg && bg !== "none" && bg.includes("url(")) {
      const match = bg.match(/url\(["']?([^"']+)["']?\)/);
      if (match && match[1]) return match[1];
    }
    return null;
  }

  function isElementVisible(el) {
    if (!el) return false;
    const rect = el.getBoundingClientRect();
    if (rect.width < CONFIG.minCardDimension || rect.height < CONFIG.minCardDimension) {
      return false;
    }
    // Check if on-screen
    if (rect.bottom <= 0 || rect.top >= window.innerHeight || rect.right <= 0 || rect.left >= window.innerWidth) {
      return false;
    }
    const style = window.getComputedStyle(el);
    if (style.display === "none" || style.visibility === "hidden") {
      return false;
    }
    const opacity = parseFloat(style.opacity || "1");
    if (opacity < 0.2) {
      return false;
    }
    return true;
  }

  function isAssociatedWithActiveCard(el, anchorEl) {
    if (!el || !anchorEl) return false;
    try {
      if (el === anchorEl) return true;
      if (typeof anchorEl.contains === "function" && anchorEl.contains(el)) return true;
      if (typeof el.contains === "function" && el.contains(anchorEl)) return true;
      const card = el.closest ? el.closest(CONFIG.activeCardSelector) : null;
      if (!card) return false;
      if (card === anchorEl) return true;
      if (typeof anchorEl.contains === "function" && anchorEl.contains(card)) return true;
      if (typeof card.contains === "function" && card.contains(anchorEl)) return true;
      return false;
    } catch (e) {
      return false;
    }
  }

  // Fail-closed gate: the Primary image element must remain connected, visible,
  // and associated with the active card. Returns false when validation fails.
  function validatePrimaryImage(el, anchorEl) {
    if (!el || !anchorEl) return false;
    try {
      if (!el.isConnected || !anchorEl.isConnected) return false;
    } catch (e) {
      return false;
    }
    if (!isElementVisible(el)) return false;
    if (el.getAttribute && el.getAttribute("aria-hidden") === "true") return false;
    if (!isAssociatedWithActiveCard(el, anchorEl)) return false;
    return true;
  }

  // Revalidate a captured artwork immediately before /api/record and before
  // dispatching any arrow-key rating signal. Fail closed on any failure.
  function revalidateCapturedArtwork(captured) {
    if (!captured || !captured.imageBase64 || !captured.element || !captured.anchorEl) {
      return false;
    }
    return validatePrimaryImage(captured.element, captured.anchorEl);
  }

  function logExtractionFailure(flow) {
    const message = `Active artwork validation failed (${flow}): no connected, visible Primary image associated with the active card. No sample recorded and no rating action dispatched.`;
    console.error(message);
    sendLogToDashboard("ERROR", message, currentMode, { flow: flow });
  }

  async function extractActiveArtworkImage() {
    // Explicit active-card strategy only: resolve the visible active card, then
    // extract the Primary image strictly inside it. Never falls back to global
    // `body`/`main`, canvas, or desktop screen capture. Fails closed (null).
    const candidateCards = Array.from(
      document.querySelectorAll(CONFIG.activeCardSelector)
    ).filter(isElementVisible);

    if (candidateCards.length === 0) {
      logExtractionFailure("extract: no visible active card");
      return null;
    }

    // Anchor on the largest visible active card.
    let anchorEl = candidateCards[0];
    let anchorArea = 0;
    for (const card of candidateCards) {
      const rect = card.getBoundingClientRect();
      const area = rect.width * rect.height;
      if (area > anchorArea) {
        anchorArea = area;
        anchorEl = card;
      }
    }
    const anchorRect = anchorEl.getBoundingClientRect();
    const centerX = Math.max(10, Math.min(window.innerWidth - 10, Math.round(anchorRect.left + anchorRect.width / 2)));
    const centerY = Math.max(10, Math.min(window.innerHeight - 10, Math.round(anchorRect.top + anchorRect.height / 2)));

    // Get all elements under the visual center in stacking order (topmost first)
    const elementsAtCenter = document.elementsFromPoint ? document.elementsFromPoint(centerX, centerY) : [];

    for (const el of elementsAtCenter) {
      // Ignore overlays, tooltips, HUD, or interactive buttons
      if (el.closest("#taste-classifier-hud") || el.tagName === "BUTTON") continue;

      // Check current element for image (must validate before accepting)
      const imgUrl = getImageUrlFromElement(el);
      if (imgUrl && validatePrimaryImage(el, anchorEl)) {
        try {
          const b64 = await convertImgSrcToBase64(imgUrl);
          if (b64 && validatePrimaryImage(el, anchorEl)) {
            const count = inspectAndLogImageSet(el);
            return { imageBase64: b64, element: el, anchorEl: anchorEl, imageSetCount: count };
          }
        } catch (e) {
          console.warn("Center hit-test image fetch failed:", e);
        }
      }

      // Check children of current element, scoped to the active card
      const childWithBg = el.querySelector("[style*='background-image'], .StretchedBox, img");
      if (childWithBg && validatePrimaryImage(childWithBg, anchorEl)) {
        const childUrl = getImageUrlFromElement(childWithBg);
        if (childUrl) {
          try {
            const b64 = await convertImgSrcToBase64(childUrl);
            if (b64 && validatePrimaryImage(childWithBg, anchorEl)) {
              const count = inspectAndLogImageSet(childWithBg);
              return { imageBase64: b64, element: childWithBg, anchorEl: anchorEl, imageSetCount: count };
            }
          } catch (e) { }
        }
      }
    }

    // 2. In-card fallback: score Primary-image candidates strictly inside the
    // active card subtree by visibility, active slide state, and opacity.
    const allCandidates = Array.from(
      anchorEl.querySelectorAll(`${CONFIG.primaryImageSelector}, ${CONFIG.imageContainerSelector}`)
    ).filter(isElementVisible);

    // Score candidates: give priority to aria-hidden != true, opacity == 1, and highest z-index
    const scoredCandidates = allCandidates.map((el) => {
      let score = 0;
      const rect = el.getBoundingClientRect();
      const style = window.getComputedStyle(el);
      const opacity = parseFloat(style.opacity || "1");

      if (el.getAttribute("aria-hidden") === "false") score += 100;
      if (el.getAttribute("aria-hidden") === "true") score -= 100;
      if (opacity >= 0.9) score += 50;
      if (style.zIndex && style.zIndex !== "auto") score += parseInt(style.zIndex, 10) || 0;

      // Distance from center of screen (prefer central elements)
      const distFromCenter = Math.abs(rect.left + rect.width / 2 - window.innerWidth / 2) +
        Math.abs(rect.top + rect.height / 2 - window.innerHeight / 2);
      score -= distFromCenter * 0.05;

      return { el, score, url: getImageUrlFromElement(el) };
    }).filter((item) => Boolean(item.url));

    scoredCandidates.sort((a, b) => b.score - a.score);

    for (const candidate of scoredCandidates) {
      if (!validatePrimaryImage(candidate.el, anchorEl)) continue;
      try {
        const b64 = await convertImgSrcToBase64(candidate.url);
        if (b64 && validatePrimaryImage(candidate.el, anchorEl)) {
          const count = inspectAndLogImageSet(candidate.el);
          return { imageBase64: b64, element: candidate.el, anchorEl: anchorEl, imageSetCount: count };
        }
      } catch (e) {
        console.warn("Scored candidate fetch failed:", e);
      }
    }

    // Fail closed: no canvas fallback and no desktop screen capture. If the
    // active card / Primary image cannot be validated, record and rate nothing.
    logExtractionFailure("extract: no validated Primary image in active card");
    return null;
  }

  function convertImgSrcToBase64(url) {
    return new Promise((resolve, reject) => {
      GM_xmlhttpRequest({
        method: "GET",
        url: url,
        responseType: "blob",
        onload: function (response) {
          if (response.status >= 200 && response.status < 300) {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(response.response);
          } else {
            reject(new Error(`Failed to fetch image: ${response.status}`));
          }
        },
        onerror: reject,
      });
    });
  }

  // ---------------------------------------------------------------------------
  // Backend API Requests
  // ---------------------------------------------------------------------------
  function sendRecordSample(payload) {
    return new Promise((resolve, reject) => {
      GM_xmlhttpRequest({
        method: "POST",
        url: `${CONFIG.apiBaseUrl}/api/record`,
        headers: { "Content-Type": "application/json" },
        data: JSON.stringify(payload),
        onload: (res) => {
          try {
            resolve(JSON.parse(res.responseText));
          } catch (e) {
            resolve(res.responseText);
          }
        },
        onerror: reject,
      });
    });
  }

  function sendPredictSample(imageBase64) {
    return new Promise((resolve, reject) => {
      GM_xmlhttpRequest({
        method: "POST",
        url: `${CONFIG.apiBaseUrl}/api/predict`,
        headers: { "Content-Type": "application/json" },
        data: JSON.stringify({ image_base64: imageBase64 }),
        onload: (res) => {
          try {
            resolve(JSON.parse(res.responseText));
          } catch (e) {
            reject(e);
          }
        },
        onerror: reject,
      });
    });
  }

  // ---------------------------------------------------------------------------
  // Full Auto Effectiveness Warning Acknowledgement
  // ---------------------------------------------------------------------------
  // Every Full auto activation checks the temporal-holdout effectiveness state
  // from the backend. When effectiveness is unavailable or below the agreed
  // recall-first target, the user must explicitly acknowledge the warning
  // before Full auto starts. After acknowledgement Full auto remains usable
  // per user decision.
  let fullAutoAcknowledged = false;

  function fetchEffectivenessState() {
    return new Promise((resolve) => {
      GM_xmlhttpRequest({
        method: "GET",
        url: `${CONFIG.apiBaseUrl}/api/metrics`,
        onload: function (res) {
          try {
            const data = JSON.parse(res.responseText);
            const model = (data && data.model_status) || {};
            const eff = model.effectiveness || {};
            const reasons = model.warning_reasons || eff.warning_reasons || [];
            resolve({
              warningActive: Boolean(model.warning_active || eff.warning_active),
              status: eff.status || "unknown",
              reasons: reasons,
            });
          } catch (e) {
            resolve({ warningActive: true, status: "unknown", reasons: ["temporal_evaluation_unavailable"] });
          }
        },
        onerror: function () {
          resolve({ warningActive: true, status: "unknown", reasons: ["temporal_evaluation_unavailable"] });
        },
      });
    });
  }

  function requestFullAutoAcknowledgement(state) {
    const reasonText = (state.reasons && state.reasons.length > 0)
      ? state.reasons.join("; ")
      : "temporal evaluation unavailable";
    const message =
      `Full auto effectiveness warning (${state.status}): ${reasonText}.\n\n` +
      `The latest temporal-holdout evaluation does not meet the agreed recall-first target ` +
      `(at least 30 holdout Likes, recall >= 0.80, precision >= 0.60).\n\n` +
      `Press OK to acknowledge this warning and enable Full auto anyway, or Cancel to stay in Manual mode.`;
    return window.confirm(message);
  }

  async function tryEnableFullAuto() {
    let state;
    try {
      state = await fetchEffectivenessState();
    } catch (e) {
      state = { warningActive: true, status: "unknown", reasons: ["temporal_evaluation_unavailable"] };
    }
    if (state.warningActive) {
      const acknowledged = requestFullAutoAcknowledgement(state);
      if (!acknowledged) {
        console.info("Full auto activation declined: effectiveness warning not acknowledged.");
        return;
      }
    }
    fullAutoAcknowledged = true;
    currentMode = "auto";
    updateHUDMode("auto");
    runFullAutoStep();
  }

  // ---------------------------------------------------------------------------
  // Operating Modes Execution
  // ---------------------------------------------------------------------------

  // Manual Mode Handler (fail closed: no record and no rating signal unless
  // the active card / Primary image validates immediately before dispatch).
  async function handleManualRating(label) {
    if (isProcessing) return;
    isProcessing = true;

    try {
      const extracted = await extractActiveArtworkImage();
      if (!extracted || !extracted.imageBase64 || !revalidateCapturedArtwork(extracted)) {
        logExtractionFailure("manual");
        return;
      }
      await sendRecordSample({
        image_base64: extracted.imageBase64,
        label: label,
        mode: "manual",
        reviewed: 1,
        image_set_count: extracted.imageSetCount,
      });
      if (!revalidateCapturedArtwork(extracted)) {
        logExtractionFailure("manual-dispatch");
        return;
      }
      if (label === 1) dispatchLikeAction();
      else dispatchDislikeAction();
    } catch (err) {
      console.error("Manual recording error:", err);
    } finally {
      isProcessing = false;
    }
  }

  // Supervised Mode Trigger
  async function runSupervisedStep() {
    if (isProcessing || currentMode !== "supervised") return;
    isProcessing = true;

    const promptCard = document.getElementById("tc-supervised-prompt");
    const predResult = document.getElementById("tc-pred-result");

    try {
      const extracted = await extractActiveArtworkImage();
      if (!extracted || !extracted.imageBase64 || !revalidateCapturedArtwork(extracted)) {
        logExtractionFailure("supervised-predict");
        isProcessing = false;
        return;
      }

      const pred = await sendPredictSample(extracted.imageBase64);

      if (!pred.model_loaded) {
        alert("Model not trained yet! Please gather ratings in Manual Mode first.");
        currentMode = "manual";
        updateHUDMode("manual");
        isProcessing = false;
        return;
      }

      if (!revalidateCapturedArtwork(extracted)) {
        logExtractionFailure("supervised-predict");
        isProcessing = false;
        return;
      }

      supervisedPendingData = {
        imageBase64: extracted.imageBase64,
        element: extracted.element,
        anchorEl: extracted.anchorEl,
        predictionScore: pred.prediction_score,
        predictedDecision: pred.decision,
        imageSetCount: extracted.imageSetCount,
      };

      const isLike = pred.decision === 1;
      const conf = (pred.prediction_score * 100).toFixed(0);

      predResult.className = `tc-badge ${isLike ? "tc-badge-like" : "tc-badge-dislike"}`;
      predResult.textContent = `${isLike ? "★ LIKE" : "✕ DISLIKE"} (${conf}% conf)`;
      promptCard.classList.remove("tc-hidden");
    } catch (err) {
      console.error("Supervised prediction error:", err);
    } finally {
      isProcessing = false;
    }
  }

  async function resolveSupervisedStep(accept) {
    if (!supervisedPendingData) return;

    const data = supervisedPendingData;
    supervisedPendingData = null;

    const promptCard = document.getElementById("tc-supervised-prompt");
    if (promptCard) promptCard.classList.add("tc-hidden");

    const finalLabel = accept ? data.predictedDecision : (data.predictedDecision === 1 ? 0 : 1);

    // Revalidate the captured artwork immediately before recording and action
    // dispatch. Fail closed: no /api/record and no arrow-key signal on failure.
    if (!revalidateCapturedArtwork(data)) {
      logExtractionFailure("supervised-resolve");
      return;
    }

    await sendRecordSample({
      image_base64: data.imageBase64,
      label: finalLabel,
      mode: "supervised",
      prediction_score: data.predictionScore,
      reviewed: 1,
      image_set_count: data.imageSetCount,
    });

    if (!revalidateCapturedArtwork(data)) {
      logExtractionFailure("supervised-dispatch");
      return;
    }

    if (finalLabel === 1) dispatchLikeAction();
    else dispatchDislikeAction();

    // Automatically trigger prediction for next artwork after a short pause
    setTimeout(() => {
      if (currentMode === "supervised") runSupervisedStep();
    }, 600);
  }

  // Full Auto Mode Loop
  async function runFullAutoStep() {
    if (currentMode !== "auto") return;

    try {
      // Enforce rate limit: wait at least 1 second between consecutive ratings
      const now = Date.now();
      const timeSinceLastAction = now - lastAutoActionTime;
      if (timeSinceLastAction < CONFIG.autoModeDelayMs) {
        const remainingWait = CONFIG.autoModeDelayMs - timeSinceLastAction;
        autoModeTimer = setTimeout(runFullAutoStep, remainingWait);
        return;
      }

      const extracted = await extractActiveArtworkImage();
      if (!extracted || !extracted.imageBase64 || !revalidateCapturedArtwork(extracted)) {
        logExtractionFailure("auto");
        autoModeTimer = setTimeout(runFullAutoStep, 500);
        return;
      }

      const pred = await sendPredictSample(extracted.imageBase64);

      if (currentMode !== "auto") return;

      if (!pred.model_loaded) {
        alert("Model not trained yet! Please gather ratings in Manual Mode first.");
        currentMode = "manual";
        updateHUDMode("manual");
        return;
      }

      // Revalidate immediately before recording and dispatching. Fail closed:
      // no /api/record and no arrow-key signal when validation fails.
      if (!revalidateCapturedArtwork(extracted)) {
        logExtractionFailure("auto");
        autoModeTimer = setTimeout(runFullAutoStep, 500);
        return;
      }

      const decision = pred.decision;

      // Log decision to review queue (reviewed = 0)
      sendRecordSample({
        image_base64: extracted.imageBase64,
        label: decision,
        mode: "auto",
        prediction_score: pred.prediction_score,
        reviewed: 0,
        image_set_count: extracted.imageSetCount,
      });

      lastAutoActionTime = Date.now();
      if (!revalidateCapturedArtwork(extracted)) {
        logExtractionFailure("auto-dispatch");
        autoModeTimer = setTimeout(runFullAutoStep, CONFIG.autoModeDelayMs);
        return;
      }
      if (decision === 1) dispatchLikeAction();
      else dispatchDislikeAction();

      // Wait at least 1 second before processing the next artwork
      autoModeTimer = setTimeout(runFullAutoStep, CONFIG.autoModeDelayMs);
    } catch (err) {
      console.error("Auto mode error:", err);
      autoModeTimer = setTimeout(runFullAutoStep, Math.max(CONFIG.autoModeDelayMs, 1000));
    }
  }

  // ---------------------------------------------------------------------------
  // Currency Popup Auto-Dismiss
  // ---------------------------------------------------------------------------
  function dismissCurrencyPopups() {
    // Only search active modal dialog containers
    const modalContainers = Array.from(
      document.querySelectorAll(
        "[role='dialog'], [aria-modal='true'], .modal-dialog, [data-testid*='modal']"
      )
    ).filter((container) => !container.closest("#taste-classifier-hud"));

    if (modalContainers.length === 0) return;

    for (const modal of modalContainers) {
      const modalText = (modal.innerText || "").toLowerCase();

      // Check for Vietnamese Dong currency indicators only
      const hasCurrency = modalText.includes("₫") || modalText.includes("vnd") || /[\d\.,\s]+đ\b/i.test(modalText);

      if (!hasCurrency) continue;

      // Find close or dismiss button strictly inside this modal
      const modalButtons = Array.from(modal.querySelectorAll("button, [role='button']"));
      const closeBtn = modalButtons.find((btn) => {
        const aria = (btn.getAttribute("aria-label") || "").toLowerCase();
        const txt = (btn.innerText || "").toLowerCase().trim();
        return (
          aria.includes("close") ||
          aria.includes("dismiss") ||
          txt === "✕" ||
          txt === "x" ||
          txt === "close" ||
          txt.includes("no thanks") ||
          txt.includes("maybe later") ||
          txt.includes("not now") ||
          btn.querySelector("svg")
        );
      });

      if (closeBtn) {
        try {
          closeBtn.click();
          console.info("Auto-dismissed currency popup:", modal);
          return;
        } catch (e) { }
      }

      // Fallback: send Escape key to close the modal
      const escOptions = {
        key: "Escape",
        code: "Escape",
        keyCode: 27,
        which: 27,
        bubbles: true,
        cancelable: true,
        view: window,
      };
      document.dispatchEvent(new KeyboardEvent("keydown", escOptions));
      document.dispatchEvent(new KeyboardEvent("keyup", escOptions));
    }
  }

  function initPopupWatcher() {
    let debounceTimer = null;
    const observer = new MutationObserver(() => {
      if (debounceTimer) clearTimeout(debounceTimer);
      debounceTimer = setTimeout(dismissCurrencyPopups, 300);
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
    });
  }

  // ---------------------------------------------------------------------------
  // Global Keyboard Listener
  // ---------------------------------------------------------------------------
  window.addEventListener("keydown", (e) => {
    // Only process trusted user keystrokes; let synthetic key signals pass to the library
    if (!e.isTrusted) {
      return;
    }

    // Ignore input fields
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) {
      return;
    }

    if (e.key === "s" || e.key === "S") {
      e.preventDefault();
      if (autoModeTimer) {
        clearTimeout(autoModeTimer);
        autoModeTimer = null;
      }
      if (currentMode === "supervised") {
        currentMode = "manual";
      } else {
        currentMode = "supervised";
        runSupervisedStep();
      }
      updateHUDMode(currentMode);
    } else if (e.key === "a" || e.key === "A") {
      e.preventDefault();
      if (currentMode === "auto") {
        currentMode = "manual";
        fullAutoAcknowledged = false;
        if (autoModeTimer) {
          clearTimeout(autoModeTimer);
          autoModeTimer = null;
        }
        updateHUDMode(currentMode);
      } else {
        // Every Full auto activation re-checks effectiveness and requires
        // explicit acknowledgement when the warning is active.
        tryEnableFullAuto();
      }
    } else if (currentMode === "manual") {
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        handleManualRating(0);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        handleManualRating(1);
      }
    } else if (currentMode === "supervised" && supervisedPendingData) {
      if (e.key === "y" || e.key === "Y") {
        e.preventDefault();
        resolveSupervisedStep(true);
      } else if (e.key === "n" || e.key === "N") {
        e.preventDefault();
        resolveSupervisedStep(false);
      }
    }
  });

  // Initialize HUD and Popup Watcher on page load
  createHUD();
  initPopupWatcher();
  console.log("Art Taste Collector userscript initialized.");
})();
