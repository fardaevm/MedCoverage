const DEFAULT_TOP_K = 5;
const HISTORY_KEY = "medical_coverage_history";
const HISTORY_MAX = 5;

const form = document.getElementById("searchForm");
const queryInput = document.getElementById("query");
const submitBtn = document.getElementById("btn");
const errorBox = document.getElementById("errorBox");
const resultsContainer = document.getElementById("results");
const historyListEl = document.getElementById("historyList");

/** @type {{ query: string, procedureTitle?: string, procedureCode?: string, decision?: 'covered'|'not_covered', reason?: string }[]} */
let history = [];

function loadHistory() {
  try {
    const raw = sessionStorage.getItem(HISTORY_KEY);
    if (raw) history = JSON.parse(raw);
  } catch (_) {
    history = [];
  }
}

function saveHistory() {
  try {
    sessionStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(-HISTORY_MAX)));
  } catch (_) {}
}

function addToHistory(entry) {
  history.push(entry);
  if (history.length > HISTORY_MAX) history = history.slice(-HISTORY_MAX);
  saveHistory();
  renderHistory();
}

function removeFromHistory(index) {
  if (index < 0 || index >= history.length) return;
  history.splice(index, 1);
  saveHistory();
  renderHistory();
}

let historyEditMode = false;

function setHistoryEditMode(on) {
  historyEditMode = on;
  const editBtn = document.getElementById("historyEditBtn");
  if (editBtn) {
    editBtn.textContent = historyEditMode ? "Done" : "Edit";
    editBtn.setAttribute("aria-label", historyEditMode ? "Done editing history" : "Edit history");
    editBtn.hidden = history.length === 0;
  }
  renderHistory();
}

function renderHistory() {
  if (!historyListEl) return;
  if (history.length === 0) {
    const editBtn = document.getElementById("historyEditBtn");
    if (editBtn) editBtn.hidden = true;
    historyListEl.innerHTML = '<li class="historyItem historyEmpty"><div class="historyItemContent">No searches yet.</div></li>';
    return;
  }
  const editBtn = document.getElementById("historyEditBtn");
  if (editBtn) editBtn.hidden = false;
  const reversed = history.slice().reverse();
  historyListEl.innerHTML = reversed
    .map((h, i) => {
      const realIndex = history.length - 1 - i;
      const proc = h.procedureTitle || h.procedureCode ? `${h.procedureTitle || ""} ${(h.procedureCode ? `(${h.procedureCode})` : "")}`.trim() : "";
      const decision = h.decision === "covered" ? "Covered" : h.decision === "not_covered" ? "Not covered" : "";
      const decisionCls = h.decision === "covered" ? "covered" : h.decision === "not_covered" ? "notCovered" : "";
      const deleteBtn = historyEditMode
        ? `<button type="button" class="historyItemDelete" data-index="${realIndex}" aria-label="Delete">×</button>`
        : "";
      return `
        <li class="historyItem">
          <div class="historyItemContent">
            <div class="historyQuery">${escapeHtml(h.query)}</div>
            ${proc ? `<div class="historyProcedure">${escapeHtml(proc)}</div>` : ""}
            ${decision ? `<div class="historyDecision ${decisionCls}">${escapeHtml(decision)}</div>` : ""}
          </div>
          ${deleteBtn}
        </li>
      `;
    })
    .join("");
}

function initHistoryEdit() {
  const editBtn = document.getElementById("historyEditBtn");
  if (!editBtn || !historyListEl) return;
  editBtn.addEventListener("click", () => setHistoryEditMode(!historyEditMode));
  historyListEl.addEventListener("click", (e) => {
    const del = e.target.closest(".historyItemDelete");
    if (!del) return;
    const index = parseInt(del.getAttribute("data-index"), 10);
    if (!Number.isNaN(index)) removeFromHistory(index);
  });
}

/** @type {HTMLDivElement | null} */
let selectedCardEl = null;

/** @type {import("./eligibility.js")} */
let eligibilityModule;

(function initHistory() {
  loadHistory();
  renderHistory();
  initHistoryEdit();
})();

(function initSidebarToggle() {
  const appLayout = document.getElementById("appLayout");
  const toggle = document.getElementById("sidebarToggle");
  if (!appLayout || !toggle) return;
  const closedKey = "sidebar_history_closed";
  const closed = sessionStorage.getItem(closedKey) === "1";
  if (closed) appLayout.classList.add("sidebarClosed");

  function updateLabel() {
    const isClosed = appLayout.classList.contains("sidebarClosed");
    toggle.setAttribute("aria-label", isClosed ? "Open history panel" : "Close history panel");
    toggle.title = isClosed ? "Open history" : "Close history";
  }
  updateLabel();

  toggle.addEventListener("click", () => {
    appLayout.classList.toggle("sidebarClosed");
    sessionStorage.setItem(closedKey, appLayout.classList.contains("sidebarClosed") ? "1" : "0");
    updateLabel();
  });
})();

(async () => {
  eligibilityModule = await import("/home/js/eligibility.js");
})();

const LOADING_PHRASES = [
  "Loading...",
  "Searching...",
  "Finding procedures...",
  "Looking it up...",
  "One moment..."
];

function showError(msg) {
  errorBox.textContent = msg || "";
  errorBox.hidden = !msg;
}

function showLoading() {
  const el = document.createElement("div");
  el.className = "loadingState";
  el.setAttribute("aria-live", "polite");
  el.innerHTML = `
    <span class="loadingPhrase">${LOADING_PHRASES[0]}</span>
    <span class="loadingDots" aria-hidden="true">
      <span class="dot"></span><span class="dot"></span><span class="dot"></span>
    </span>
  `;
  let i = 0;
  const phraseEl = el.querySelector(".loadingPhrase");
  const interval = setInterval(() => {
    i = (i + 1) % LOADING_PHRASES.length;
    if (phraseEl) phraseEl.textContent = LOADING_PHRASES[i];
  }, 1800);
  el._clear = () => clearInterval(interval);
  resultsContainer.appendChild(el);
  return el;
}

function hideLoading(loadingEl) {
  if (loadingEl && loadingEl._clear) loadingEl._clear();
  const el = resultsContainer.querySelector(".loadingState");
  if (el) el.remove();
}

const FLOW_LOADING_PHRASES = [
  "Loading",
  "Processing",
  "Buffering",
  "Preparing",
  "One moment",
  "Getting pathways"
];

function showFlowLoading(cardEl, collapseSelection) {
  cardEl.querySelector(".cardDetails")?.remove();
  const wrap = document.createElement("div");
  wrap.className = "cardDetails flowLoading";
  wrap.innerHTML = `
    <div class="flowHeader">
      <a href="#" class="changeLink" role="button">Change</a>
    </div>
    <div class="flowBody">
      <div class="flowLoadingState">
        <span class="flowLoadingPhrase">${FLOW_LOADING_PHRASES[0]}</span>
        <span class="flowLoadingDots" aria-hidden="true">
          <span class="dot"></span><span class="dot"></span><span class="dot"></span>
        </span>
      </div>
    </div>
  `;
  wrap.querySelector(".changeLink").addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    collapseSelection();
  });
  const phraseEl = wrap.querySelector(".flowLoadingPhrase");
  let i = 0;
  const interval = setInterval(() => {
    i = (i + 1) % FLOW_LOADING_PHRASES.length;
    if (phraseEl) phraseEl.textContent = FLOW_LOADING_PHRASES[i];
  }, 1600);
  const clear = () => {
    clearInterval(interval);
  };
  const descr = cardEl.querySelector(".description");
  if (descr) descr.insertAdjacentElement("afterend", wrap);
  else cardEl.appendChild(wrap);
  return { clear };
}

function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function matchBadgeHtml(candidate) {
  const rank = candidate.rank ?? null;
  const score = candidate.rerank_score ?? null;
  if (rank == null && score == null) return "";

  let label, cls;
  if (rank === 1) {
    label = "Best match"; cls = "matchBadge--best";
  } else if (rank <= 3) {
    label = "Good match"; cls = "matchBadge--good";
  } else {
    label = "Possible match"; cls = "matchBadge--possible";
  }

  const pct = score != null ? ` · ${Math.round(score * 100)}%` : "";
  return `<span class="matchBadge ${cls}" title="Match quality based on relevance ranking">${label}${escapeHtml(pct)}</span>`;
}

function renderCard(candidate) {
  const title = candidate.title ?? "Procedure";
  const code = candidate.code ?? "";
  const category = candidate.category ?? "";
  const description = candidate.description ?? "";

  const card = document.createElement("div");
  card.className = "card";
  card.setAttribute("data-faiss-id", String(candidate.faiss_id ?? ""));
  card.innerHTML = `
    <div class="cardTop">
      <h2 class="title">${escapeHtml(title)}</h2>
    </div>
    <div class="cardMeta">
      <span class="category">${escapeHtml(category)}</span>
      ${matchBadgeHtml(candidate)}
    </div>
    <div class="description">${escapeHtml(description)}</div>
  `;
  card.addEventListener("click", (e) => {
    if (e.target.closest(".cardDetails a, .cardDetails button")) return;
    selectCard(card, candidate);
  });
  return card;
}

function removeInlineDetails(cardEl) {
  const details = cardEl.querySelector(".cardDetails");
  if (details) details.remove();
  cardEl.classList.remove("selected", "expanded");
  cardEl.querySelectorAll(".cardTop, .category, .description").forEach((el) => {
    el.hidden = false;
  });
}

function removeInlineOutcome(cardEl) {
  const box = cardEl.querySelector(".outcomeBox");
  if (box) box.remove();
}

function showCoveredOutcome(cardEl, reason) {
  removeInlineOutcome(cardEl);
  const box = document.createElement("div");
  box.className = "outcomeBox covered";
  box.innerHTML = `
    <div class="outcomeTitle">Covered</div>
    <div class="outcomeText">${escapeHtml(reason || "This procedure is covered under the pathway you qualified for.")}</div>
  `;
  const flowBody = cardEl.querySelector(".flowBody");
  if (flowBody) {
    const qBlock = flowBody.querySelector(".questionBlock");
    if (qBlock) qBlock.after(box);
    else flowBody.appendChild(box);
  }
}

function showNotCoveredOutcome(cardEl, reason) {
  removeInlineOutcome(cardEl);
  const box = document.createElement("div");
  box.className = "outcomeBox notCovered";
  box.innerHTML = `
    <div class="outcomeTitle">Not Covered</div>
    <div class="outcomeText">${escapeHtml(reason || "This procedure is not covered under the current criteria.")}</div>
  `;
  const flowBody = cardEl.querySelector(".flowBody");
  if (flowBody) {
    const qBlock = flowBody.querySelector(".questionBlock");
    if (qBlock) qBlock.after(box);
    else flowBody.appendChild(box);
  }
}

function collapseSelection() {
  if (!selectedCardEl) return;
  removeInlineDetails(selectedCardEl);
  resultsContainer.querySelectorAll(".card").forEach((c) => c.classList.remove("hidden"));
  selectedCardEl = null;
}

function selectCard(cardEl, candidate) {
  if (selectedCardEl === cardEl) return;

  if (selectedCardEl) {
    collapseSelection();
  }

  selectedCardEl = cardEl;
  cardEl.classList.add("selected", "expanded");
  cardEl.querySelectorAll(".cardTop, .category, .description").forEach((el) => {
    el.hidden = true;
  });
  resultsContainer.querySelectorAll(".card").forEach((c) => {
    if (c !== cardEl) c.classList.add("hidden");
  });

  submitBtn.disabled = true;
  showError("");

  const flowLoading = showFlowLoading(cardEl, collapseSelection);
  eligibilityModule
    .fetchEligibilityQuestions(candidate, queryInput.value.trim(), DEFAULT_TOP_K)
    .then((payload) => {
      flowLoading.clear();
      eligibilityModule.insertInlineDetails(cardEl, payload, {
        candidate,
        queryEl: queryInput,
        topkEl: { value: String(DEFAULT_TOP_K) },
        removeInlineDetails: () => {
          removeInlineDetails(cardEl);
          resultsContainer.querySelectorAll(".card").forEach((c) => c.classList.remove("hidden"));
          selectedCardEl = null;
        },
        removeInlineOutcome,
        showCoveredOutcome,
        showNotCoveredOutcome,
        collapseSelection,
        onDecision: (candidate, decision, reason) => {
          const q = queryInput.value.trim();
          const idx = history.map((h, i) => i).reverse().find((i) => history[i].query === q && history[i].decision == null);
          const entry = {
            query: q,
            procedureTitle: candidate?.title ?? "",
            procedureCode: candidate?.code ?? "",
            decision: decision === "covered" ? "covered" : "not_covered",
            reason: reason || "",
          };
          if (idx !== undefined) {
            history[idx] = { ...history[idx], ...entry };
          } else {
            history.push(entry);
          }
          if (history.length > HISTORY_MAX) history = history.slice(-HISTORY_MAX);
          saveHistory();
          renderHistory();
        },
      });
    })
    .catch((err) => {
      flowLoading.clear();
      showError(err?.message || "Could not load eligibility.");
      collapseSelection();
    })
    .finally(() => {
      submitBtn.disabled = false;
    });
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = queryInput.value.trim();
  if (!q) return;

  showError("");
  submitBtn.disabled = true;
  resultsContainer.innerHTML = "";
  const loadingEl = showLoading();

  try {
    const res = await fetch("/match", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: q, top_k: DEFAULT_TOP_K }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const candidates = data.candidates ?? [];
    hideLoading(loadingEl);
    if (candidates.length === 0) {
      showError("No procedures found. Try different search terms.");
      return;
    }
    addToHistory({ query: q });
    candidates.forEach((c) => resultsContainer.appendChild(renderCard(c)));
  } catch (err) {
    hideLoading(loadingEl);
    showError(err?.message || "Search failed.");
  } finally {
    submitBtn.disabled = false;
  }
});
