// frontend/js/app.js
import { fetchEligibilityQuestions, insertInlineDetails } from "/home/js/eligibility.js";

const queryEl = document.getElementById("query");
const topkEl = document.getElementById("topk");
const btnEl = document.getElementById("btn");
const resultsEl = document.getElementById("results");
const errorBox = document.getElementById("errorBox");

let selectedCode = null;

let eligibilityAbort = null;

/* -----------------------------
   Helpers
------------------------------ */
function getScoreClass(score) {
  if (typeof score !== "number") return "low";
  if (score >= 0.7) return "high";
  if (score >= 0.5) return "medium";
  return "low";
}

function showError(msg) {
  errorBox.textContent = msg || "";
}

function clearResults() {
  resultsEl.innerHTML = "";
}

/* -----------------------------
   Outcome UI (inline)
------------------------------ */
function removeInlineOutcome(cardEl) {
  const existing = cardEl.querySelector(".outcomeBox");
  if (existing) existing.remove();
}

function insertOutcomeAboveContinue(cardEl, boxEl) {
  const details = cardEl.querySelector(".cardDetails");
  if (!details) {
    cardEl.appendChild(boxEl);
    return;
  }
  const continueBtn = details.querySelector(".primaryBtn");
  if (continueBtn) continueBtn.insertAdjacentElement("beforebegin", boxEl);
  else details.appendChild(boxEl);
}

function showNotCoveredOutcome(cardEl, reasonText = "Not covered") {
  removeInlineOutcome(cardEl);

  const box = document.createElement("div");
  box.className = "outcomeBox outcomeNotCovered";
  box.innerHTML = `
    <div class="outcomeTop">
      <div class="outcomeTitle">Not covered</div>
      <div class="outcomePill">Result</div>
    </div>
    <div class="outcomeText">${reasonText}</div>
  `;

  insertOutcomeAboveContinue(cardEl, box);
}

function showCoveredOutcome(cardEl, reasonText = "Covered") {
  removeInlineOutcome(cardEl);

  const box = document.createElement("div");
  box.className = "outcomeBox outcomeCovered";
  box.innerHTML = `
    <div class="outcomeTop">
      <div class="outcomeTitle">Covered</div>
      <div class="outcomePill">Result</div>
    </div>
    <div class="outcomeText">${reasonText}</div>
  `;

  insertOutcomeAboveContinue(cardEl, box);
}

/* -----------------------------
   Inline expanded questions
------------------------------ */
function removeInlineDetails(cardEl) {
  const existing = cardEl.querySelector(".cardDetails");
  if (existing) existing.remove();
}

function collapseSelection() {
  selectedCode = null;

  const cards = resultsEl.querySelectorAll(".card");
  cards.forEach(card => {
    card.classList.remove("selected");
    card.style.display = "";
    removeInlineDetails(card);
    removeInlineOutcome(card);
  });
}

/* -----------------------------
   Selection logic
------------------------------ */
async function selectCard(cardEl, candidate) {
  selectedCode = candidate.code ?? candidate.id ?? candidate.title ?? "selected";

  // cancel any in-flight request from prior selection
  if (eligibilityAbort) eligibilityAbort.abort();
  eligibilityAbort = new AbortController();

  const cards = resultsEl.querySelectorAll(".card");
  cards.forEach(card => {
    if (card === cardEl) {
      card.classList.add("selected");
      card.style.display = "";
    } else {
      card.classList.remove("selected");
      card.style.display = "none";
      removeInlineDetails(card);
      removeInlineOutcome(card);
    }
  });

  removeInlineDetails(cardEl);
  removeInlineOutcome(cardEl);

  const loading = document.createElement("div");
  loading.className = "cardDetails";
  loading.innerHTML = `
    <div class="detailsHeader">
      <button type="button" class="changeSelectionBtn">Change selection</button>
    </div>
    <div class="detailsBody">
      <div class="detailsSub">Loading eligibility questions...</div>
    </div>
  `;

  const descr = cardEl.querySelector(".description");
  if (descr) descr.insertAdjacentElement("afterend", loading);
  else cardEl.appendChild(loading);

  loading.querySelector(".changeSelectionBtn").addEventListener("click", (e) => {
    e.stopPropagation();
    if (eligibilityAbort) eligibilityAbort.abort();
    eligibilityAbort = null;
    collapseSelection();
  });

  try {
    const userText = queryEl.value.trim();
    const top_k = parseInt(topkEl.value, 10) || 10;

    // IMPORTANT: pass signal to fetchEligibilityQuestions
    const payload = await fetchEligibilityQuestions(candidate, userText, top_k, eligibilityAbort.signal);

    // if user already changed selection, ignore late response
    if (!cardEl.classList.contains("selected")) return;

    insertInlineDetails(cardEl, payload, {
      candidate,
      queryEl,
      topkEl,
      removeInlineDetails,
      removeInlineOutcome,
      showCoveredOutcome,
      showNotCoveredOutcome,
      collapseSelection
    });
  } catch (err) {
    if (err?.name === "AbortError") return; // user changed selection -> stop silently
    showError(err?.message || String(err));
    removeInlineDetails(cardEl);
  }
}

/* -----------------------------
   Card rendering
------------------------------ */
function createCandidateCard(candidate) {
  const scoreClass = getScoreClass(candidate.score);

  const rerank =
  typeof candidate.rerank_score === "number"
    ? candidate.rerank_score.toFixed(3)
    : null;

  const desc =
    typeof candidate.description === "string" &&
    candidate.description.trim() &&
    !["none","nan","null","undefined"].includes(candidate.description.trim().toLowerCase())
      ? candidate.description
      : "No description available.";

  const card = document.createElement("div");
  card.className = "card";
  card.dataset.code = candidate.code ?? "";
  card.dataset.title = candidate.title ?? "";


  card.innerHTML = `
    <div class="cardTop">
      <div class="title">${candidate.title ?? "No title"}</div>
      ${rerank ? `<div class="score high">Rerank: ${rerank}</div>` : ``}
      <!--<div class="score ${scoreClass}">
        Score: ${typeof candidate.score === "number" ? candidate.score.toFixed(3) : "N/A"}
      </div>-->
    </div>

    <div class="small">
      <span class="tag">${candidate.category ?? "Unknown"}</span>
    </div>

    <div class="price ${candidate.basic_rate == null ? "unavailable" : ""}">
      ${candidate.basic_rate != null
        ? "Medi-Cal Rate: $" + Number(candidate.basic_rate).toFixed(2)
        : "Rate not available"}
    </div>

    <button class="toggleDescr" type="button">Show description</button>

    <div class="description" hidden>
      <div>${desc}</div>
      <div style="margin-top:10px;opacity:.75;font-size:12px">
        CPT/HCPCS: <span style="font-weight:800">${candidate.code ?? "—"}</span>
      </div>
    </div>
  `;

  // Toggle description visibility (prevent select)
  const toggleBtn = card.querySelector(".toggleDescr");
  const descr = card.querySelector(".description");

  toggleBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    const willShow = descr.hasAttribute("hidden");
    if (willShow) descr.removeAttribute("hidden");
    else descr.setAttribute("hidden", "");

    card.classList.toggle("descOpen", willShow);
    toggleBtn.textContent = willShow ? "Hide description" : "Show description";
    toggleBtn.setAttribute("aria-expanded", String(willShow));
  });


  // Select on card click (except button clicks)
  card.addEventListener("click", (e) => {
    if (selectedCode) return;

    if (e.target.closest(".toggleDescr")) return;
    if (e.target.closest(".changeSelectionBtn")) return;
    if (e.target.closest(".primaryBtn")) return;

    // Async selection (fetches questions)
    selectCard(card, candidate);
  });

  return card;
}

/* -----------------------------
   Search
------------------------------ */
async function runSearch() {
  clearResults();
  showError("");
  selectedCode = null;

  const text = queryEl.value.trim();
  const top_k = parseInt(topkEl.value, 10);

  if (!text) {
    showError("Please enter a query.");
    return;
  }

  btnEl.disabled = true;

  try {
    const res = await fetch("/match", {
      method: "POST",
      headers: { "Content-Type": "application/json", "style": "expanded-question" },
      body: JSON.stringify({ text, top_k })
    });

    if (!res.ok) throw new Error(await res.text());

    const data = await res.json();
    const candidates = data.candidates || [];

    for (const c of candidates) {
      resultsEl.appendChild(createCandidateCard(c));
    }
  } catch (err) {
    showError(err?.message || String(err));
  } finally {
    btnEl.disabled = false;
  }
}

/* -----------------------------
   Events
------------------------------ */
btnEl.addEventListener("click", runSearch);
queryEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") runSearch();
});
