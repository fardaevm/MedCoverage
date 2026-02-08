const queryEl = document.getElementById("query");
const topkEl = document.getElementById("topk");
const btnEl = document.getElementById("btn");
const resultsEl = document.getElementById("results");
const errorBox = document.getElementById("errorBox");

let selectedCode = null;

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

function insertInlineDetails(cardEl) {
  removeInlineDetails(cardEl);
  removeInlineOutcome(cardEl);

  const details = document.createElement("div");
  details.className = "cardDetails";

  details.innerHTML = `
    <div class="detailsHeader">
      <button type="button" class="changeSelectionBtn">Change selection</button>
    </div>

    <div class="detailsBody">
      <div class="detailsSub">Do any of these apply to you?</div>

      <div class="detailsText">
        <label class="checkRow">
          <input type="checkbox" id="hasInsurance" name="q" value="insurance" />
          <span>
            You've already used two outpatient therapy services (acupuncture, chirpractic, speech therapy) this calendar month
          </span>

        </label>

        <label class="checkRow">
          <input type="checkbox" id="appt2w" name="q" value="appt_2w" />
          <span>You don't have an appointment within 2 weeks</span>
        </label>

        <label class="checkRow">
          <input type="checkbox" id="neitherApply" name="q" value="neither" />
          <span>Neither apply</span>
        </label>
      </div>

      <button class="primaryBtn" id="continueBtn" type="button">Continue</button>
    </div>
  `;

  // Insert below description
  const descr = cardEl.querySelector(".description");
  if (descr) descr.insertAdjacentElement("afterend", details);
  else cardEl.appendChild(details);

  // Wire Change selection
  const changeBtn = details.querySelector(".changeSelectionBtn");
  changeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    collapseSelection();
  });

  // Mutual exclusivity: "Neither apply" vs others
  const cbInsurance = details.querySelector("#hasInsurance");
  const cbNew = details.querySelector("#newPatient");
  const cbAppt = details.querySelector("#appt2w");
  const cbNeither = details.querySelector("#neitherApply");

  const otherCbs = [cbInsurance, cbNew, cbAppt].filter(Boolean);

  function clearOutcomeOnChange() {
    removeInlineOutcome(cardEl);
  }

  if (cbNeither) {
    cbNeither.addEventListener("change", () => {
      clearOutcomeOnChange();
      if (cbNeither.checked) {
        otherCbs.forEach(cb => (cb.checked = false));
      }
    });
  }

  otherCbs.forEach(cb => {
    cb.addEventListener("change", () => {
      clearOutcomeOnChange();
      if (cb.checked && cbNeither) cbNeither.checked = false;
    });
  });

  // Continue logic:
  // - neither checked => Covered
  // - any other checked => Not covered
  // - nothing checked => prompt
  const continueBtn = details.querySelector("#continueBtn");
  continueBtn.addEventListener("click", (e) => {
    e.stopPropagation();

    const neither = cbNeither?.checked === true;
    const anyOtherChecked = otherCbs.some(cb => cb.checked);

    if (neither) {
      showCoveredOutcome(cardEl, "Covered");
      return;
    }

    if (anyOtherChecked) {
      showNotCoveredOutcome(cardEl, "Not covered");
      return;
    }

    showNotCoveredOutcome(cardEl, "Please select an option to continue.");
  });
}

/* -----------------------------
   Selection logic
------------------------------ */
function selectCard(cardEl, candidate) {
  selectedCode = candidate.code ?? candidate.id ?? candidate.title ?? "selected";

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

  insertInlineDetails(cardEl);
  cardEl.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* -----------------------------
   Card rendering
------------------------------ */
function createCandidateCard(candidate) {
  const scoreClass = getScoreClass(candidate.score);

  const card = document.createElement("div");
  card.className = "card";
  card.dataset.code = candidate.code ?? "";
  card.dataset.title = candidate.title ?? "";

  card.innerHTML = `
    <div class="cardTop">
      <div class="title">${candidate.title ?? "No title"}</div>
      <div class="score ${scoreClass}">
        Score: ${typeof candidate.score === "number" ? candidate.score.toFixed(3) : "N/A"}
      </div>
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

    <div class="description" style="display:none;">
      ${candidate.description ?? "No description available."}
    </div>
  `;

  // Toggle description visibility (prevent select)
  const toggleBtn = card.querySelector(".toggleDescr");
  const descr = card.querySelector(".description");
  toggleBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    const hidden = descr.style.display === "none";
    descr.style.display = hidden ? "block" : "none";
    toggleBtn.textContent = hidden ? "Hide description" : "Show description";
  });

  // Select on card click (except button clicks)
  card.addEventListener("click", (e) => {
    if (selectedCode) return;

    if (e.target.closest(".toggleDescr")) return;
    if (e.target.closest(".changeSelectionBtn")) return;
    if (e.target.closest(".primaryBtn")) return;

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
