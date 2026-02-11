// frontend/js/eligibility.js


export async function fetchEligibilityQuestions(candidate, userText, top_k = 10) {
  const res = await fetch("/eligibility/questions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_text: userText || null,
      selected: candidate,
      top_k
    })
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

export async function fetchEligibilityDecision(candidate, userText, questions, answers, none_apply, top_k = 10) {
  const res = await fetch("/eligibility/decision", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_text: userText || null,
      selected: candidate,
      questions: Array.isArray(questions) ? questions : [],
      answers: Array.isArray(answers) ? answers : [],
      none_apply: !!none_apply,
      top_k
    })
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

export function parseQuestionsText(questionsText) {
  const lines = String(questionsText || "")
    .split("\n")
    .map(s => s.trim())
    .filter(Boolean);

  let title = "";
  const titleLine = lines.find(l => l.toLowerCase().startsWith("title:"));
  if (titleLine) title = titleLine.split(":", 2)[1].trim();

  const questions = lines
    .filter(l => l.startsWith("- "))
    .map(l => l.slice(2).trim())
    .filter(Boolean);

  return { title, questions };
}

export function renderQuestionsCheckboxes(questions) {
  const safeQs = Array.isArray(questions) ? questions : [];
  const rows = safeQs.map((q, idx) => `
    <label class="checkRow">
      <input type="checkbox" data-qindex="${idx}" />
      <span>${q}</span>
    </label>
  `).join("");

  return `
    <div class="detailsSub">Do any of these apply to you?</div>
    <div class="detailsText">
      ${rows || `<div class="small">No questions found.</div>`}
      <label class="checkRow">
        <input type="checkbox" id="neitherApply" />
        <span>None apply</span>
      </label>
    </div>
  `;
}

export function insertInlineDetails(cardEl, payload, deps) {
  const {
    candidate,
    queryEl,
    topkEl,
    removeInlineDetails,
    removeInlineOutcome,
    showCoveredOutcome,
    showNotCoveredOutcome,
    collapseSelection,
  } = deps;

  removeInlineDetails(cardEl);
  removeInlineOutcome(cardEl);

  let title = payload?.title || "";
  let questions = payload?.questions;

  if (!Array.isArray(questions)) {
    const parsed = parseQuestionsText(payload?.questions_text || "");
    if (!title) title = parsed.title || "Eligibility questions";
    questions = parsed.questions;
  }
  if (!title) title = "Eligibility questions";

  const details = document.createElement("div");
  details.className = "cardDetails";
  details.innerHTML = `
    <div class="detailsHeader">
      <div class="detailsTitle">${title}</div>
      <button type="button" class="changeSelectionBtn">Change selection</button>
    </div>
    <div class="detailsBody">
      ${renderQuestionsCheckboxes(questions)}
      <button class="primaryBtn" id="continueBtn" type="button">Continue</button>
    </div>
  `;

  const descr = cardEl.querySelector(".description");
  if (descr) descr.insertAdjacentElement("afterend", details);
  else cardEl.appendChild(details);

  details.querySelector(".changeSelectionBtn").addEventListener("click", (e) => {
    e.stopPropagation();
    collapseSelection();
  });

  const cbNeither = details.querySelector("#neitherApply");
  const otherCbs = Array.from(details.querySelectorAll('input[type="checkbox"]')).filter(
    cb => cb !== cbNeither
  );

  const clearOutcomeOnChange = () => removeInlineOutcome(cardEl);

  if (cbNeither) {
    cbNeither.addEventListener("change", () => {
      clearOutcomeOnChange();
      if (cbNeither.checked) otherCbs.forEach(cb => (cb.checked = false));
    });
  }

  otherCbs.forEach(cb => {
    cb.addEventListener("change", () => {
      clearOutcomeOnChange();
      if (cb.checked && cbNeither) cbNeither.checked = false;
    });
  });

  const continueBtn = details.querySelector("#continueBtn");
  continueBtn.style.cursor = "pointer";
  const CONTINUE_TEXT = continueBtn.textContent;

  continueBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    removeInlineOutcome(cardEl);

    const neither = cbNeither?.checked === true;
    const anyOtherChecked = otherCbs.some(cb => cb.checked);

    if (!neither && !anyOtherChecked) {
      showNotCoveredOutcome(cardEl, "Please select an option to continue.");
      return;
    }

    continueBtn.textContent = "Evaluating…";
    continueBtn.disabled = true;

    try {
      const userText = queryEl.value.trim();
      const top_k = parseInt(topkEl.value, 10) || 10;

      const answers = (Array.isArray(questions) ? questions : []).map((_, idx) => {
        const cb = details.querySelector(`input[data-qindex="${idx}"]`);
        return cb ? cb.checked === true : false;
      });

      const resp = await fetchEligibilityDecision(
        candidate,
        userText,
        questions,
        answers,
        neither,
        top_k
      );

      const decision = resp?.decision;
      const conf = typeof resp?.confidence === "number" ? resp.confidence : null;
      const reason = (resp?.reason || "").trim();
      const label = conf == null ? "" : ` (confidence ${(conf * 100).toFixed(0)}%)`;

      if (decision === "covered")
        showCoveredOutcome(cardEl, (reason || "Covered") + label);
      else if (decision === "not_covered")
        showNotCoveredOutcome(cardEl, (reason || "Not covered") + label);
      else
        showNotCoveredOutcome(cardEl, (reason || "Uncertain — needs more info") + label);
    } catch (err) {
      showNotCoveredOutcome(
        cardEl,
        `Could not evaluate coverage: ${err?.message || String(err)}`
      );
    } finally {
      continueBtn.disabled = false;
      continueBtn.textContent = CONTINUE_TEXT;
    }
  });
}
