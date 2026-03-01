export async function startEligibility(candidate, userText, top_k = 10, signal = null) {
  const res = await fetch("/eligibility/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_text: userText || null,
      selected: candidate,
      top_k
    }),
    ...(signal ? { signal } : {})
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

export async function fetchEligibilityQuestions(candidate, userText, top_k = 10, signal = null) {
  return await startEligibility(candidate, userText, top_k, signal);
}

export async function nextEligibility(candidate, userText, pathways, qa_so_far, top_k = 10, signal = null) {
  const res = await fetch("/eligibility/next", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_text: userText || null,
      selected: candidate,
      pathways: Array.isArray(pathways) ? pathways : [],
      qa_so_far: Array.isArray(qa_so_far) ? qa_so_far : [],
      top_k
    }),
    ...(signal ? { signal } : {})
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

function escapeHtml(s) {
  return String(s || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
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
    onDecision
  } = deps;

  cardEl.querySelector(".cardDetails")?.remove();
  removeInlineOutcome(cardEl);

  const pathways = Array.isArray(payload?.pathways) ? payload.pathways : [];
  const firstQ = (payload?.question || "").trim();
  const noPathways = payload?.no_pathways === true || (!firstQ && pathways.every((p) => !(p || "").trim()));
  const noPathwaysReason = (payload?.no_pathways_reason || "We couldn't find coverage pathways for this procedure in our policy.").trim();

  const details = document.createElement("div");
  details.className = "cardDetails";

  const pathwaysHtml = pathways
    .map((p, i) => `<li>${i + 1}. ${escapeHtml(p || "")}</li>`)
    .join("");

  details.innerHTML = `
    <div class="flowHeader">
      <a href="#" class="changeLink" role="button">Change</a>
    </div>
    <div class="flowBody">
      <div class="pathwaysLabel">PATHWAYS</div>
      <ul class="pathwaysList">${pathwaysHtml || "<li>—</li>"}</ul>
      <div class="questionBlock" id="questionBlock">
        <div class="answerHistory" id="answerHistory"></div>
        <div class="currentQuestion" id="currentQuestion">
          <p class="currentQuestionText" id="currentQuestionText">${firstQ ? escapeHtml(firstQ) : "No question generated."}</p>
        </div>
        <div class="ynRow" id="ynRow" ${firstQ ? "" : "hidden"}>
          <button class="ynBtn yes" type="button" data-ans="yes">Yes</button>
          <button class="ynBtn no" type="button" data-ans="no">No</button>
        </div>
        <div class="thinkingState" id="chatFoot" hidden></div>
      </div>
    </div>
  `;

  const descr = cardEl.querySelector(".description");
  if (descr) descr.insertAdjacentElement("afterend", details);
  else cardEl.appendChild(details);

  if (noPathways) {
    showNotCoveredOutcome(cardEl, noPathwaysReason);
    onDecision?.(candidate, "not_covered", noPathwaysReason);
    const ynRowEl = details.querySelector("#ynRow");
    const currentQEl = details.querySelector("#currentQuestionText");
    if (ynRowEl) ynRowEl.hidden = true;
    if (currentQEl) currentQEl.textContent = noPathwaysReason;
    details.querySelector(".changeLink").addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      collapseSelection();
    });
    return;
  }

  details.querySelector(".changeLink").addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    collapseSelection();
  });

  const answerHistoryEl = details.querySelector("#answerHistory");
  const currentQuestionTextEl = details.querySelector("#currentQuestionText");
  const ynRow = details.querySelector("#ynRow");
  const foot = details.querySelector("#chatFoot");

  const NEXT_LOADING_PHRASES = [
    "Loading",
    "Processing",
    "Checking",
    "Getting next question",
    "One moment",
    "Thinking",
    "Working on it",
    "Almost there"
  ];

  function startThinking() {
    foot.hidden = false;
    foot.classList.remove("thinkingSingleDot");
    foot.classList.add("thinkingState");
    foot.innerHTML = `
      <span class="thinkingPhrase">${NEXT_LOADING_PHRASES[0]}</span>
      <span class="thinkingDots" aria-hidden="true">
        <span class="dot"></span><span class="dot"></span><span class="dot"></span>
      </span>
    `;
    let i = 0;
    const phraseEl = foot.querySelector(".thinkingPhrase");
    const interval = setInterval(() => {
      i = (i + 1) % NEXT_LOADING_PHRASES.length;
      if (phraseEl) phraseEl.textContent = NEXT_LOADING_PHRASES[i];
    }, 1800);
    return () => clearInterval(interval);
  }

  function stopThinking() {
    foot.hidden = true;
    foot.innerHTML = "";
    foot.classList.remove("thinkingState", "thinkingSingleDot");
  }

  function renderAnswerHistory(qa) {
    if (!qa.length) {
      answerHistoryEl.innerHTML = "";
      return;
    }
    answerHistoryEl.innerHTML = qa
      .map(
        ({ q: qText, a }) => `
        <div class="answerHistoryRow">
          <span class="answerHistoryDot ${a ? "yes" : "no"}"></span>
          <span class="answerHistoryQ">${escapeHtml(qText)}</span>
          <span class="answerHistoryA ${a ? "yes" : "no"}">${a ? "Yes" : "No"}</span>
        </div>
      `
      )
      .join("");
  }

  function setCurrentQuestion(text) {
    currentQuestionTextEl.textContent = text || "No question generated.";
    if (text) {
      ynRow.hidden = false;
      ynRow.querySelectorAll("button").forEach((b) => (b.disabled = false));
    } else {
      ynRow.hidden = true;
    }
  }

  let qa = [];
  let currentQ = firstQ;

  async function step(answerBool) {
    if (!currentQ) return;

    qa.push({ q: currentQ, a: !!answerBool });
    renderAnswerHistory(qa);
    setCurrentQuestion("");

    ynRow.querySelectorAll("button").forEach((b) => (b.disabled = true));
    const stopThinkingFn = startThinking();

    try {
      const userText = queryEl && queryEl.value ? queryEl.value.trim() : "";
      const top_k = parseInt(topkEl && topkEl.value ? topkEl.value : "5", 10) || 10;

      const resp = await nextEligibility(candidate, userText, pathways, qa, top_k);
      stopThinkingFn();
      stopThinking();

      const decision = resp?.decision;
      const reason = (resp?.reason || "").trim();

      if (decision === "covered") {
        const msg = reason || "Based on your answers, this procedure is covered under your plan.";
        showCoveredOutcome(cardEl, msg);
        onDecision?.(candidate, "covered", msg);
        ynRow.hidden = true;
        return;
      }

      if (decision === "not_covered") {
        const msg = reason || "Based on your answers, this procedure does not appear to be covered.";
        showNotCoveredOutcome(cardEl, msg);
        onDecision?.(candidate, "not_covered", msg);
        ynRow.hidden = true;
        return;
      }

      const nextQ = (resp?.next_question || "").trim();
      if (!nextQ) {
        const msg = reason || "We need more information to determine coverage.";
        showNotCoveredOutcome(cardEl, msg);
        onDecision?.(candidate, "not_covered", msg);
        ynRow.hidden = true;
        return;
      }

      currentQ = nextQ;
      setCurrentQuestion(nextQ);
    } catch (err) {
      stopThinkingFn();
      stopThinking();
      const msg = `Could not continue: ${err?.message || String(err)}`;
      showNotCoveredOutcome(cardEl, msg);
      onDecision?.(candidate, "not_covered", msg);
      ynRow.hidden = true;
    }
  }

  ynRow.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-ans]");
    if (!btn) return;
    e.stopPropagation();
    step(btn.dataset.ans === "yes");
  });
}
