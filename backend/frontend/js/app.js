const queryEl = document.getElementById("query");
const topkEl = document.getElementById("topk");
const btnEl = document.getElementById("btn");
const resultsEl = document.getElementById("results");
const errorBox = document.getElementById("errorBox");

async function runSearch() {
  resultsEl.innerHTML = "";
  errorBox.textContent = "";

  const text = queryEl.value.trim();
  const top_k = parseInt(topkEl.value, 10);

  if (!text) {
    errorBox.textContent = "Please enter a query.";
    return;
  }

  btnEl.disabled = true;

  try {
    const res = await fetch("/match", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, top_k })
    });

    if (!res.ok) {
      throw new Error(await res.text());
    }

    const data = await res.json();
    const candidates = data.candidates || [];
    

    for (const c of candidates) {
      const div = document.createElement("div");
      div.className = "card";
      div.innerHTML = `
        <strong>${c.code}</strong>
        <div>${c.title}</div>
        <small>${c.category}</small>
        <small>${c.search_text}</small>
      `;
      resultsEl.appendChild(div);
    }
  } catch (err) {
    errorBox.textContent = err.message;
  } finally {
    btnEl.disabled = false;
  }
}

btnEl.addEventListener("click", runSearch);
queryEl.addEventListener("keydown", e => {
  if (e.key === "Enter") runSearch();
});
