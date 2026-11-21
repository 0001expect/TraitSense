/**
 * Render questionnaire results from sessionStorage.
 */
(function () {
  const root = document.getElementById("results-root");
  const loading = document.getElementById("loading-msg");
  if (!root) return;

  let result;
  try {
    result = JSON.parse(sessionStorage.getItem("personatype_questionnaire_result") || "null");
  } catch {
    result = null;
  }

  if (!result || !result.personality_type) {
    if (loading) loading.textContent = "No results found.";
    root.insertAdjacentHTML(
      "beforeend",
      `<p class="lede">Please <a class="text-link" href="/questionnaire">retake the questionnaire</a>.</p>`
    );
    return;
  }

  if (loading) loading.remove();

  const profile = result.profile || {};
  const dims = result.dimensions || {};

  const dimHtml = Object.keys(dims)
    .map((axis) => {
      const d = dims[axis];
      return `
        <div class="dim-row">
          <div class="dim-labels">
            <span class="${d.chosen === d.left ? "is-strong" : ""}">${d.left} ${d.percent_left}%</span>
            <span class="${d.chosen === d.right ? "is-strong" : ""}">${d.right} ${d.percent_right}%</span>
          </div>
          <div class="dim-track">
            <div class="dim-fill" style="width:${d.percent_right}%"></div>
          </div>
          <p class="dim-chosen">Leans <strong>${d.chosen}</strong> · ${d.confidence}% confidence</p>
        </div>`;
    })
    .join("");

  const list = (arr) =>
    Array.isArray(arr) && arr.length
      ? `<ul>${arr.map((x) => `<li>${x}</li>`).join("")}</ul>`
      : "<p>—</p>";

  root.innerHTML = `
    <article class="result anim-in">
      <p class="result-label">Questionnaire result</p>
      <p class="result-type">${result.personality_type}</p>
      ${profile.title ? `<p class="result-title">${profile.title}</p>` : ""}
      <p class="result-meta">Overall confidence: ${result.confidence}%</p>
      ${profile.description ? `<p class="result-desc">${profile.description}</p>` : ""}

      <div class="block">
        <h2>Dimension breakdown</h2>
        <div class="dim-grid">${dimHtml}</div>
      </div>

      <div class="two-col">
        <div class="block"><h2>Strengths</h2>${list(profile.strengths)}</div>
        <div class="block"><h2>Growth areas</h2>${list(profile.weaknesses)}</div>
      </div>

      <div class="block">
        <h2>Career suggestions</h2>
        <p>${(profile.careers || []).join(", ") || "—"}</p>
      </div>
      <div class="block">
        <h2>Learning style</h2>
        <p>${profile.learning_style || "—"}</p>
      </div>
      <div class="block">
        <h2>Communication style</h2>
        <p>${profile.communication_style || "—"}</p>
      </div>

      <div class="quiz-nav" style="margin-top:1.5rem">
        <a class="btn-ghost" href="/questionnaire">Retake questionnaire</a>
        <a class="cta" href="/">Choose another method</a>
      </div>
    </article>
  `;
})();
