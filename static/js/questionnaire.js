/**
 * Professional 40-question Likert assessment with navigator, auto-save,
 * editable answers, review, and validation.
 */
(function () {
  const STORAGE_KEY = "personatype_questionnaire_v1";
  const app = document.getElementById("quiz-app");
  if (!app) return;

  const payload = JSON.parse(document.getElementById("quiz-data").textContent);
  const questions = payload.questions;
  const likert = payload.likert;
  const total = questions.length;

  const els = {
    navigator: document.getElementById("quiz-navigator"),
    navGrid: document.getElementById("nav-grid"),
    navBackdrop: document.getElementById("nav-backdrop"),
    btnToggleNav: document.getElementById("btn-toggle-nav"),
    btnGoReview: document.getElementById("btn-go-review"),
    btnNavClose: document.getElementById("btn-nav-close"),
    btnResetAll: document.getElementById("btn-reset-all"),
    resetModal: document.getElementById("reset-modal"),
    btnResetCancel: document.getElementById("btn-reset-cancel"),
    btnResetConfirm: document.getElementById("btn-reset-confirm"),
    resetError: document.getElementById("reset-error"),
    sectionLabel: document.getElementById("section-label"),
    counter: document.getElementById("question-counter"),
    fill: document.getElementById("progress-fill"),
    track: document.querySelector(".progress-track"),
    stepQuestion: document.getElementById("step-question"),
    stepReview: document.getElementById("step-review"),
    kicker: document.getElementById("question-kicker"),
    text: document.getElementById("question-text"),
    options: document.getElementById("likert-options"),
    savedAnswer: document.getElementById("saved-answer-note"),
    hint: document.getElementById("quiz-hint"),
    btnClear: document.getElementById("btn-clear"),
    btnPrev: document.getElementById("btn-prev"),
    btnNext: document.getElementById("btn-next"),
    reviewSummary: document.getElementById("review-summary"),
    reviewList: document.getElementById("review-list"),
    btnBackEdit: document.getElementById("btn-back-edit"),
    btnSubmit: document.getElementById("btn-submit"),
    submitError: document.getElementById("submit-error"),
  };

  function normalizeAnswers(raw) {
    const answers = {};
    if (!raw || typeof raw !== "object") return answers;
    Object.entries(raw).forEach(([key, value]) => {
      const num = Number(value);
      if (value != null && !Number.isNaN(num) && num >= 1 && num <= 5) {
        answers[String(key)] = num;
      }
    });
    return answers;
  }

  function loadState() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      return {
        index: Math.min(Math.max(Number(parsed.index) || 0, 0), total - 1),
        answers: normalizeAnswers(parsed.answers),
        mode: parsed.mode === "review" ? "review" : "question",
        navigatorOpen: Boolean(parsed.navigatorOpen),
      };
    } catch {
      return null;
    }
  }

  let state = loadState() || {
    index: 0,
    answers: {},
    mode: "question",
    navigatorOpen: false,
  };

  function saveState() {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        index: state.index,
        answers: state.answers,
        mode: state.mode,
        navigatorOpen: state.navigatorOpen,
      })
    );
  }

  function getAnswer(qid) {
    const val = state.answers[String(qid)];
    return val == null ? null : Number(val);
  }

  function setAnswer(qid, value) {
    state.answers[String(qid)] = Number(value);
    saveState();
    renderNavigator();
    updateProgress();
  }

  function clearAnswer(qid) {
    delete state.answers[String(qid)];
    saveState();
    renderNavigator();
    updateProgress();
  }

  function answeredCount() {
    return questions.filter((q) => getAnswer(q.id) != null).length;
  }

  function firstUnansweredIndex() {
    return questions.findIndex((q) => getAnswer(q.id) == null);
  }

  function missingCount() {
    return total - answeredCount();
  }

  function likertLabel(value) {
    const found = likert.find((o) => o.value === Number(value));
    return found ? found.label : "—";
  }

  function updateProgress() {
    const done = answeredCount();
    const pct = Math.round((done / total) * 100);
    els.fill.style.width = pct + "%";
    els.track.setAttribute("aria-valuenow", String(pct));
    els.counter.textContent = `${done} / ${total} Questions Completed`;

    if (state.mode === "question") {
      const q = questions[state.index];
      els.sectionLabel.textContent = q ? `${q.section} · Q${state.index + 1}` : "Question";
    } else {
      els.sectionLabel.textContent = "Review";
    }
  }

  function questionStatus(i) {
    const q = questions[i];
    const answered = getAnswer(q.id) != null;
    if (state.mode === "question" && i === state.index) return "current";
    return answered ? "answered" : "unanswered";
  }

  function renderNavigator() {
    if (!els.navGrid) return;
    els.navGrid.innerHTML = "";

    questions.forEach((q, i) => {
      const status = questionStatus(i);
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `nav-q nav-q--${status}`;
      btn.title = `Question ${i + 1}${getAnswer(q.id) != null ? " (answered)" : ""}`;
      btn.innerHTML =
        `<span class="nav-q-num">Q${i + 1}</span>` +
        (status === "answered" ? `<span class="nav-q-check" aria-hidden="true">✓</span>` : "");
      btn.addEventListener("click", () => goToQuestion(i));
      els.navGrid.appendChild(btn);
    });
  }

  function setNavigatorOpen(open) {
    state.navigatorOpen = open;
    saveState();
    if (els.navigator) {
      els.navigator.classList.toggle("is-open", open);
    }
    if (els.navBackdrop) {
      els.navBackdrop.hidden = !open;
    }
  }

  function renderLikert(qid) {
    const selected = getAnswer(qid);
    els.options.innerHTML = "";

    likert.forEach((opt) => {
      const isSelected = selected === opt.value;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "likert-option" + (isSelected ? " is-selected" : "");
      btn.setAttribute("role", "radio");
      btn.setAttribute("aria-checked", isSelected ? "true" : "false");
      btn.dataset.value = String(opt.value);
      btn.innerHTML =
        `<span class="likert-value">${opt.value}</span>` +
        `<span class="likert-label">${opt.label}</span>` +
        (isSelected ? `<span class="likert-check" aria-hidden="true">✓</span>` : "");

      btn.addEventListener("click", () => {
        setAnswer(qid, opt.value);
        els.hint.hidden = true;
        renderLikert(qid);
        updateSavedAnswerNote(qid);
        if (els.btnClear) els.btnClear.disabled = false;
      });

      els.options.appendChild(btn);
    });
  }

  function updateSavedAnswerNote(qid) {
    if (!els.savedAnswer) return;
    const selected = getAnswer(qid);
    if (selected == null) {
      els.savedAnswer.hidden = true;
      els.savedAnswer.textContent = "";
      return;
    }
    els.savedAnswer.hidden = false;
    els.savedAnswer.textContent =
      `Saved answer: ${likertLabel(selected)} — tap another option to change it.`;
  }

  function goToQuestion(index) {
    state.index = Math.min(Math.max(index, 0), total - 1);
    showQuestion();
    setNavigatorOpen(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function showQuestion() {
    state.mode = "question";
    els.stepQuestion.hidden = false;
    els.stepReview.hidden = true;
    if (els.submitError) els.submitError.hidden = true;

    const q = questions[state.index];
    els.kicker.textContent = q.section;
    els.text.textContent = q.text;
    els.text.classList.remove("anim-in");
    void els.text.offsetWidth;
    els.text.classList.add("anim-in");

    const existing = getAnswer(q.id);
    els.hint.hidden = existing != null;

    renderLikert(q.id);
    updateSavedAnswerNote(q.id);
    renderNavigator();
    updateProgress();

    if (els.btnClear) els.btnClear.disabled = existing == null;
    if (els.btnPrev) els.btnPrev.disabled = state.index === 0;
    if (els.btnNext) {
      els.btnNext.textContent = state.index === total - 1 ? "Review answers" : "Next";
    }

    saveState();
  }

  function showReview() {
    state.mode = "review";
    els.stepQuestion.hidden = true;
    els.stepReview.hidden = false;

    const done = answeredCount();
    const missing = missingCount();

    if (els.reviewSummary) {
      if (missing === 0) {
        els.reviewSummary.className = "review-summary review-summary--ok";
        els.reviewSummary.innerHTML =
          `<strong>All ${total} questions answered.</strong> Review below, then submit when ready.`;
      } else {
        els.reviewSummary.className = "review-summary review-summary--warn";
        els.reviewSummary.innerHTML =
          `<strong>${missing} question${missing === 1 ? "" : "s"} still unanswered.</strong> ` +
          `Complete them before submitting.` +
          `<button type="button" class="cta cta-inline" id="btn-jump-missing">Go to first unanswered</button>`;
        document.getElementById("btn-jump-missing")?.addEventListener("click", () => {
          const idx = firstUnansweredIndex();
          if (idx >= 0) goToQuestion(idx);
        });
      }
    }

    els.reviewList.innerHTML = "";
    questions.forEach((q, i) => {
      const val = getAnswer(q.id);
      const row = document.createElement("button");
      row.type = "button";
      row.className = "review-row" + (val == null ? " is-missing" : " is-answered");
      row.innerHTML =
        `<span class="review-num">${i + 1}</span>` +
        `<span class="review-text">${q.text}</span>` +
        `<span class="review-ans">${val == null ? "Not answered" : likertLabel(val)}</span>` +
        (val != null ? `<span class="review-check" aria-hidden="true">✓</span>` : "");
      row.addEventListener("click", () => goToQuestion(i));
      els.reviewList.appendChild(row);
    });

    if (els.btnSubmit) {
      els.btnSubmit.disabled = missing > 0;
      els.btnSubmit.title = missing > 0 ? `${missing} unanswered question(s)` : "";
    }

    if (els.submitError) {
      els.submitError.hidden = missing === 0;
      if (missing > 0) {
        els.submitError.textContent =
          `Cannot submit yet — ${missing} question${missing === 1 ? "" : "s"} need an answer.`;
      }
    }

    renderNavigator();
    updateProgress();
    saveState();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  let resetInProgress = false;

  function openResetModal() {
    if (!els.resetModal || resetInProgress) return;
    els.resetModal.hidden = false;
    els.resetModal.classList.add("is-open");
    els.resetModal.setAttribute("aria-hidden", "false");
    document.body.classList.add("modal-open");
    if (els.resetError) els.resetError.hidden = true;
    els.btnResetConfirm?.focus();
  }

  function closeResetModal() {
    if (!els.resetModal) return;
    els.resetModal.classList.remove("is-open");
    els.resetModal.hidden = true;
    els.resetModal.setAttribute("aria-hidden", "true");
    document.body.classList.remove("modal-open");
  }

  function showResetError(message) {
    if (!els.resetError) return;
    els.resetError.textContent = message;
    els.resetError.hidden = false;
  }

  function resetAllResponses() {
    if (resetInProgress) return;
    resetInProgress = true;

    if (els.btnResetConfirm) {
      els.btnResetConfirm.disabled = true;
      els.btnResetConfirm.textContent = "Resetting…";
    }
    if (els.btnResetAll) els.btnResetAll.disabled = true;

    // Close modal immediately so overlay never blocks the page
    closeResetModal();
    setNavigatorOpen(false);

    try {
      state = {
        index: 0,
        answers: {},
        mode: "question",
        navigatorOpen: window.matchMedia("(min-width: 960px)").matches,
      };

      localStorage.removeItem(STORAGE_KEY);
      sessionStorage.removeItem("personatype_questionnaire_result");

      if (els.submitError) els.submitError.hidden = true;
      if (els.btnSubmit) {
        els.btnSubmit.disabled = true;
        els.btnSubmit.textContent = "Submit assessment";
      }

      showQuestion();

      // Re-open sidebar on desktop after reset
      if (window.matchMedia("(min-width: 960px)").matches) {
        setNavigatorOpen(true);
      }

      window.scrollTo({ top: 0, behavior: "auto" });
    } catch (err) {
      console.error("Reset failed:", err);
      showResetError(
        "We could not fully reset your responses. Please refresh the page and try again."
      );
    } finally {
      resetInProgress = false;
      if (els.btnResetConfirm) {
        els.btnResetConfirm.disabled = false;
        els.btnResetConfirm.textContent = "Yes, reset everything";
      }
      if (els.btnResetAll) els.btnResetAll.disabled = false;
      closeResetModal();
    }
  }

  // ---- Event listeners ----
  els.btnToggleNav?.addEventListener("click", () => setNavigatorOpen(true));
  els.btnGoReview?.addEventListener("click", () => showReview());
  els.btnNavClose?.addEventListener("click", () => setNavigatorOpen(false));
  els.navBackdrop?.addEventListener("click", () => setNavigatorOpen(false));

  els.btnClear?.addEventListener("click", () => {
    const q = questions[state.index];
    clearAnswer(q.id);
    els.hint.hidden = false;
    renderLikert(q.id);
    updateSavedAnswerNote(q.id);
    if (els.btnClear) els.btnClear.disabled = true;
  });

  els.btnResetAll?.addEventListener("click", openResetModal);

  els.btnResetCancel?.addEventListener("click", closeResetModal);

  els.btnResetConfirm?.addEventListener("click", resetAllResponses);

  // Close modal when clicking the dimmed backdrop (not the card)
  els.resetModal?.addEventListener("click", (e) => {
    if (e.target === els.resetModal) closeResetModal();
  });

  els.btnPrev?.addEventListener("click", () => {
    if (state.index > 0) goToQuestion(state.index - 1);
  });

  els.btnNext?.addEventListener("click", () => {
    const q = questions[state.index];
    if (getAnswer(q.id) == null) {
      els.hint.hidden = false;
      return;
    }
    if (state.index >= total - 1) {
      showReview();
      return;
    }
    goToQuestion(state.index + 1);
  });

  els.btnBackEdit?.addEventListener("click", () => {
    const idx = firstUnansweredIndex();
    goToQuestion(idx >= 0 ? idx : state.index);
  });

  els.btnSubmit?.addEventListener("click", async () => {
    const missing = missingCount();
    if (missing > 0) {
      showReview();
      return;
    }

    if (els.submitError) els.submitError.hidden = true;
    els.btnSubmit.disabled = true;
    els.btnSubmit.textContent = "Scoring…";

    try {
      const res = await fetch(app.dataset.scoreUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ answers: state.answers }),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        throw new Error((data && data.error) || "Scoring failed");
      }
      sessionStorage.setItem("personatype_questionnaire_result", JSON.stringify(data.result));
      localStorage.removeItem(STORAGE_KEY);
      window.location.href = app.dataset.resultsUrl;
    } catch (err) {
      if (els.submitError) {
        els.submitError.hidden = false;
        els.submitError.textContent = err.message || "Something went wrong.";
      }
      els.btnSubmit.disabled = false;
      els.btnSubmit.textContent = "Submit assessment";
    }
  });

  // Keyboard: Escape closes navigator / modal
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      setNavigatorOpen(false);
      closeResetModal();
    }
  });

  // Boot — restore last visited question and saved answers
  renderNavigator();
  if (state.mode === "review") {
    showReview();
  } else {
    showQuestion();
  }

  // Desktop: show navigator by default on wide screens
  if (window.matchMedia("(min-width: 960px)").matches) {
    setNavigatorOpen(true);
  } else if (state.navigatorOpen) {
    setNavigatorOpen(true);
  }
})();
