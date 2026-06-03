/* =========================================================
   MOVIE-RECSYS  —  Inspection Dashboard  (vanilla JS)
   ========================================================= */

const API = "";   // same-origin; all fetches are relative

// ── State ────────────────────────────────────────────────
let currentUserId   = null;
let currentStage    = "both";
const recsCache     = { both: null, retrieval: null };  // per-user cache

// ── DOM refs ─────────────────────────────────────────────
const userInput      = document.getElementById("user-input");
const searchBtn      = document.getElementById("search-btn");
const profileError   = document.getElementById("profile-error");
const profileContent = document.getElementById("profile-content");
const statRatings    = document.getElementById("stat-ratings");
const statAvg        = document.getElementById("stat-avg");
const statGenres     = document.getElementById("stat-genres");
const affinityBars   = document.getElementById("affinity-bars");

const recsList       = document.getElementById("recs-list");
const recsSpinner    = document.getElementById("recs-spinner");
const recsError      = document.getElementById("recs-error");
const recsEmpty      = document.getElementById("recs-empty");
const recsStageLbl   = document.getElementById("recs-stage-label");

const metricsSpinner = document.getElementById("metrics-spinner");
const metricsError   = document.getElementById("metrics-error");
const metricsContent = document.getElementById("metrics-content");
const ndcgBars       = document.getElementById("ndcg-bars");
const featureBars    = document.getElementById("feature-bars");

const btnBoth        = document.getElementById("btn-both");
const btnRetrieval   = document.getElementById("btn-retrieval");

// ── Helpers ──────────────────────────────────────────────

/** Escape special HTML chars — defensive even for trusted data. */
function esc(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function fmt3(n) {
  return Number(n).toFixed(3);
}

function pct(val, max) {
  if (!max) return 0;
  return Math.max(0, Math.min(100, (val / max) * 100));
}

function showElem(el)  { el.hidden = false; }
function hideElem(el)  { el.hidden = true;  }

function setError(el, msg) {
  el.textContent = msg;
  showElem(el);
}

function clearError(el) {
  el.textContent = "";
  hideElem(el);
}

/**
 * Render a list of { label, value } items as horizontal bar rows.
 * maxVal: the value that maps to 100% width.
 * barClass: extra CSS class on the .bar div (e.g. "green").
 */
function renderBars(container, items, maxVal, barClass) {
  container.innerHTML = items.map(({ label, value, extra }) => {
    const w = pct(value, maxVal).toFixed(1);
    const extraLabel = extra !== undefined ? ` <span class="bar-val">${esc(extra)}</span>` : "";
    return `
      <div class="bar-row">
        <div class="bar-label">
          <span>${esc(label)}</span>${extraLabel}
        </div>
        <div class="bar-track">
          <div class="bar ${barClass || ""}" style="width:${w}%"></div>
        </div>
      </div>`.trim();
  }).join("\n");
}

// ── Metrics ──────────────────────────────────────────────

async function loadMetrics() {
  showElem(metricsSpinner);
  hideElem(metricsContent);
  clearError(metricsError);

  try {
    const res = await fetch(API + "/metrics");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const models = data.models || {};
    const fi     = data.feature_importance || {};

    // NDCG@10 bars
    const ndcgKey = "ndcg@10";
    const ndcgItems = Object.entries(models)
      .map(([name, m]) => ({ label: name, value: m[ndcgKey] ?? 0 }));
    const ndcgMax = Math.max(...ndcgItems.map(x => x.value), 0.001);
    renderBars(ndcgBars, ndcgItems.map(x => ({
      ...x,
      extra: fmt3(x.value)
    })), ndcgMax, "");

    // Feature importance bars
    const fiItems = Object.entries(fi)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([name, w]) => ({ label: name, value: w, extra: fmt3(w) }));
    const fiMax = Math.max(...fiItems.map(x => x.value), 0.001);
    renderBars(featureBars, fiItems, fiMax, "green");

    hideElem(metricsSpinner);
    showElem(metricsContent);

    if (!ndcgItems.length && !fiItems.length) {
      setError(metricsError, "Metrics endpoint returned no data.");
    }
  } catch (err) {
    hideElem(metricsSpinner);
    setError(metricsError, "Could not load metrics: " + err.message);
  }
}

// ── User Profile ─────────────────────────────────────────

async function selectUser(id) {
  clearError(profileError);
  hideElem(profileContent);

  try {
    const res = await fetch(API + "/users/" + id);
    if (res.status === 404) {
      setError(profileError, "User " + id + " not found.");
      clearCenter();
      return;
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const u = await res.json();

    // Stats
    statRatings.textContent = u.n_ratings ?? "—";
    statAvg.textContent     = u.avg_rating != null
      ? "★ " + Number(u.avg_rating).toFixed(2)
      : "—";
    statGenres.textContent  = (u.top_genres || []).join(", ") || "—";

    // Affinity bars — top 10 nonzero, sorted desc
    const aff = u.genre_affinity || {};
    const affItems = Object.entries(aff)
      .filter(([, v]) => v > 0)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([genre, val]) => ({ label: genre, value: val, extra: fmt3(val) }));
    const affMax = Math.max(...affItems.map(x => x.value), 0.001);
    renderBars(affinityBars, affItems, affMax, "");

    showElem(profileContent);

    // Cache invalidate when user changes
    if (currentUserId !== id) {
      recsCache.both      = null;
      recsCache.retrieval = null;
    }
    currentUserId = id;
    await loadRecs(id, currentStage);
  } catch (err) {
    setError(profileError, "Error loading user: " + err.message);
  }
}

// ── Recommendations ──────────────────────────────────────

function clearCenter() {
  hideElem(recsList);
  hideElem(recsError);
  hideElem(recsSpinner);
  showElem(recsEmpty);
  recsStageLbl.textContent = "";
}

function positionMap(recommendations) {
  // movie_id → 0-based position
  const map = {};
  recommendations.forEach((r, i) => { map[r.movie_id] = i; });
  return map;
}

function deltaHtml(currentPos, otherPosMap, movieId) {
  if (otherPosMap === null || !(movieId in otherPosMap)) {
    return `<span class="rec-delta delta-same">–</span>`;
  }
  const diff = otherPosMap[movieId] - currentPos;  // positive = moved up
  if (diff === 0) return `<span class="rec-delta delta-same">–</span>`;
  if (diff > 0)   return `<span class="rec-delta delta-up">▲${diff}</span>`;
  return              `<span class="rec-delta delta-down">▼${Math.abs(diff)}</span>`;
}

async function loadRecs(id, stage) {
  clearError(recsError);
  hideElem(recsList);
  hideElem(recsEmpty);
  showElem(recsSpinner);
  recsStageLbl.textContent = stage === "both" ? "re-ranked" : "retrieval";

  try {
    // Use cache if available
    if (!recsCache[stage]) {
      const res = await fetch(API + "/recommend/" + id + "?k=10&stage=" + stage);
      if (res.status === 404) {
        hideElem(recsSpinner);
        setError(recsError, "User " + id + " not found for recommendations.");
        return;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      recsCache[stage] = await res.json();
    }

    const data  = recsCache[stage];
    const recs  = data.recommendations || [];
    const other = stage === "both" ? recsCache.retrieval : recsCache.both;
    const otherMap = other ? positionMap(other.recommendations || []) : null;

    recsList.innerHTML = recs.map((r, i) => {
      const genres   = (r.genres || []).join(" · ");
      const score    = fmt3(r.score);
      const delta    = deltaHtml(i, otherMap, r.movie_id);

      return `<li>
        <span class="rec-rank">${i + 1}</span>
        <div class="rec-info">
          <div class="rec-title">${esc(r.title)}</div>
          <div class="rec-genres">${esc(genres)}</div>
        </div>
        <div class="rec-right">
          <span class="rec-score">${esc(score)}</span>
          ${delta}
        </div>
      </li>`;
    }).join("\n");

    hideElem(recsSpinner);
    showElem(recsList);
  } catch (err) {
    hideElem(recsSpinner);
    setError(recsError, "Error loading recommendations: " + err.message);
  }
}

// ── Stage Toggle ─────────────────────────────────────────

function setStage(stage) {
  currentStage = stage;
  btnBoth.classList.toggle("active", stage === "both");
  btnRetrieval.classList.toggle("active", stage === "retrieval");

  if (currentUserId !== null) {
    loadRecs(currentUserId, stage);
  }
}

btnBoth.addEventListener("click",      () => setStage("both"));
btnRetrieval.addEventListener("click", () => setStage("retrieval"));

// ── Search ───────────────────────────────────────────────

function handleSearch() {
  const raw = userInput.value.trim();
  const id  = parseInt(raw, 10);
  if (!raw || isNaN(id) || id < 1) {
    setError(profileError, "Please enter a valid positive integer user ID.");
    return;
  }
  selectUser(id);
}

searchBtn.addEventListener("click", handleSearch);
userInput.addEventListener("keydown", e => {
  if (e.key === "Enter") handleSearch();
});

// ── Init ─────────────────────────────────────────────────

showElem(recsEmpty);
loadMetrics();
