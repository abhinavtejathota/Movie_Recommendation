(() => {
  const $ = (id) => document.getElementById(id);
  const statusEl = $("status");
  let abortController = null;
  let suggestTimer = null;
  let ready = false;

  function setStatus(text, cls) {
    statusEl.textContent = text;
    statusEl.className = `status ${cls}`;
  }

  function showMsg(el, text, kind) {
    el.hidden = !text;
    el.className = `msg ${kind || ""}`;
    el.innerHTML = text || "";
  }

  function renderTable(container, rows) {
    if (!rows || !rows.length) {
      container.innerHTML = "";
      return;
    }
    const cols = ["Title", "Genres", "Moods", "Rating", "Language", "Year"];
    const thead = cols.map((c) => `<th>${c}</th>`).join("");
    const body = rows
      .map(
        (r) =>
          `<tr>${cols
            .map((c) => `<td>${escapeHtml(r[c] ?? "")}</td>`)
            .join("")}</tr>`
      )
      .join("");
    container.innerHTML = `<table><thead><tr>${thead}</tr></thead><tbody>${body}</tbody></table>`;
  }

  function escapeHtml(v) {
    return String(v)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  async function api(path, options = {}) {
    if (abortController) abortController.abort();
    abortController = new AbortController();
    const res = await fetch(path, {
      ...options,
      signal: abortController.signal,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
    });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = body.detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }
    return res.json();
  }

  async function waitUntilReady() {
    setStatus("Starting…", "starting");
    for (let i = 0; i < 120; i++) {
      try {
        const h = await fetch("/api/health").then((r) => r.json());
        if (h.ready) {
          ready = true;
          setStatus("Ready", "ready");
          return true;
        }
        setStatus("Loading model…", "starting");
      } catch (_) {
        setStatus("Connecting…", "starting");
      }
      await new Promise((r) => setTimeout(r, 500));
    }
    setStatus("Failed to start", "error");
    return false;
  }

  async function loadInfo() {
    try {
      const info = await api("/api/info");
      const lang = $("language");
      lang.innerHTML = `<option value="Any">Any</option>` +
        (info.languages || [])
          .map((l) => `<option value="${escapeHtml(l)}">${escapeHtml(l)}</option>`)
          .join("");
      $("dataset-info").innerHTML =
        `Movies loaded: <strong>${info.movies.toLocaleString()}</strong><br>` +
        `Languages: ${escapeHtml((info.languages || []).join(", "))}<br>` +
        `Mood labels: ${escapeHtml((info.moods || []).join(", "))}`;
    } catch (err) {
      $("dataset-info").textContent = err.message;
    }
  }

  // Tabs
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      $(`panel-${btn.dataset.tab}`).classList.add("active");
    });
  });

  // Sliders
  $("count").addEventListener("input", (e) => {
    $("count-val").textContent = e.target.value;
  });
  $("min-rating").addEventListener("input", (e) => {
    $("rating-val").textContent = e.target.value;
  });
  $("sim-count").addEventListener("input", (e) => {
    $("sim-count-val").textContent = e.target.value;
  });

  // Mood recommend
  $("btn-mood").addEventListener("click", async () => {
    const mood = $("mood").value.trim();
    const msg = $("mood-msg");
    if (!mood) {
      showMsg(msg, "Enter a mood first.", "warn");
      return;
    }
    if (!ready) {
      showMsg(msg, "Model is still loading — try again in a moment.", "warn");
      return;
    }
    $("btn-mood").disabled = true;
    showMsg(msg, "Searching…", "warn");
    try {
      const data = await api("/api/recommend/mood", {
        method: "POST",
        body: JSON.stringify({
          mood,
          language: $("language").value,
          count: Number($("count").value),
          min_rating: Number($("min-rating").value),
        }),
      });
      if (data.cancelled) {
        showMsg(msg, "Request cancelled.", "warn");
        return;
      }
      if (!data.results.length) {
        showMsg(
          msg,
          `No matches for '${escapeHtml(mood)}'. Try happy, sad, excited, or another mood label.`,
          "warn"
        );
        renderTable($("mood-results"), []);
        return;
      }
      showMsg(
        msg,
        `Mapped <strong>${escapeHtml(mood)}</strong> → ${escapeHtml((data.mapped || []).join(", "))}`,
        "ok"
      );
      renderTable($("mood-results"), data.results);
    } catch (err) {
      if (err.name === "AbortError") return;
      showMsg(msg, err.message, "err");
    } finally {
      $("btn-mood").disabled = false;
    }
  });

  // Title suggestions (debounced, abortable)
  $("title").addEventListener("input", () => {
    const q = $("title").value.trim();
    clearTimeout(suggestTimer);
    if (!q || !ready) {
      $("suggestions").textContent = "";
      return;
    }
    suggestTimer = setTimeout(async () => {
      try {
        const data = await api(`/api/suggest?q=${encodeURIComponent(q)}&limit=8`);
        if (data.suggestions?.length) {
          $("suggestions").textContent = "Suggestions: " + data.suggestions.join(" · ");
        } else {
          $("suggestions").textContent = "";
        }
      } catch (_) {
        /* ignore aborted / transient */
      }
    }, 250);
  });

  $("btn-similar").addEventListener("click", async () => {
    const title = $("title").value.trim();
    const msg = $("similar-msg");
    if (!title) {
      showMsg(msg, "Enter a movie title.", "warn");
      return;
    }
    if (!ready) {
      showMsg(msg, "Model is still loading — try again in a moment.", "warn");
      return;
    }
    $("btn-similar").disabled = true;
    showMsg(msg, "Searching…", "warn");
    try {
      const data = await api("/api/recommend/similar", {
        method: "POST",
        body: JSON.stringify({
          title,
          count: Number($("sim-count").value),
        }),
      });
      if (!data.results.length) {
        showMsg(msg, `No movie found matching '${escapeHtml(title)}'.`, "warn");
        renderTable($("similar-results"), []);
        return;
      }
      showMsg(msg, "", "");
      msg.hidden = true;
      renderTable($("similar-results"), data.results);
    } catch (err) {
      if (err.name === "AbortError") return;
      showMsg(msg, err.message, "err");
    } finally {
      $("btn-similar").disabled = false;
    }
  });

  // Enter key shortcuts
  $("mood").addEventListener("keydown", (e) => {
    if (e.key === "Enter") $("btn-mood").click();
  });
  $("title").addEventListener("keydown", (e) => {
    if (e.key === "Enter") $("btn-similar").click();
  });

  (async () => {
    const ok = await waitUntilReady();
    if (ok) await loadInfo();
  })();
})();
