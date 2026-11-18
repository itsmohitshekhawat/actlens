// ActLens demo page: streams the agent's steps (Server-Sent Events), renders the answer with clickable citations.
const $ = (id) => document.getElementById(id);
const EXAMPLES = [
  "Can my company use emotion recognition on employees?",
  "Is an AI tool that screens CVs high-risk?",
  "What is the maximum fine for a prohibited AI practice?",
  "Must a chatbot say it is an AI?",
  "When is a general-purpose AI model 'systemic risk'?",
];
let sources = {}, check = {}, busy = false;

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Minimal, safe Markdown: escape everything first, then bold, bullet lists, paragraphs, citations.
function render(md) {
  const verified = new Map((check.citations || []).map((c) => [c.citation, c.verified]));
  const inline = (t) => esc(t)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[((?:Art\.|Article|Annex|Recital)[^\]]{0,40})\]/g, (m, label) => {
      const parts = label.split(/;\s*/);
      return parts.map((p) => {
        const raw = p.replace(/&amp;/g, "&");
        const ok = verified.get(raw) !== false;
        return `<button class="cite${ok ? "" : " bad"}" data-c="${p}" title="${ok ? "Verified: the agent read this passage" : "Unverified: not in the passages the agent read"}">${p}</button>`;
      }).join("");
    });
  const out = [];
  let list = [];
  const flush = () => { if (list.length) { out.push("<ul>" + list.map((li) => `<li>${inline(li)}</li>`).join("") + "</ul>"); list = []; } };
  for (const block of md.split(/\n{2,}/)) {
    const lines = block.split("\n");
    if (lines.every((l) => /^\s*[-*•]\s+/.test(l))) { lines.forEach((l) => list.push(l.replace(/^\s*[-*•]\s+/, ""))); continue; }
    flush();
    out.push(`<p>${lines.map(inline).join("<br>")}</p>`);
  }
  flush();
  return out.join("");
}

function showSource(label) {
  const raw = label.replace(/&amp;/g, "&");
  const s = sources[raw];
  $("source").innerHTML = s
    ? `<div class="source"><b>${esc(s.label)} · official text</b>${esc(s.text)}${s.text.length >= 1200 ? "…" : ""}</div>`
    : `<div class="source"><b>${esc(raw)}</b>This citation does not match any passage the agent read.</div>`;
}

$("answer").addEventListener("click", (e) => { const b = e.target.closest(".cite"); if (b) showSource(b.dataset.c); });

function traceItem(html) { const li = document.createElement("li"); li.innerHTML = html; $("trace").appendChild(li); return li; }

async function quota() {
  try {
    const s = await (await fetch("/api/status")).json();
    $("quota").textContent = s.open ? `${s.your_questions_left} questions left for you today.` : "The free daily limit is reached. Search and the examples below still work.";
  } catch { /* ignore */ }
}

function ask(q) {
  if (busy || !q.trim()) return;
  busy = true; $("go").disabled = true; sources = {}; check = {};
  $("q").value = q;
  $("answer").innerHTML = '<p class="empty"><span class="spinner"></span>The agent is searching the AI Act…</p>';
  $("source").innerHTML = ""; $("stats").innerHTML = ""; $("trace").innerHTML = "";
  let pending = null;
  const es = new EventSource("/api/ask?q=" + encodeURIComponent(q));
  const finish = () => { es.close(); busy = false; $("go").disabled = false; if (pending) pending.querySelector(".spinner")?.remove(); quota(); };
  es.addEventListener("thought", (e) => { const d = JSON.parse(e.data); traceItem(`<span class="found">${esc(d.text)}</span>`); });
  es.addEventListener("tool_call", (e) => {
    const d = JSON.parse(e.data);
    const arg = d.tool === "search_law" ? d.args.query : d.args.ref;
    pending = traceItem(`<span class="tool"><span class="spinner"></span>${d.tool === "search_law" ? "search" : "read"}</span> <span class="q">${esc(arg || "")}</span>`);
  });
  es.addEventListener("tool_result", (e) => {
    const d = JSON.parse(e.data);
    if (pending) {
      pending.querySelector(".spinner")?.remove();
      const labels = d.found.map((f) => f.label);
      pending.insertAdjacentHTML("beforeend", `<div class="found">${labels.length ? "found " + esc(labels.join(", ")) : "nothing found"}</div>`);
    }
  });
  es.addEventListener("answer", (e) => {
    const d = JSON.parse(e.data);
    sources = d.sources || {}; check = d.check || {};
    $("answer").innerHTML = render(d.text);
    const c = check;
    $("stats").innerHTML = c.total
      ? `<span class="stat ${c.grounded ? "ok" : "warn"}">${c.verified}/${c.total} citations verified</span>`
      : `<span class="stat warn">no citations</span>`;
  });
  es.addEventListener("done", (e) => {
    const d = JSON.parse(e.data);
    $("stats").insertAdjacentHTML("beforeend",
      `<span class="stat">${d.steps} model calls</span><span class="stat">${d.seconds}s</span><span class="stat">${(d.tokens_in + d.tokens_out).toLocaleString()} tokens</span><span class="stat">$${d.cost_usd.toFixed(4)}</span>`);
    traceItem('<span class="tool">answer written</span><div class="found">citations checked against what was read</div>');
    finish();
  });
  es.addEventListener("limit", () => { $("answer").innerHTML = '<p class="empty">The free daily limit for this demo is reached. Try the search comparison below, or come back tomorrow.</p>'; finish(); });
  es.addEventListener("failed", () => { $("answer").innerHTML = '<p class="empty">The AI model did not answer in time. Please try again.</p>'; finish(); });
  es.onerror = () => { if (busy) { $("answer").innerHTML = '<p class="empty">Connection lost. Please try again.</p>'; finish(); } };
}

$("ask").addEventListener("submit", (e) => { e.preventDefault(); ask($("q").value); });
$("chips").innerHTML = EXAMPLES.map((x) => `<button class="chip" type="button">${esc(x)}</button>`).join("");
$("chips").addEventListener("click", (e) => { const b = e.target.closest(".chip"); if (b) ask(b.textContent); });

async function compare(q) {
  const modes = [["keyword", "Keyword (BM25)"], ["semantic", "Meaning (embeddings)"], ["hybrid", "Hybrid (RRF)"]];
  const results = await Promise.all(modes.map(([m]) => fetch(`/api/search?k=5&mode=${m}&q=${encodeURIComponent(q)}`).then((r) => r.json())));
  $("cmpres").innerHTML = results.map((r, i) => `<div class="card"><h2>${modes[i][1]}</h2><ol class="res">${
    (r.results || []).map((h) => `<li><code>${esc(h.label)}</code> ${esc(h.title || h.text.slice(0, 60))}</li>`).join("")}</ol></div>`).join("");
}
$("cmp").addEventListener("submit", (e) => { e.preventDefault(); compare($("cq").value); });
compare($("cq").value);
quota();
