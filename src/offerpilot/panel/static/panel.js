// Job text and model output are both untrusted here: the posting is scraped
// from the public web, and the brief was written by a model that read it.
// Every insertion below goes through el() -> textContent, so a posting that
// contains markup is displayed, never parsed.
//
// The markup-injecting DOM properties are banned outright in this file, and
// tests/test_panel.py enforces that by grepping the source for their names --
// so do not write them here, not even inside a comment explaining the ban.

const $ = (id) => document.getElementById(id);

function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = String(text);
  if (cls) node.className = cls;
  return node;
}

function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

function setStatus(message) { $("status").textContent = message; }

async function loadQueue() {
  const data = await (await fetch("/api/queue")).json();
  $("count").textContent = data.items.length;
  const list = $("queue-list");
  clear(list);
  for (const item of data.items) {
    const li = el("li");
    const btn = el("button", `${item.total_score}  ${item.title}`, "queue-btn");
    btn.addEventListener("click", () => loadItem(item.job_version_id));
    li.appendChild(btn);
    li.appendChild(el("div", `${item.company_id} · ${item.location || "—"}`, "muted"));
    list.appendChild(li);
  }
}

function scoreRow(match) {
  const wrap = el("div", null, "scores");
  const parts = [["skills", match.skills_score, 30],
                 ["projects", match.project_score, 20],
                 ["domain", match.domain_score, 15],
                 ["seniority", match.seniority_score, 15],
                 ["preferences", match.preference_score, 20]];
  for (const [name, got, max] of parts) {
    const cell = el("div", null, "score");
    cell.appendChild(el("strong", `${got}/${max}`));
    cell.appendChild(el("span", name, "muted"));
    wrap.appendChild(cell);
  }
  return wrap;
}

function list(title, items) {
  const box = el("div", null, "block");
  box.appendChild(el("h3", title));
  if (!items || items.length === 0) {
    box.appendChild(el("p", "none", "muted"));
    return box;
  }
  const ul = el("ul");
  for (const item of items) ul.appendChild(el("li", item));
  box.appendChild(ul);
  return box;
}

function evidenceBlock(evidence) {
  const box = el("div", null, "block");
  box.appendChild(el("h3", "Cited evidence"));
  if (!evidence || evidence.length === 0) {
    box.appendChild(el("p", "none", "muted"));
    return box;
  }
  for (const ref of evidence) {
    const card = el("div", null, "evidence");
    card.appendChild(el("code", ref.source_id));
    card.appendChild(el("p", ref.supporting_text));
    box.appendChild(card);
  }
  return box;
}

function externalLink(url) {
  // The posting's URL arrives inside the ATS payload. `canonicalize_url`
  // refuses a non-http(s) scheme at the collector boundary, but demo fixtures
  // build NormalizedJob directly and never pass through it, so this check is
  // load-bearing rather than belt-and-braces. An href is the last place in
  // this file where job data could still become executable, so only http(s)
  // becomes a link and anything else is shown as plain text.
  if (typeof url !== "string" || !/^https?:\/\//i.test(url)) {
    return el("p", url || "no link recorded", "muted");
  }
  const link = el("a", url);
  link.href = url;
  link.rel = "noopener noreferrer";
  link.target = "_blank";
  return link;
}

function briefEditor(versionId, brief) {
  const box = el("div", null, "block");
  box.appendChild(el("h3", "Application brief"));
  if (!brief) { box.appendChild(el("p", "no brief generated", "muted")); return box; }
  const area = el("textarea");
  area.value = JSON.stringify(brief, null, 2);
  area.rows = 16;
  const save = el("button", "Save edited brief");
  save.addEventListener("click", async () => {
    let parsed;
    try { parsed = JSON.parse(area.value); }
    catch (e) { setStatus("brief is not valid JSON"); return; }
    const res = await fetch(`/api/item/${versionId}/brief`, {
      method: "PUT", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({brief: parsed})});
    setStatus(res.ok ? "brief saved" : `brief rejected (${res.status})`);
  });
  box.appendChild(area);
  box.appendChild(save);
  return box;
}

function decisionBar(versionId) {
  const bar = el("div", null, "decisions");
  const fit = el("select");
  // The placeholder is appended first so it is the option the browser
  // preselects. Without it "good_fit" was option zero, so clicking Reject
  // without touching this dropdown wrote a good_fit label next to the
  // rejection -- and those rows are auxiliary signal the eval reads, so it
  // was corrupt data rather than a cosmetic default. An untouched select
  // sends null below; `Decision.fit_label` is Optional and stores NULL.
  fit.appendChild(new Option("(fit label)", ""));
  for (const v of ["good_fit", "uncertain", "poor_fit"]) fit.appendChild(new Option(v, v));
  const reason = el("select");
  reason.appendChild(new Option("(rejection reason)", ""));
  for (const v of ["skills", "seniority", "location", "compensation",
                   "duplicate", "expired", "not_interested", "bad_draft",
                   "other"]) reason.appendChild(new Option(v, v));
  const notes = el("input");
  notes.placeholder = "notes (optional)";
  bar.appendChild(el("label", "fit:"));
  bar.appendChild(fit);
  bar.appendChild(reason);
  bar.appendChild(notes);

  const send = async (action, actionLabel) => {
    // `|| null` and not `fit.value`: the placeholder's value is the empty
    // string, which is not in the FitLabel vocabulary and would come back 422.
    const body = {action, fit_label: fit.value || null, action_label: actionLabel,
                  notes: notes.value || null};
    if (action === "reject") {
      if (!reason.value) { setStatus("pick a rejection reason first"); return; }
      body.rejection_reason = reason.value;
    }
    const res = await fetch(`/api/item/${versionId}/decision`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body)});
    setStatus(res.ok ? `saved: ${action}` : `failed (${res.status})`);
    if (res.ok) { await loadQueue(); clear($("detail")); }
  };

  for (const [label, action, actionLabel] of [["Approve", "approve", "apply"],
                                              ["Save for later", "save", "save"],
                                              ["Reject", "reject", "skip"]]) {
    const b = el("button", label);
    b.addEventListener("click", () => send(action, actionLabel));
    bar.appendChild(b);
  }
  return bar;
}

async function loadItem(versionId) {
  const d = await (await fetch(`/api/item/${versionId}`)).json();
  const panel = $("detail");
  clear(panel);

  if (d.eligibility_unresolved) {
    panel.appendChild(el("div",
      "Eligibility unresolved — the model could not confirm you meet the hard "
      + "requirements. Check the posting yourself before applying.", "banner"));
  }
  panel.appendChild(el("h2", d.job.title));
  panel.appendChild(el("p",
    `${d.job.company_id} · ${d.job.location || "—"} · score ${d.total_score}/100`,
    "muted"));
  panel.appendChild(externalLink(d.job.url));
  panel.appendChild(scoreRow(d.match));
  panel.appendChild(evidenceBlock(d.match.evidence));
  panel.appendChild(list("Gaps", d.match.gaps));
  panel.appendChild(list("Uncertainties", d.match.uncertainties));
  panel.appendChild(briefEditor(versionId, d.brief));
  const posting = el("details");
  posting.appendChild(el("summary", "Full posting text"));
  posting.appendChild(el("pre", d.job.description_text));
  panel.appendChild(posting);
  panel.appendChild(decisionBar(versionId));
}

loadQueue();
