// The blind labeling page. The server sends the posting and the profile and
// nothing the model produced, so there is nothing here to accidentally show
// -- but the posting is still scraped from the public web, so every insertion
// goes through el() -> textContent, exactly as in panel.js.
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

function renderProfile(summary) {
  const box = $("profile-body");
  clear(box);
  box.appendChild(el("p", summary.identity.education));
  box.appendChild(el("p", `graduating ${summary.identity.graduation}`, "muted"));
  const ul = el("ul");
  for (const exp of summary.experiences) {
    const li = el("li");
    li.appendChild(el("strong", exp.title));
    li.appendChild(el("div", exp.summary, "muted"));
    ul.appendChild(li);
  }
  box.appendChild(ul);
}

async function next() {
  const data = await (await fetch("/api/blind/next")).json();
  renderProfile(data.profile_summary);
  const progress = await (await fetch("/api/blind/progress")).json();
  $("progress").textContent =
    `${progress.labeled} of ${progress.total} labeled · target `
    + `${progress.target_min}-${progress.target_max}`;

  const body = $("job-body");
  clear(body);
  if (!data.job) {
    body.appendChild(el("h2", "Nothing left to label."));
    return;
  }
  body.appendChild(el("h2", data.job.title));
  body.appendChild(el("p", `${data.job.company_id} · ${data.job.location || "—"}`, "muted"));
  // Shown as text, never as a link: the posting URL comes from the ATS
  // payload with whatever scheme it carried, and this page has no reason to
  // navigate anywhere.
  body.appendChild(el("p", data.job.url, "muted"));
  body.appendChild(el("pre", data.job.description_text));

  const bar = el("div", null, "decisions");
  // One job version gets one blind label, and the server returns 409 for a
  // second. That backstop must not be what a normal double-click hits: the
  // POST is awaited, and `next()` only redraws afterwards, so without this
  // the other two verdict buttons stay live for the whole round trip and
  // "good fit" then "poor fit" both land for the same job.
  const buttons = [];
  let posting = false;
  for (const fit of ["good_fit", "uncertain", "poor_fit"]) {
    const b = el("button", fit.replace("_", " "));
    b.addEventListener("click", async () => {
      if (posting) return;
      posting = true;
      for (const other of buttons) other.disabled = true;
      try {
        const res = await fetch(`/api/blind/${data.job.job_version_id}/label`, {
          method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({fit_label: fit})});
        if (res.ok) { next(); return; }
        // Say so rather than looking dead: a 409 here means this version was
        // already labeled, so the queue is stale and a reload fixes it.
        $("progress").textContent = `label rejected (HTTP ${res.status})`;
      } finally {
        posting = false;
        for (const other of buttons) other.disabled = false;
      }
    });
    buttons.push(b);
    bar.appendChild(b);
  }
  body.appendChild(bar);
}

next();
