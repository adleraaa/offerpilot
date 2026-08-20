"""README truthfulness guards.

The plan's Task 10 lists this task's test as "manual: run every command block
in a clean shell". Manual checks rot the moment someone renames a file, so the
machine-checkable half of that check lives here: the README may only document
subcommands that exist, may only link files that exist, must keep demo mode
ahead of real mode (spec section "Demo mode"), and must describe the eval in
the words the spec fixes (spec section 5).

These read the README as text on purpose. There is no rendering step to hook
into, and a broken image or an invented flag is exactly the kind of defect that
only shows up on someone else's machine.
"""

import pathlib
import re
import subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

# Named in the README as a TODO for a human with a browser; capturing them
# needs a screen, so the repo ships without them rather than with dead links.
SCREENSHOTS = ("docs/images/panel.png", "docs/images/blind.png")

_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_PATH_SUFFIXES = (".py", ".yaml", ".yml", ".toml", ".md", ".json", ".png")


SPEC = "docs/superpowers/specs/2026-07-24-offerpilot-design.md"

# Subsystems the frozen spec's build-status banner lists as "Not built", each
# with the module that proves otherwise. The spec is frozen, so the README --
# not the banner -- is the lever when one of these ships.
_BANNER_CLAIMS = {
    "LangGraph orchestration": "src/offerpilot/graph.py",
    "the application brief node": "src/offerpilot/brief.py",
    "the review panel": "src/offerpilot/panel/app.py",
    "the blind-labeled evaluation set": "src/offerpilot/evaluate.py",
}
_STALENESS_WORDS = ("stale", "predates", "out of date", "outdated",
                    "before Week 2", "pre-Week-2")


def _section(heading: str) -> str:
    start = README.index(heading)
    try:
        rest = README.index("\n## ", start + len(heading))
    except ValueError:  # last section in the file
        rest = len(README)
    return README[start:rest]


def test_readme_has_no_broken_relative_links():
    """A dead link in a public README is worse than a missing one."""
    missing = []
    for target in _LINK_RE.findall(README):
        target = target.split("#", 1)[0].strip()
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        if not (REPO_ROOT / target).exists():
            missing.append(target)
    assert missing == [], f"README links files that do not exist: {missing}"


def test_readme_states_the_no_submission_boundary_in_bold():
    """The hard boundary is the first thing a reader must not miss."""
    bolded = re.findall(r"\*\*(.+?)\*\*", README, re.S)
    assert any("submitted to an employer" in b and "review queue" in b
               for b in bolded), "no bolded no-submission boundary in README"


def test_demo_mode_is_documented_before_real_mode():
    """Spec section "Demo mode": README documents demo first, real second."""
    for marker in ("## Try it (no API key)", "## How it works", "## Real mode"):
        assert marker in README, f"missing README section: {marker}"
    demo = README.index("python -m offerpilot demo")
    how = README.index("## How it works")
    real = README.index("## Real mode")
    assert demo < how < real, "demo quickstart must lead, real mode must follow"
    # The key-free path must also come before the first mention of the key.
    assert demo < README.index("DEEPSEEK_API_KEY")


def test_screenshots_are_named_as_a_todo_and_not_embedded_while_missing():
    for path in SCREENSHOTS:
        assert path in README, f"README does not name the screenshot {path}"
        if not (REPO_ROOT / path).exists():
            assert f"({path})" not in README, (
                f"{path} is embedded but does not exist")


def test_readme_only_documents_real_subcommands():
    cli_src = (REPO_ROOT / "src" / "offerpilot" / "cli.py").read_text(
        encoding="utf-8")
    match = re.search(r'"command",\s*choices=\[(.*?)\]', cli_src, re.S)
    assert match, "could not read the subcommand list out of cli.py"
    real = set(re.findall(r'"([a-z_]+)"', match.group(1)))
    documented = set(re.findall(r"python -m offerpilot ([a-z_]+)", README))
    assert documented, "README documents no subcommands at all"
    assert documented <= real, f"README invents subcommands: {documented - real}"


def test_readme_documents_every_subcommand():
    cli_src = (REPO_ROOT / "src" / "offerpilot" / "cli.py").read_text(
        encoding="utf-8")
    match = re.search(r'"command",\s*choices=\[(.*?)\]', cli_src, re.S)
    real = set(re.findall(r'"([a-z_]+)"', match.group(1)))
    documented = set(re.findall(r"python -m offerpilot ([a-z_]+)", README))
    assert real <= documented, f"undocumented subcommands: {real - documented}"


def test_readme_calls_the_eval_a_small_blind_labeled_set_never_a_benchmark():
    """Spec section 5 fixes this wording; 40-60 labels is not a benchmark."""
    assert "small blind-labeled evaluation set" in README
    assert "benchmark" not in README.lower()


def _committed_paths():
    """Every path git tracks, plus the directories they imply.

    Deliberately not `Path.exists()`. `config.yaml`, `profile.yaml` and
    `data/` are gitignored by design and are present in a working tree that
    has ever been used, so an existence check passes here and fails on the
    clean clone CI makes -- the one place where nobody sees it until the
    badge goes red. Asking git instead makes that failure local.

    Falls back to the filesystem when git is unavailable (a source tarball,
    say), which is the weaker check but never wrongly red.
    """
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO_ROOT,
                             capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return None
    tracked = {p for p in out.split("\0") if p}
    for f in list(tracked):
        parts = f.split("/")
        for i in range(1, len(parts)):
            tracked.add("/".join(parts[:i]))
    return tracked


def _layout_paths(section: str | None = None):
    """The path column of the layout table, and only the path column.

    The description column legitimately names gitignored files -- the
    `profile.py` row explains that it loads `profile.yaml` -- so scraping
    every backticked token out of the section turns prose into a path
    assertion and fails on a clean clone.
    """
    if section is None:
        section = _section("## Project layout")
    first_cells = re.findall(r"^\|\s*`([^`]+)`", section, re.M)
    return [t for t in first_cells if "/" in t or t.endswith(_PATH_SUFFIXES)]


def test_layout_scrape_reads_the_path_column_only():
    """Regression: a whole-row scrape reads prose as if it were a path.

    Synthetic on purpose -- reading the real row would re-couple this to the
    README's wording, and the property under test is about the scrape.
    """
    table = ("## Project layout\n\n"
             "| Path | What lives there |\n"
             "|---|---|\n"
             "| `src/offerpilot/profile.py` | `profile.yaml` loading |\n")
    assert _layout_paths(table) == ["src/offerpilot/profile.py"]


def test_project_layout_lists_only_paths_that_exist():
    """Only committed paths belong in the layout table.

    `config.yaml`, `profile.yaml` and `data/` are gitignored by design, so
    naming them in the path column would fail this test on a clean clone --
    describe them in the Real mode section instead.
    """
    assert "## Project layout" in README
    paths = _layout_paths()
    assert paths, "the project layout section lists no paths"
    committed = _committed_paths()
    if committed is None:  # pragma: no cover - git-less checkout
        missing = [p for p in paths if not (REPO_ROOT / p.rstrip("/")).exists()]
    else:
        missing = [p for p in paths if p.rstrip("/") not in committed]
    assert missing == [], f"project layout names missing paths: {missing}"


def _spec_not_built() -> str:
    """The `Not built:` sentence out of the frozen spec's status banner."""
    text = (REPO_ROOT / SPEC).read_text(encoding="utf-8")
    banner = re.sub(r"^> ?", "", text[:text.index("\n## ")], flags=re.M)
    return " ".join(banner.split("Not built:", 1)[1].split())


def test_readme_does_not_vouch_for_a_stale_spec_banner():
    """The README may point at the spec banner, but not certify it as current.

    The spec is frozen, and its banner was written before Week 2 shipped the
    graph, the brief node, the panel and the eval harness -- so it now lists
    built subsystems as absent. A reader who follows the README's pointer
    lands on a document that understates the build, which is the same defect
    as claiming something absent in the README itself, one indirection out.
    The lever available here is the README sentence, not the frozen file.
    """
    not_built = _spec_not_built()
    stale = sorted(claim for claim, path in _BANNER_CLAIMS.items()
                   if claim in not_built and (REPO_ROOT / path).exists())
    docs = _section("## Docs").lower()
    if not stale:
        # Banner refreshed: nothing left to warn about, so do not demand it.
        return
    assert any(w.lower() in docs for w in _STALENESS_WORDS), (
        f"the spec banner still lists {stale} as not built, but the README's "
        "Docs section does not say the banner is out of date")


def test_readme_test_count_matches_the_suite(request):
    """The README states a test count; nothing stopped it going stale.

    It said 280 while the suite was at 285. A number in a portfolio README is
    a claim like any other, so it is pinned here rather than trusted.
    Self-skips on a partial run, where the collected count is meaningless.
    """
    collected = request.session.testscollected
    if collected < 200:
        pytest.skip(f"partial run ({collected} collected); count is meaningless")
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    claimed = re.findall(r"(\d+) tests, all passing", text)
    assert claimed, "README no longer states a test count"
    assert int(claimed[0]) == collected, (
        f"README claims {claimed[0]} tests, suite has {collected}")
