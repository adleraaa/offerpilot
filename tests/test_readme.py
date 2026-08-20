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

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

# Named in the README as a TODO for a human with a browser; capturing them
# needs a screen, so the repo ships without them rather than with dead links.
SCREENSHOTS = ("docs/images/panel.png", "docs/images/blind.png")

_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_PATH_SUFFIXES = (".py", ".yaml", ".yml", ".toml", ".md", ".json", ".png")


def _section(heading: str) -> str:
    start = README.index(heading)
    rest = README.index("\n## ", start + len(heading))
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


def test_project_layout_lists_only_paths_that_exist():
    """Only committed paths belong in the layout table.

    `config.yaml`, `profile.yaml` and `data/` are gitignored by design, so
    naming them here would fail this test on a clean clone -- describe them in
    the Real mode section instead.
    """
    assert "## Project layout" in README
    section = _section("## Project layout")
    paths = [t for t in _BACKTICK_RE.findall(section)
             if "/" in t or t.endswith(_PATH_SUFFIXES)]
    assert paths, "the project layout section lists no paths"
    missing = [p for p in paths if not (REPO_ROOT / p.rstrip("/")).exists()]
    assert missing == [], f"project layout names missing paths: {missing}"
