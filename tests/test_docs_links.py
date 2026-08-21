"""Every relative link in the docs must point at a file that exists.

Written after a folder tidy-up, but the first thing it found was older damage:
``bao_cao_tom_tat.md`` linked to the full report through an absolute
``file:///E:/...`` path on somebody else's machine. That link had never worked
for anyone but its author, and nothing said so.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# [text](target) — markdown inline links
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _tracked_markdown() -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [ROOT / line for line in output.splitlines() if line]


def test_no_link_points_at_a_missing_file() -> None:
    broken: list[str] = []
    for path in _tracked_markdown():
        if not path.is_file():
            continue
        for target in LINK.findall(path.read_text(encoding="utf-8", errors="replace")):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            if target.startswith("file:///"):
                broken.append(
                    f"{path.relative_to(ROOT).as_posix()} -> {target} "
                    "(đường dẫn tuyệt đối trên máy người khác)"
                )
                continue
            resolved = (path.parent / target.split("#", 1)[0]).resolve()
            if not resolved.exists():
                broken.append(
                    f"{path.relative_to(ROOT).as_posix()} -> {target}"
                )
    assert broken == [], "link hỏng:\n  " + "\n  ".join(broken)


def test_the_repository_root_holds_no_loose_reports() -> None:
    """Reports belong in ``Docs/``. Three of them had accumulated at the root,
    one of them a stale copy named ``bao_cao_tien_do (1).md`` -- the ``(1)``
    that a browser adds when the same file is downloaded twice."""

    output = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    root_files = [line for line in output.splitlines() if "/" not in line]
    allowed = {
        ".gitignore",
        "AGENTS.md",
        "README.md",
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-model.txt",
        "requirements-train.txt",
    }
    unexpected = sorted(set(root_files) - allowed)
    assert unexpected == [], f"file lạ ở gốc repo: {unexpected}"


def test_tracked_paths_are_ascii() -> None:
    """A filename with Vietnamese diacritics makes git quote the path in every
    listing and breaks on filesystems that are not UTF-8. The content can be
    Vietnamese; the path should not have to be."""

    output = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    non_ascii = [line for line in output.splitlines() if not line.isascii()]
    assert non_ascii == [], f"đường dẫn không phải ASCII: {non_ascii}"
