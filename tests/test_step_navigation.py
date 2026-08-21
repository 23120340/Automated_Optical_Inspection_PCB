"""Every navigable step must actually be renderable.

The step list, the per-step status map, the reset map and the renderer table are
four separate structures that all have to agree. They drifted the moment step
6.2 was added to the navigation only: ``statuses`` was built from a literal
``range(7)``, so the sidebar raised ``KeyError: 7`` on the very first render and
the whole app died before showing anything.

These tests compare the four against each other rather than against a hard-coded
count, so adding a step 8 later cannot reintroduce the same class of bug.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

import app.streamlit_app as ui


def _step_indices() -> list[int]:
    return [step for step, *_ in ui.STEP_DEFINITIONS]


def test_step_indices_are_unique_and_contiguous_from_zero() -> None:
    indices = _step_indices()
    assert indices == sorted(indices)
    assert len(set(indices)) == len(indices)
    assert indices == list(range(len(indices)))


@pytest.fixture(scope="module")
def rendered_app():
    """The app after one real render, so these assertions test the running
    thing rather than a re-implementation of its initialisation."""

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(ui.__file__)), default_timeout=120)
    app.run()
    assert not app.exception, [str(e.value) for e in app.exception]
    return app


def test_the_default_status_map_covers_every_navigable_step(rendered_app) -> None:
    """The exact failure seen in the app: KeyError: 7 raised by the sidebar
    before a single widget was drawn."""

    statuses = rendered_app.session_state["statuses"]
    missing = [step for step in _step_indices() if step not in statuses]
    assert missing == [], f"bước không có status mặc định: {missing}"


def test_init_state_declares_every_session_key_the_ui_reads() -> None:
    """Streamlit does not invent attributes; an undeclared key is an
    AttributeError the first time a tab touches it. Step 6.2 needed eight of
    them (``solder_result``, the model/manifest triples, ``cad_summary``) and
    none were declared when the section was first wired up."""

    source = Path(ui.__file__).read_text(encoding="utf-8")
    used = set(re.findall(r"st\.session_state\.([a-z][a-z0-9_]*)", source))
    declared = set(re.findall(r'^\s*"([a-z][a-z0-9_]*)":', inspect.getsource(ui._init_state), re.M))
    # Widget-bound keys are created by Streamlit on render, not by _init_state.
    widget_keys = set(re.findall(r'key="([a-z][a-z0-9_]*)"', source))
    missing = sorted(used - declared - widget_keys)
    assert missing == [], f"key chưa khai báo trong _init_state: {missing}"


def test_every_step_has_a_renderer() -> None:
    source = inspect.getsource(ui.main)
    for step in _step_indices():
        assert f"{step}: _render_step_" in source, f"bước {step} không có renderer"


def test_resetting_a_step_invalidates_every_later_step() -> None:
    """``_invalidate_after`` walks the step list; a literal bound would leave
    the last step holding stale results from a previous board."""

    source = inspect.getsource(ui._invalidate_after)
    assert "len(STEP_DEFINITIONS)" in source, (
        "_invalidate_after phải chạy tới hết STEP_DEFINITIONS, không dùng số cứng"
    )
    covered = {int(index) for index in re.findall(r"^\s*(\d+): \"\w+\",", source, re.M)}
    missing = [step for step in _step_indices() if step not in covered]
    assert missing == [], f"bước không có result_key khi reset: {missing}"
