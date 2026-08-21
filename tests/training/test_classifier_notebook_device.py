"""``evaluate`` must ask the model where it lives, not a module-level global.

The step-6.1 notebook trains on GPU, then the temperature-calibration cell does
``model_cpu = model.to("cpu")``. ``nn.Module.to`` moves **in place** and returns
the same object, so from that line on ``model`` is on the CPU too -- while the
global ``device`` still says ``cuda``. The test cell then failed with

    RuntimeError: Input type (torch.cuda.FloatTensor) and weight type
    (torch.FloatTensor) should be the same

after the whole training run had already finished. That is the expensive place
to fail.

This machine has no CUDA, which makes the bug reproducible exactly: the broken
version calls ``.to(device)`` with ``device = cuda`` and raises; the fixed one
never looks at the global at all.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "training" / "kaggle" / "pcb_classifier_v2_kaggle.py"
)


@pytest.fixture(scope="module")
def notebook_source() -> str:
    assert NOTEBOOK.is_file(), f"thiếu notebook {NOTEBOOK}"
    return NOTEBOOK.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def evaluate_fn(notebook_source: str):
    """The notebook's ``evaluate``, pulled out and given a hostile global.

    ``device`` is deliberately set to a CUDA device that does not exist here.
    Any code path that still reaches for it will raise, which is the point.
    """

    match = re.search(r"^def evaluate\(network, loader\):.*?(?=^\S)",
                      notebook_source, re.S | re.M)
    assert match, "không tìm thấy def evaluate trong notebook"
    namespace = {
        "torch": torch,
        "Counter": Counter,
        "device": torch.device("cuda"),
    }
    exec(match.group(0), namespace)  # noqa: S102 - our own file, read from disk
    return namespace["evaluate"]


def _cpu_model_and_loader():
    """A two-class linear model on the CPU and one deterministic batch."""

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    images = torch.zeros(6, 1, 2, 2)
    targets = torch.tensor([0, 1, 0, 1, 0, 1])
    return model, [(images, targets)]


def test_evaluate_runs_on_a_model_that_was_moved_to_cpu(evaluate_fn) -> None:
    """The exact failure: model on CPU, global device still saying cuda."""

    model, loader = _cpu_model_and_loader()
    accuracy, macro_recall = evaluate_fn(model, loader)

    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= macro_recall <= 1.0


def test_evaluate_never_reads_the_global_device_for_the_inputs(
    notebook_source: str,
) -> None:
    """Stated as a rule, not as one fixed call site: the inputs have to follow
    the weights, wherever the weights happen to be."""

    match = re.search(r"^def evaluate\(network, loader\):.*?(?=^\S)",
                      notebook_source, re.S | re.M)
    body = match.group(0)
    assert "images.to(device)" not in body, (
        "evaluate vẫn gửi ảnh tới biến device toàn cục"
    )
    assert "next(network.parameters()).device" in body


def test_the_notebook_warns_that_to_cpu_moves_in_place(notebook_source: str) -> None:
    """``model_cpu = model.to("cpu")`` reads like it makes a copy. It does not,
    and that is precisely what made the bug hard to see."""

    index = notebook_source.index('model_cpu = model.to("cpu")')
    preceding = notebook_source[max(0, index - 400):index]
    assert "TẠI CHỖ" in preceding or "in place" in preceding.lower(), (
        "cần ghi rõ .to() chuyển tại chỗ, ngay trên dòng gây ra chuyện"
    )
