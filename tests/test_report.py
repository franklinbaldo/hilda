# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""The report must render even when a family reaches no affordable point."""

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

FIELDS = [
    "encoder",
    "split",
    "depth",
    "n_probes",
    "budgeted",
    "recall",
    "recall_stderr",
    "scan_mean",
    "scan_p50",
    "scan_p95",
    "scan_max",
    "n_ranges",
    "budget_filled",
]


def _load_report() -> ModuleType:
    """Import the report script by path, since scripts/ is not a package."""
    path = Path(__file__).resolve().parent.parent / "scripts" / "report.py"
    spec = importlib.util.spec_from_file_location("report", path)
    module = importlib.util.module_from_spec(spec)
    # A dataclass resolves its own module through sys.modules while it is being
    # built, so the module has to be registered before it is executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(**overrides: object) -> dict[str, object]:
    """One CSV row, with sane defaults for everything not under test."""
    row = dict.fromkeys(FIELDS, 0)
    row.update(
        encoder="cheap",
        split="test",
        depth=1,
        n_probes=1,
        budgeted=0,
        recall=0.5,
        scan_mean=0.001,
        scan_p95=0.002,
        n_ranges=1,
        budget_filled=1.0,
    )
    row.update(overrides)
    return row


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    """Build a run where one family never fits any budget."""
    rows = [
        _row(split=split, encoder=encoder, scan_mean=scan)
        for split in ("validation", "test")
        for encoder, scan in (("cheap", 0.001), ("ruinous", 0.9))
    ]
    path = tmp_path / "ablation.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_a_family_with_no_affordable_point_renders_a_dash(csv_path: Path) -> None:
    """A family with no affordable point renders a dash."""
    report = _load_report()
    rendered = report.render(report.read_points(csv_path), notes={})
    assert "`ruinous`" in rendered
    assert "—" in rendered


def test_the_affordable_family_is_ranked_first(csv_path: Path) -> None:
    """The affordable family is ranked first."""
    report = _load_report()
    rendered = report.render(report.read_points(csv_path), notes={})
    assert rendered.index("`cheap`") < rendered.index("`ruinous`")
