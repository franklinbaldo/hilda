# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Tests for the memory-pressure protocol's pure parts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hilda_ablation.memory import ServerSpec, limits_for, resolve_binaries
from hilda_ablation.postgres_plans import exact_plan, hnsw_plan, render_vector

WORKING_SET = 1_000_000_000
FLOOR = 500_000


def test_limits_scale_with_the_working_set() -> None:
    """A ratio of the measured working set is what sets the cap."""
    limits = limits_for(WORKING_SET, [2.0, 0.5])
    assert [limit.limit_bytes for limit in limits] == [2_000_000_000, 500_000_000]
    assert all(not limit.clamped for limit in limits)


def test_a_cap_below_the_floor_is_clamped_and_says_so() -> None:
    """A corpus too small to pressure the server must not report a fake ratio."""
    (limit,) = limits_for(1_000_000, [0.1], minimum_bytes=FLOOR)
    assert limit.limit_bytes == FLOOR
    assert limit.clamped
    assert limit.achieved_ratio == pytest.approx(0.5)


def test_an_unclamped_cap_reports_the_ratio_it_was_asked_for() -> None:
    """Achieved and target agree whenever the floor did not intervene."""
    (limit,) = limits_for(WORKING_SET, [0.35])
    assert limit.achieved_ratio == pytest.approx(0.35)
    assert limit.label == "333MB"


@pytest.mark.parametrize("working_set", [0, -1])
def test_a_non_positive_working_set_is_rejected(working_set: int) -> None:
    """Dividing by an unmeasured working set would produce a meaningless ratio."""
    with pytest.raises(ValueError, match="working set"):
        limits_for(working_set, [1.0])


def test_a_non_positive_ratio_is_rejected() -> None:
    """A cap of zero bytes is not a condition, it is a broken sweep."""
    with pytest.raises(ValueError, match="ratios"):
        limits_for(WORKING_SET, [0.0])


def test_server_options_hold_every_fixed_setting() -> None:
    """Everything but the memory cap must be identical across conditions."""
    spec = ServerSpec(
        binaries=Path("/bin"),
        datadir=Path("/var/lib/none"),
        port=5434,
        shared_buffers_mb=32,
        settings=("-c jit=off",),
    )
    assert spec.options() == "-p 5434 -c shared_buffers=32MB -c jit=off"
    assert spec.dsn.endswith(":5434/postgres")


def test_resolve_binaries_rejects_a_directory_without_pg_ctl(tmp_path: Path) -> None:
    """A wrong binary path must fail loudly, not halfway through a sweep."""
    with pytest.raises(FileNotFoundError):
        resolve_binaries([tmp_path])


def test_the_exact_plan_disables_index_scans() -> None:
    """With HNSW on the table, only that switch keeps the exact plan exact."""
    plan = exact_plan()
    assert "SET enable_indexscan = off" in plan.settings


def test_the_hnsw_plan_carries_its_search_width() -> None:
    """The operating point has to reach the session, and the row label."""
    plan = hnsw_plan(40)
    assert plan.name == "pgvector-hnsw-ef40"
    assert "SET hnsw.ef_search = 40" in plan.settings


def test_render_vector_matches_pgvector_literal_syntax() -> None:
    """The COPY path and the query path must agree on the same text."""
    rendered = render_vector(np.array([0.5, -0.25], dtype=np.float32))
    assert rendered == "[0.500000,-0.250000]"
