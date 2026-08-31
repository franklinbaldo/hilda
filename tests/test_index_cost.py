# Copyright (c) 2026 Franklin Baldo. See LICENSE.

from __future__ import annotations

import pytest

from hilda_ablation.index_cost import IndexCost, relative_cost


def _cost(
    strategy: str,
    *,
    size_bytes: int,
    build_seconds: float,
    insert_seconds: float,
) -> IndexCost:
    return IndexCost(
        strategy=strategy,
        rows=100,
        size_bytes=size_bytes,
        build_seconds=build_seconds,
        insert_rows=10,
        insert_seconds=insert_seconds,
    )


def test_index_cost_derived_metrics() -> None:
    cost = _cost(
        "btree",
        size_bytes=1_000,
        build_seconds=2.0,
        insert_seconds=0.5,
    )

    assert cost.bytes_per_row == 10.0
    assert cost.insert_rows_per_second == 20.0
    assert cost.as_row()["bytes_per_row"] == 10.0


def test_relative_cost_reports_dimensionless_ratios() -> None:
    hnsw = _cost(
        "hnsw",
        size_bytes=10_000,
        build_seconds=8.0,
        insert_seconds=2.0,
    )
    btree = _cost(
        "btree",
        size_bytes=1_000,
        build_seconds=2.0,
        insert_seconds=0.5,
    )

    ratios = relative_cost(hnsw, btree)

    assert ratios == {
        "storage_ratio": 10.0,
        "build_time_ratio": 4.0,
        "insert_time_ratio": 4.0,
    }


def test_relative_cost_rejects_mismatched_rows() -> None:
    left = _cost("left", size_bytes=1_000, build_seconds=1.0, insert_seconds=1.0)
    right = IndexCost(
        strategy="right",
        rows=101,
        size_bytes=1_000,
        build_seconds=1.0,
        insert_rows=10,
        insert_seconds=1.0,
    )

    with pytest.raises(ValueError, match="same base row count"):
        relative_cost(left, right)


def test_relative_cost_rejects_mismatched_insert_batch() -> None:
    left = _cost("left", size_bytes=1_000, build_seconds=1.0, insert_seconds=1.0)
    right = IndexCost(
        strategy="right",
        rows=100,
        size_bytes=1_000,
        build_seconds=1.0,
        insert_rows=11,
        insert_seconds=1.0,
    )

    with pytest.raises(ValueError, match="same insert batch size"):
        relative_cost(left, right)
