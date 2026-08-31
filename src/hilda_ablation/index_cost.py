# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Pure accounting helpers for the scalar-vs-vector index cost frontier."""

from __future__ import annotations

from dataclasses import dataclass

MIB = 1024 * 1024


@dataclass(frozen=True)
class IndexCost:
    """Build, storage, and steady-state write cost for one index strategy."""

    strategy: str
    rows: int
    size_bytes: int
    build_seconds: float
    build_wal_bytes: int
    insert_rows: int
    insert_seconds: float
    insert_wal_bytes: int

    @property
    def bytes_per_row(self) -> float:
        """Physical index bytes divided by indexed rows."""
        return self.size_bytes / self.rows

    @property
    def insert_rows_per_second(self) -> float:
        """Observed append throughput while this strategy is maintained."""
        return self.insert_rows / self.insert_seconds

    @property
    def insert_wal_bytes_per_row(self) -> float:
        """WAL generated per appended row."""
        return self.insert_wal_bytes / self.insert_rows

    def as_row(self) -> dict[str, object]:
        """Flatten to a stable JSON representation."""
        return {
            "strategy": self.strategy,
            "rows": self.rows,
            "size_bytes": self.size_bytes,
            "size_mib": round(self.size_bytes / MIB, 3),
            "bytes_per_row": round(self.bytes_per_row, 3),
            "build_seconds": round(self.build_seconds, 3),
            "build_wal_bytes": self.build_wal_bytes,
            "insert_rows": self.insert_rows,
            "insert_seconds": round(self.insert_seconds, 3),
            "insert_rows_per_second": round(self.insert_rows_per_second, 3),
            "insert_wal_bytes": self.insert_wal_bytes,
            "insert_wal_bytes_per_row": round(self.insert_wal_bytes_per_row, 3),
        }


def relative_cost(numerator: IndexCost, denominator: IndexCost) -> dict[str, float]:
    """Return dimensionless cost ratios, numerator over denominator."""
    if numerator.rows != denominator.rows:
        message = "index cost ratios require the same base row count"
        raise ValueError(message)
    if numerator.insert_rows != denominator.insert_rows:
        message = "index cost ratios require the same insert batch size"
        raise ValueError(message)
    return {
        "storage_ratio": numerator.size_bytes / denominator.size_bytes,
        "build_time_ratio": numerator.build_seconds / denominator.build_seconds,
        "build_wal_ratio": numerator.build_wal_bytes / denominator.build_wal_bytes,
        "insert_time_ratio": numerator.insert_seconds / denominator.insert_seconds,
        "insert_wal_ratio": numerator.insert_wal_bytes / denominator.insert_wal_bytes,
    }
