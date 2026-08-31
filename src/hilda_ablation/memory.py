# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Read the memory boundary that makes a pressure benchmark falsifiable."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

CGROUP_V2 = Path("/sys/fs/cgroup")
UNLIMITED = "max"


@dataclass(frozen=True)
class MemoryEvidence:
    """Cgroup-v2 memory state captured next to one benchmark run."""

    limit_bytes: int | None
    current_bytes: int | None
    peak_bytes: int | None
    events: dict[str, int]
    stat: dict[str, int]

    def as_row(self) -> dict[str, object]:
        """Flatten to a JSON-friendly mapping."""
        return asdict(self)


def _read_int(path: Path) -> int | None:
    """Read an integer cgroup file, returning ``None`` when unavailable."""
    try:
        value = path.read_text().strip()
    except OSError:
        return None
    if value == UNLIMITED:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _read_pairs(path: Path) -> dict[str, int]:
    """Read whitespace-separated ``name value`` counters from cgroup v2."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    pairs: dict[str, int] = {}
    for line in lines:
        fields = line.split()
        if len(fields) != 2:
            continue
        name, raw = fields
        try:
            pairs[name] = int(raw)
        except ValueError:
            continue
    return pairs


def read_memory_evidence(root: Path = CGROUP_V2) -> MemoryEvidence:
    """Capture cgroup-v2 memory limit, usage, peak and pressure counters."""
    return MemoryEvidence(
        limit_bytes=_read_int(root / "memory.max"),
        current_bytes=_read_int(root / "memory.current"),
        peak_bytes=_read_int(root / "memory.peak"),
        events=_read_pairs(root / "memory.events"),
        stat=_read_pairs(root / "memory.stat"),
    )


def pressure_delta(before: MemoryEvidence, after: MemoryEvidence) -> dict[str, int]:
    """Return counter deltas useful for proving eviction/refault pressure."""
    keys = {
        "events.high": (before.events, after.events, "high"),
        "events.max": (before.events, after.events, "max"),
        "events.oom": (before.events, after.events, "oom"),
        "events.oom_kill": (before.events, after.events, "oom_kill"),
        "stat.workingset_refault_anon": (
            before.stat,
            after.stat,
            "workingset_refault_anon",
        ),
        "stat.workingset_refault_file": (
            before.stat,
            after.stat,
            "workingset_refault_file",
        ),
        "stat.pgmajfault": (before.stat, after.stat, "pgmajfault"),
    }
    return {
        label: max(0, newer.get(key, 0) - older.get(key, 0))
        for label, (older, newer, key) in keys.items()
    }
