# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Memory-pressure evidence helpers."""

from pathlib import Path

from hilda_ablation.memory import MemoryEvidence, pressure_delta, read_memory_evidence


def test_read_memory_evidence_parses_cgroup_v2(tmp_path: Path) -> None:
    """The helper records the total cgroup boundary, usage and counters."""
    (tmp_path / "memory.max").write_text("536870912\n")
    (tmp_path / "memory.current").write_text("123456\n")
    (tmp_path / "memory.peak").write_text("234567\n")
    (tmp_path / "memory.events").write_text("low 0\nhigh 3\nmax 5\noom 1\noom_kill 0\n")
    (tmp_path / "memory.stat").write_text(
        "workingset_refault_file 9\npgmajfault 4\n"
    )

    evidence = read_memory_evidence(tmp_path)

    assert evidence.limit_bytes == 536_870_912
    assert evidence.current_bytes == 123_456
    assert evidence.peak_bytes == 234_567
    assert evidence.events["high"] == 3
    assert evidence.stat["workingset_refault_file"] == 9


def test_unlimited_or_missing_values_are_not_claimed_as_limits(tmp_path: Path) -> None:
    """An unlimited cgroup must not masquerade as a controlled-memory run."""
    (tmp_path / "memory.max").write_text("max\n")

    evidence = read_memory_evidence(tmp_path)

    assert evidence.limit_bytes is None
    assert evidence.current_bytes is None
    assert evidence.events == {}


def test_pressure_delta_reports_only_increases() -> None:
    """Before/after snapshots expose refault and memory-limit pressure."""
    before = MemoryEvidence(
        limit_bytes=100,
        current_bytes=50,
        peak_bytes=60,
        events={"high": 1, "max": 2, "oom": 0},
        stat={"workingset_refault_file": 10, "pgmajfault": 3},
    )
    after = MemoryEvidence(
        limit_bytes=100,
        current_bytes=80,
        peak_bytes=95,
        events={"high": 4, "max": 7, "oom": 0},
        stat={"workingset_refault_file": 17, "pgmajfault": 5},
    )

    delta = pressure_delta(before, after)

    assert delta["events.high"] == 3
    assert delta["events.max"] == 5
    assert delta["stat.workingset_refault_file"] == 7
    assert delta["stat.pgmajfault"] == 2
    assert delta["events.oom"] == 0
