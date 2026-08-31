# Copyright (c) 2026 Franklin Baldo. See LICENSE.
"""Run PostgreSQL under a known total memory limit, warm or cold.

``shared_buffers`` is not a memory limit. PostgreSQL also reads through the
operating system's page cache, so an index far larger than ``shared_buffers``
can still be entirely resident and never touch a device. Measuring the
out-of-memory regime therefore needs two things this module provides: a cap on
*total* memory charged to the server's processes, and a cache state that is
established rather than assumed.

The cap is a cgroup the server is started inside, so every backend inherits it
and page cache read on their behalf is charged to it. The cold state is a full
stop, a global page-cache drop, and a fresh start -- dropping the page cache
alone would leave ``shared_buffers`` populated, which is a shared-buffer
experiment, not a memory-pressure one.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

logger = logging.getLogger("hilda_ablation.memory")

CGROUP_ROOT = Path("/sys/fs/cgroup")
DROP_CACHES = Path("/proc/sys/vm/drop_caches")
DROP_PAGECACHE_DENTRIES_INODES = "3"
"""What ``drop_caches`` wants for page cache plus reclaimable slab."""

MIN_LIMIT_BYTES = 192 * 1024 * 1024
"""Below this a server with a small ``shared_buffers`` stops making progress."""

STARTUP_TIMEOUT_SECONDS = 60.0
STARTUP_POLL_SECONDS = 0.2


@dataclass(frozen=True)
class MemoryLimit:
    """One point on the memory-pressure axis.

    ``target_ratio`` is what the experiment asked for as a share of the working
    set; ``achieved_ratio`` is what the floor allowed. They differ only when a
    corpus is small enough that the requested cap would starve the server
    rather than pressure it, and both are reported so the difference is never
    silently absorbed into the result.
    """

    target_ratio: float
    limit_bytes: int
    working_set_bytes: int

    @property
    def achieved_ratio(self) -> float:
        """The cap actually applied, as a share of the working set."""
        return self.limit_bytes / self.working_set_bytes

    @property
    def clamped(self) -> bool:
        """Whether the floor, not the target ratio, decided this cap."""
        return self.limit_bytes > round(self.target_ratio * self.working_set_bytes)

    @property
    def label(self) -> str:
        """A short name for tables and log lines."""
        return f"{self.limit_bytes // (1024 * 1024)}MB"


def limits_for(
    working_set_bytes: int,
    ratios: Sequence[float],
    minimum_bytes: int = MIN_LIMIT_BYTES,
) -> list[MemoryLimit]:
    """Turn ratios of a measured working set into memory caps.

    The working set is measured after loading rather than assumed, because the
    only ratio worth reporting is one against the bytes the plans actually have
    to reach.
    """
    if working_set_bytes <= 0:
        message = "working set must be positive"
        raise ValueError(message)
    if any(ratio <= 0 for ratio in ratios):
        message = "ratios must be positive"
        raise ValueError(message)
    return [
        MemoryLimit(
            target_ratio=ratio,
            limit_bytes=max(round(ratio * working_set_bytes), minimum_bytes),
            working_set_bytes=working_set_bytes,
        )
        for ratio in ratios
    ]


class CgroupUnavailableError(RuntimeError):
    """Raised when no writable memory cgroup interface exists.

    The experiment has no fallback: without a total-memory cap every number it
    would produce describes the resident regime that is already measured.
    """


@dataclass(frozen=True)
class _CgroupInterface:
    """Where a cgroup's limit and membership files live."""

    directory: Path
    limit_file: str
    procs_file: str


def _interface(name: str) -> _CgroupInterface:
    """Locate the memory cgroup interface, preferring v2 over v1."""
    if (CGROUP_ROOT / "cgroup.controllers").exists():
        return _CgroupInterface(CGROUP_ROOT / name, "memory.max", "cgroup.procs")
    if (CGROUP_ROOT / "memory").is_dir():
        return _CgroupInterface(
            CGROUP_ROOT / "memory" / name, "memory.limit_in_bytes", "cgroup.procs"
        )
    message = f"no memory cgroup interface under {CGROUP_ROOT}"
    raise CgroupUnavailableError(message)


class Cgroup:
    """A memory-capped cgroup that processes can be started inside.

    Membership is inherited, so putting a shell in the cgroup and starting the
    server from it caps the postmaster and every backend it forks.
    """

    def __init__(self, name: str) -> None:
        """Create the cgroup, or adopt it if a previous run left it behind."""
        self._interface = _interface(name)
        try:
            self._interface.directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            message = f"cannot create {self._interface.directory}: {exc}"
            raise CgroupUnavailableError(message) from exc

    @property
    def procs_path(self) -> Path:
        """The file a PID is written to in order to join."""
        return self._interface.directory / self._interface.procs_file

    def set_limit(self, limit_bytes: int) -> None:
        """Cap total memory charged to this cgroup."""
        path = self._interface.directory / self._interface.limit_file
        try:
            path.write_text(str(limit_bytes))
        except OSError as exc:
            message = f"cannot write {path}: {exc}"
            raise CgroupUnavailableError(message) from exc
        logger.info("memory cap set to %d bytes", limit_bytes)

    def destroy(self) -> None:
        """Remove the cgroup once it is empty."""
        try:
            self._interface.directory.rmdir()
        except OSError as exc:  # pragma: no cover - best-effort teardown
            logger.warning("could not remove %s: %s", self._interface.directory, exc)


def drop_page_cache() -> None:
    """Flush dirty pages and evict the page cache machine-wide.

    This is global, not per-cgroup: there is no interface that evicts only one
    cgroup's cache, and a partial eviction would leave the cold trial holding
    exactly the pages the previous plan happened to warm.
    """
    subprocess.run(["/bin/sync"], check=True)
    try:
        DROP_CACHES.write_text(DROP_PAGECACHE_DENTRIES_INODES)
    except OSError as exc:
        message = f"cannot write {DROP_CACHES}: {exc}"
        raise CgroupUnavailableError(message) from exc


@dataclass(frozen=True)
class ServerSpec:
    """Everything about the server that is held fixed across memory caps."""

    binaries: Path
    datadir: Path
    port: int
    shared_buffers_mb: int
    settings: tuple[str, ...] = ()
    user: str = "postgres"
    """PostgreSQL refuses to run as root, so every control binary drops to this."""

    @property
    def dsn(self) -> str:
        """A libpq connection string for this server."""
        return f"postgresql://postgres@127.0.0.1:{self.port}/postgres"

    def options(self) -> str:
        """Render the ``-o`` argument holding every fixed server setting."""
        fixed = (
            f"-p {self.port}",
            f"-c shared_buffers={self.shared_buffers_mb}MB",
            *self.settings,
        )
        return " ".join(fixed)


class PostgresServer:
    """Start and stop one PostgreSQL instance inside a memory cgroup."""

    def __init__(self, spec: ServerSpec, cgroup: Cgroup) -> None:
        """Bind a server specification to the cgroup that will cap it."""
        self._spec = spec
        self._cgroup = cgroup

    def _run(self, argv: Sequence[str], *, in_cgroup: bool, check: bool = True) -> None:
        """Run one control binary as the server user, optionally inside the cgroup.

        The cgroup is joined by the root shell before it drops privileges,
        because writing to ``cgroup.procs`` needs root while PostgreSQL refuses
        to start with it. Membership survives the transition, so the postmaster
        and every backend it forks are charged against the same cap.
        """
        inner = shlex.join(str(part) for part in argv)
        script = (
            f"exec su -s /bin/sh {shlex.quote(self._spec.user)} -c {shlex.quote(inner)}"
        )
        if in_cgroup:
            script = f'echo $$ > "{self._cgroup.procs_path}" && {script}'
        subprocess.run(["/bin/sh", "-c", script], check=check)  # noqa: S603

    def initdb(self) -> None:
        """Create a fresh data directory owned by the server user."""
        self._spec.datadir.parent.mkdir(parents=True, exist_ok=True)
        self._spec.datadir.mkdir(mode=0o700, exist_ok=True)
        shutil.chown(self._spec.datadir, user=self._spec.user, group=self._spec.user)
        self._run(
            [
                str(self._spec.binaries / "initdb"),
                "-D",
                str(self._spec.datadir),
                "-A",
                "trust",
                "-U",
                "postgres",
            ],
            in_cgroup=False,
        )

    def start(self) -> None:
        """Start the server inside the cgroup and wait for it to accept queries."""
        self._run(
            [
                str(self._spec.binaries / "pg_ctl"),
                "-D",
                str(self._spec.datadir),
                "-o",
                self._spec.options(),
                "-w",
                "-l",
                str(self._spec.datadir / "server.log"),
                "start",
            ],
            in_cgroup=True,
        )
        self._await_ready()

    def _await_ready(self) -> None:
        """Poll until the server answers, so no timing includes its startup."""
        ready = str(self._spec.binaries / "pg_isready")
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            probe = subprocess.run(  # noqa: S603
                [ready, "-h", "127.0.0.1", "-p", str(self._spec.port), "-q"],
                check=False,
            )
            if probe.returncode == 0:
                return
            time.sleep(STARTUP_POLL_SECONDS)
        log = self._spec.datadir / "server.log"
        tail = log.read_text()[-2000:] if log.exists() else "(no server log)"
        message = f"server on port {self._spec.port} did not become ready:\n{tail}"
        raise TimeoutError(message)

    def stop(self) -> None:
        """Stop the server, ignoring the case where it is already down."""
        if not (self._spec.datadir / "postmaster.pid").exists():
            return
        self._run(
            [
                str(self._spec.binaries / "pg_ctl"),
                "-D",
                str(self._spec.datadir),
                "-m",
                "fast",
                "-w",
                "stop",
            ],
            in_cgroup=False,
            check=False,
        )

    def restart_cold(self) -> None:
        """Bring the server up with neither shared buffers nor page cache warm."""
        self.stop()
        drop_page_cache()
        self.start()

    def restart(self) -> None:
        """Bring the server up again, keeping the page cache as it is."""
        self.stop()
        self.start()


def resolve_binaries(candidates: Iterable[Path]) -> Path:
    """Pick the first directory that holds a ``pg_ctl``."""
    for candidate in candidates:
        if (candidate / "pg_ctl").exists():
            return candidate
    message = "no PostgreSQL binary directory found"
    raise FileNotFoundError(message)


def available_bytes() -> int:
    """Total memory the machine reports, for the record the protocol requires."""
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
