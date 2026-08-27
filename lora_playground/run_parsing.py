"""Neutral, single-file JSONL parsing for run logs.

This boundary records only events that are physically present in one log
file.  It does not parse launcher commands, fill missing defaults, merge resume
siblings, consult manifests, or apply exclusions.  Plotting's legacy loader may
normalize the returned data afterward; the run catalog consumes it directly.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .run_records import freeze_value, thaw_value


@dataclass(frozen=True, slots=True)
class ParsedRunFile:
    """Events physically recorded in one JSONL task-output file."""

    config: Mapping[str, Any] | None
    evals: tuple[Mapping[str, Any], ...]
    optim_steps: tuple[Mapping[str, Any], ...]
    optim_steps_included: bool
    resume_event: Mapping[str, Any] | None
    abort_event: Mapping[str, Any] | None
    init_override_event: Mapping[str, Any] | None
    log_filename: str

    def raw_config(self) -> dict[str, Any] | None:
        """Mutable raw config plus parser-owned audit event attachments."""
        if self.config is None:
            return None
        cfg = thaw_value(self.config)
        cfg["_log_filename"] = self.log_filename
        if self.optim_steps_included:
            cfg["_optim_steps"] = thaw_value(self.optim_steps)
        if self.resume_event is not None:
            cfg["_resume"] = thaw_value(self.resume_event)
        if self.abort_event is not None:
            cfg["_aborted"] = thaw_value(self.abort_event)
        if self.init_override_event is not None:
            cfg["_lora_init_override"] = thaw_value(self.init_override_event)
        return cfg

    def mutable_evals(self) -> list[dict[str, Any]]:
        return thaw_value(self.evals)


@dataclass(frozen=True, slots=True)
class ParsedRunHeader:
    """The producer-recorded config event without the run history."""

    config: Mapping[str, Any] | None
    resume_event: Mapping[str, Any] | None
    log_filename: str

    def raw_config(self) -> dict[str, Any] | None:
        if self.config is None:
            return None
        config = thaw_value(self.config)
        config["_log_filename"] = self.log_filename
        if self.resume_event is not None:
            config["_resume"] = thaw_value(self.resume_event)
        return config


_PARSE_CACHE: dict[tuple[str, bool], tuple[tuple[int, int], ParsedRunFile]] = {}
_HEADER_CACHE: dict[str, tuple[tuple[int, int], ParsedRunHeader]] = {}
_LEADING_OPTIM_STEP = re.compile(
    r'^\s*\{\s*"event"\s*:\s*"optim_step"\s*[,}]'
)


def _file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def parse_run_header(log_path: str | Path) -> ParsedRunHeader:
    """Read through the unique startup config event, not the full JSONL.

    Training emits exactly one ``config`` event before eval/diagnostic events.
    Non-JSON launcher preamble is tolerated. The returned config is suitable
    only for conservative query rejection; matching files are still parsed by
    :func:`parse_run_file` before they become records.
    """
    path = Path(log_path)
    path_key = str(path)
    signature = _file_signature(path)
    cached = _HEADER_CACHE.get(path_key) if signature is not None else None
    if cached is not None and cached[0] == signature:
        return cached[1]

    config = None
    resume_event = None
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event_type = event.get("event")
            if event_type == "config" and config is None:
                config = event
                continue
            if event_type == "resume" and config is not None:
                resume_event = event
                continue
            if config is not None:
                # Resume metadata is a startup event. The first ordinary run
                # event seals the header and avoids scanning diagnostics.
                break

    parsed = ParsedRunHeader(
        config=None if config is None else freeze_value(config),
        resume_event=(
            None if resume_event is None else freeze_value(resume_event)
        ),
        log_filename=path.name,
    )
    if signature is not None:
        _HEADER_CACHE[path_key] = (signature, parsed)
    return parsed


def parse_run_file(
    log_path: str | Path,
    *,
    include_optim_steps: bool = True,
) -> ParsedRunFile:
    """Parse one physical log without semantic reconstruction or stitching.

    ``include_optim_steps=False`` is the records/plotting fast path. It skips
    producer-canonical diagnostic lines whose leading JSON field identifies an
    ``optim_step`` event; unusual key orders still take the normal JSON path.
    Config, eval, resume, abort, and init-override events are unchanged.
    """
    path = Path(log_path)
    path_key = str(path)
    signature = _file_signature(path)
    cache_key = (path_key, include_optim_steps)
    cached = _PARSE_CACHE.get(cache_key) if signature is not None else None
    if cached is not None and cached[0] == signature:
        return cached[1]

    config = None
    evals = []
    optim_steps = []
    resume_event = None
    abort_event = None
    init_override_event = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not include_optim_steps and _LEADING_OPTIM_STEP.match(line):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("event")
        if event_type == "config" and config is None:
            # The producer emits one startup config. Treat that first event as
            # authoritative so header screening and full parsing cannot
            # disagree on malformed files containing a later duplicate.
            config = event
        elif event_type == "eval":
            evals.append(event)
        elif event_type == "optim_step":
            if include_optim_steps:
                optim_steps.append(event)
        elif event_type == "resume":
            resume_event = event
        elif event_type == "lora_init_override":
            init_override_event = event
        elif isinstance(event_type, str) and event_type.startswith("abort_on_"):
            abort_event = event

    parsed = ParsedRunFile(
        config=None if config is None else freeze_value(config),
        evals=tuple(freeze_value(event) for event in evals),
        optim_steps=tuple(freeze_value(event) for event in optim_steps),
        optim_steps_included=include_optim_steps,
        resume_event=(None if resume_event is None
                      else freeze_value(resume_event)),
        abort_event=(None if abort_event is None else freeze_value(abort_event)),
        init_override_event=(None if init_override_event is None
                             else freeze_value(init_override_event)),
        log_filename=path.name,
    )
    if signature is not None:
        _PARSE_CACHE[cache_key] = (signature, parsed)
    return parsed


def clear_run_file_cache() -> None:
    _PARSE_CACHE.clear()
    _HEADER_CACHE.clear()


__all__ = [
    "ParsedRunFile",
    "ParsedRunHeader",
    "clear_run_file_cache",
    "parse_run_file",
    "parse_run_header",
]
