"""The one read-only query service, plus the script workspace.

Every adaptive condition uses the same tool names, page size, ordering,
and truncation rules. Script execution is a separate capability. The
sandbox is a local process boundary for analytical scripts over a mount;
it is not a multi-tenant isolation product.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from beyond_retrieval.hashing import content_hash
from beyond_retrieval.models import (
    QUERY_TOOLS,
    AgentRecord,
    ObservationStatus,
    QueryLimits,
    ToolResult,
)

GROUP_BY = frozenset({"source", "event_type", "parser_status"})
NOT_FOUND = "no record with that id is in the visible snapshot"

TOOL_SCHEMA: dict[str, dict[str, object]] = {
    "list_sources": {"params": []},
    "describe_schema": {"params": []},
    "search_events": {
        "params": [
            "source",
            "event_type",
            "entity_key",
            "entity_value",
            "start",
            "end",
            "cursor",
            "limit",
        ]
    },
    "get_records": {"params": ["record_ids"]},
    "aggregate_events": {"params": ["source", "group_by", "start", "end"]},
    "ordering": ["timestamp", "record_id"],
    "page_size_cap": 200,
    "max_chars_cap": 12_000,
}


class QueryService:
    def __init__(self, records: list[AgentRecord], limits: QueryLimits | None = None) -> None:
        self.limits = limits or QueryLimits()
        self._records = sorted(records, key=lambda item: (item.timestamp, item.record_id))
        self._by_id = {item.record_id: item for item in self._records}

    def execute(self, tool: str, parameters: dict[str, object]) -> ToolResult:
        if tool not in QUERY_TOOLS:
            return _failed(f"unknown tool {tool}")
        try:
            if tool == "list_sources":
                return self._list_sources()
            if tool == "describe_schema":
                return self._describe_schema()
            if tool == "search_events":
                return self._search(parameters)
            if tool == "get_records":
                return self._get(parameters)
            if tool == "aggregate_events":
                return self._aggregate(parameters)
        except (TypeError, ValueError) as exc:
            return _failed(str(exc))
        return _failed(f"unhandled tool {tool}")

    def _list_sources(self) -> ToolResult:
        counts: dict[str, int] = {}
        for record in self._records:
            counts[record.source] = counts.get(record.source, 0) + 1
        lines = [f"{name}\t{counts[name]}" for name in sorted(counts)]
        raw = "\n".join(lines) if lines else ""
        status = ObservationStatus.OK if lines else ObservationStatus.EMPTY
        return _text_result(status, raw)

    def _describe_schema(self) -> ToolResult:
        columns = [
            "record_id",
            "source",
            "event_type",
            "timestamp",
            "timezone",
            "entities",
            "raw",
            "content_hash",
            "origin_offset",
            "parser_status",
        ]
        return _text_result(ObservationStatus.OK, "\n".join(columns))

    def _search(self, parameters: dict[str, object]) -> ToolResult:
        source = _optional_str(parameters.get("source"))
        event_type = _optional_str(parameters.get("event_type"))
        entity_key = _optional_str(parameters.get("entity_key"))
        entity_value = _optional_str(parameters.get("entity_value"))
        start = _optional_str(parameters.get("start"))
        end = _optional_str(parameters.get("end"))
        cursor = _optional_str(parameters.get("cursor"))
        matched = []
        for record in self._records:
            if source is not None and record.source != source:
                continue
            if event_type is not None and record.event_type != event_type:
                continue
            if start is not None and record.timestamp < start:
                continue
            if end is not None and record.timestamp > end:
                continue
            if entity_key is not None:
                if record.entities.get(entity_key) != entity_value:
                    continue
            matched.append(record)
        if cursor:
            origin_ts, origin_id = _decode_cursor(cursor)
            matched = [
                record
                for record in matched
                if (record.timestamp, record.record_id) > (origin_ts, origin_id)
            ]
        limit = _page_limit(parameters.get("limit"), self.limits.page_size)
        page = matched[:limit]
        next_cursor = None
        if len(matched) > len(page) and page:
            last = page[-1]
            next_cursor = _encode_cursor(last.timestamp, last.record_id)
        return _page(page, next_cursor, self.limits.max_chars, exhausted=not matched)

    def _get(self, parameters: dict[str, object]) -> ToolResult:
        raw_ids = parameters.get("record_ids")
        if not isinstance(raw_ids, list) or not all(isinstance(item, str) for item in raw_ids):
            return _failed("record_ids must be a list of strings")
        found: list[AgentRecord] = []
        for record_id in raw_ids:
            record = self._by_id.get(record_id)
            if record is not None:
                found.append(record)
        if not found:
            return ToolResult(
                status=ObservationStatus.VALID_NEGATIVE,
                error=NOT_FOUND,
                content_hash=content_hash({"missing": list(raw_ids)}),
            )
        return _page(found, None, self.limits.max_chars, exhausted=False)

    def _aggregate(self, parameters: dict[str, object]) -> ToolResult:
        group_by = parameters.get("group_by")
        if not isinstance(group_by, str) or group_by not in GROUP_BY:
            return _failed("group_by is not an allowed column")
        source = _optional_str(parameters.get("source"))
        start = _optional_str(parameters.get("start"))
        end = _optional_str(parameters.get("end"))
        counts: dict[str, int] = {}
        for record in self._records:
            if source is not None and record.source != source:
                continue
            if start is not None and record.timestamp < start:
                continue
            if end is not None and record.timestamp > end:
                continue
            key = getattr(record, group_by)
            counts[str(key)] = counts.get(str(key), 0) + 1
        lines = [f"{key}\t{counts[key]}" for key in sorted(counts)]
        status = ObservationStatus.OK if lines else ObservationStatus.EMPTY
        return _text_result(status, "\n".join(lines))


class ScriptRun:
    def __init__(
        self,
        execution_success: bool,
        stdout: str,
        stderr: str,
        cpu_seconds: float,
        findings: list[dict[str, object]],
        error: str | None,
    ) -> None:
        self.execution_success = execution_success
        self.stdout = stdout
        self.stderr = stderr
        self.cpu_seconds = cpu_seconds
        self.findings = findings
        self.error = error


class ScriptSandbox:
    """Run one analytical script against a caller-supplied database path."""

    def __init__(self, timeout_seconds: float = 120) -> None:
        self.timeout_seconds = timeout_seconds

    def run(self, source: str, database: Path) -> ScriptRun:
        if len(source) > 100_000:
            return ScriptRun(False, "", "", 0.0, [], "script exceeds size limit")
        with tempfile.TemporaryDirectory(prefix="br-script-") as work:
            script_path = Path(work) / "script.py"
            script_path.write_text(source, encoding="utf-8")
            env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONNOUSERSITE": "1",
                "PYTHONHASHSEED": "0",
            }
            started = time.monotonic()
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", str(script_path), str(database)],
                    cwd=work,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    env=env,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return ScriptRun(
                    False,
                    exc.stdout or "",
                    exc.stderr or "",
                    self.timeout_seconds,
                    [],
                    "script exceeded the per-call CPU limit",
                )
        findings: list[dict[str, object]] = []
        error = None
        success = completed.returncode == 0
        if success:
            try:
                parsed = json.loads(completed.stdout or "[]")
            except json.JSONDecodeError:
                success = False
                error = "script stdout is not JSON"
                parsed = []
            if success and not isinstance(parsed, list):
                success = False
                error = "script stdout must be a JSON list"
            elif success:
                if not all(isinstance(item, dict) for item in parsed):
                    success = False
                    error = "each script finding must be an object"
                else:
                    findings = parsed
        else:
            error = (completed.stderr or "script exited non-zero").strip()[:500]
        elapsed = time.monotonic() - started
        return ScriptRun(success, completed.stdout, completed.stderr, elapsed, findings, error)


class SplitValidator:
    """Known-answer fixture, then an independent row-count check on the case mount.

    This is the harness validator. A confirmatory study replaces the row-count
    check with the preregistered reference computation for that script family.
    """

    def __init__(self, fixture_db: Path, expected_fixture: list[dict[str, object]]) -> None:
        self.fixture_db = fixture_db
        self.expected_fixture = expected_fixture
        self.sandbox = ScriptSandbox(timeout_seconds=30)

    def validate(
        self,
        source: str,
        contract: object,
        findings: list[dict[str, object]],
        case_db: Path,
    ) -> bool:
        if contract is None:
            return False
        inputs = getattr(contract, "inputs", None)
        schema = getattr(contract, "output_schema", None)
        assumptions = getattr(contract, "assumptions", None)
        join_keys = getattr(contract, "join_keys", None)
        units = getattr(contract, "units", None)
        if not inputs or not schema or not assumptions or not join_keys or not units:
            return False
        trial = self.sandbox.run(source, self.fixture_db)
        if not trial.execution_success or trial.findings != self.expected_fixture:
            return False
        count = sqlite3.connect(case_db).execute("select count(*) from events").fetchone()
        if count is None:
            return False
        return any(item.get("n") == count[0] for item in findings)


def validation_status(required: bool, passed: bool | None) -> str:
    if not required:
        return "not_applicable"
    if passed is None:
        return "not_run"
    return "pass" if passed else "fail"


def _page(
    records: list[AgentRecord],
    cursor: str | None,
    max_chars: int,
    exhausted: bool,
) -> ToolResult:
    kept: list[AgentRecord] = []
    used = 0
    truncated = False
    for record in records:
        size = len(record.raw)
        if kept and used + size > max_chars:
            truncated = True
            break
        kept.append(record)
        used += size
        if used > max_chars:
            truncated = True
            break
    if not kept and exhausted:
        status = ObservationStatus.EMPTY
    elif truncated:
        status = ObservationStatus.TRUNCATED
    else:
        status = ObservationStatus.OK
    payload = [record.model_dump(mode="json") for record in kept]
    return ToolResult(
        status=status,
        records=kept,
        truncated=truncated,
        cursor=None if truncated else cursor,
        content_hash=content_hash(payload),
    )


def _text_result(status: ObservationStatus, raw: str) -> ToolResult:
    return ToolResult(
        status=status,
        summary=raw,
        content_hash=content_hash({"status": status.value, "summary": raw}),
    )


def _failed(message: str) -> ToolResult:
    return ToolResult(status=ObservationStatus.FAILED_TOOL, error=message, content_hash=content_hash(message))


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("expected a string parameter")
    return value


def _page_limit(value: object, cap: int) -> int:
    if value is None:
        return cap
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise TypeError("limit must be a positive integer")
    return min(value, cap)


def _encode_cursor(timestamp: str, record_id: str) -> str:
    return f"{timestamp}\x1f{record_id}"


def _decode_cursor(cursor: str) -> tuple[str, str]:
    if "\x1f" not in cursor:
        raise ValueError("invalid cursor")
    timestamp, record_id = cursor.split("\x1f", 1)
    return timestamp, record_id
