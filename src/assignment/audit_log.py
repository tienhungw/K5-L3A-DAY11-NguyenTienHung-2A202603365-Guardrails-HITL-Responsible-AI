"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.

Secrets/PII are never written to disk in clear text: both the input and
the output are passed through ``content_filter`` before being stored.

Author: Nguyen Tien Hung — 2A202603365
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


def _scrub(text: str) -> str:
    """Redact secrets/PII before persisting (the audit log is itself a data sink)."""
    try:
        from guardrails.output_guardrails import content_filter

        return content_filter(text or "")["redacted"]
    except Exception:  # pragma: no cover - never let logging break the pipeline
        return text or ""


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self, max_text_chars: int = 500):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._pending: dict[str, dict] = {}
        self.max_text_chars = max_text_chars

    def _key(self, user_id: str, request_id: str | None) -> str:
        return request_id or f"{user_id}:latest"

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input + start timestamp keyed by request_id/user_id. Returns request_id."""
        request_id = request_id or uuid.uuid4().hex[:12]
        key = self._key(user_id, request_id)
        self._open[key] = time.perf_counter()
        self._pending[key] = {
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": utc_now_iso(),
            "input": _scrub(text)[: self.max_text_chars],
            "input_chars": len(text or ""),
        }
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
        extra: dict | None = None,
    ):
        """Store output, layer decision, latency; append to self.logs."""
        key = self._key(user_id, request_id)
        started = self._open.pop(key, None)
        entry = self._pending.pop(
            key,
            {
                "request_id": request_id,
                "user_id": user_id,
                "timestamp": utc_now_iso(),
                "input": None,
                "input_chars": 0,
            },
        )
        latency_ms = (time.perf_counter() - started) * 1000 if started is not None else None
        entry.update(
            {
                "output": _scrub(text)[: self.max_text_chars],
                "blocked": bool(blocked),
                "layer": layer,
                "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
                "completed_at": utc_now_iso(),
            }
        )
        if extra:
            entry["extra"] = extra
        self.logs.append(entry)
        return entry

    def summary(self) -> dict:
        by_layer: dict[str, int] = {}
        for e in self.logs:
            k = e.get("layer") or "passed"
            by_layer[k] = by_layer.get(k, 0) + 1
        return {
            "total": len(self.logs),
            "blocked": sum(1 for e in self.logs if e.get("blocked")),
            "by_layer": by_layer,
        }

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
