"""
Assignment 11 — Monitoring & Alerts.

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.

Author: Nguyen Tien Hung — 2A202603365
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0
    # Extra breakdowns
    blocked_by_layer: dict = field(default_factory=dict)
    output_redactions: int = 0
    egress_denied: int = 0
    llm_errors: int = 0
    latencies_ms: list = field(default_factory=list)

    def record(
        self,
        *,
        blocked: bool,
        layer: str | None,
        latency_ms: float | None = None,
        redacted: bool = False,
    ) -> None:
        """Update counters after one request has gone through the pipeline."""
        self.total_requests += 1
        if blocked:
            self.blocked_requests += 1
            key = layer or "unknown"
            self.blocked_by_layer[key] = self.blocked_by_layer.get(key, 0) + 1
        if layer == "rate_limiter":
            self.rate_limit_hits += 1
        if layer == "llm_error":
            self.llm_errors += 1
        if redacted:
            self.output_redactions += 1
        if latency_ms is not None:
            self.latencies_ms.append(latency_ms)

    def _add_alert(self, metric: str, value: float, threshold: float, message: str) -> Alert:
        # Avoid duplicate alerts for the same metric (keep the latest value)
        self.alerts = [a for a in self.alerts if a.metric != metric]
        alert = Alert(metric=metric, value=value, threshold=threshold, message=message)
        self.alerts.append(alert)
        return alert

    def check_metrics(self) -> list[Alert]:
        """Compute rates, append Alert objects when thresholds exceeded."""
        new_alerts: list[Alert] = []
        snap = self.snapshot()

        if self.total_requests and snap["block_rate"] > self.block_rate_threshold:
            new_alerts.append(self._add_alert(
                "block_rate", round(snap["block_rate"], 4), self.block_rate_threshold,
                f"Block rate {snap['block_rate']:.0%} > {self.block_rate_threshold:.0%} — "
                "possible attack campaign or over-blocking guardrail.",
            ))
        if self.rate_limit_hits >= self.rate_limit_hit_threshold:
            new_alerts.append(self._add_alert(
                "rate_limit_hits", self.rate_limit_hits, self.rate_limit_hit_threshold,
                f"{self.rate_limit_hits} rate-limit hits ≥ {self.rate_limit_hit_threshold} — "
                "flooding / cost attack suspected.",
            ))
        if self.judge_checks and snap["judge_fail_rate"] > self.judge_fail_rate_threshold:
            new_alerts.append(self._add_alert(
                "judge_fail_rate", round(snap["judge_fail_rate"], 4),
                self.judge_fail_rate_threshold,
                f"Judge fail rate {snap['judge_fail_rate']:.0%} > "
                f"{self.judge_fail_rate_threshold:.0%} — model output quality degraded.",
            ))
        if self.llm_errors and self.total_requests and self.llm_errors / self.total_requests > 0.2:
            new_alerts.append(self._add_alert(
                "llm_error_rate", round(self.llm_errors / self.total_requests, 4), 0.2,
                "More than 20% of requests failed at the LLM provider.",
            ))
        return new_alerts

    def export_json(self, filepath: str | None = None):
        """Write metrics + alerts to JSON under repo-root ``outputs/`` by default."""
        self.check_metrics()
        path = Path(filepath or default_metrics_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.snapshot()
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        payload["thresholds"] = {
            "block_rate": self.block_rate_threshold,
            "rate_limit_hits": self.rate_limit_hit_threshold,
            "judge_fail_rate": self.judge_fail_rate_threshold,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        lat = sorted(self.latencies_ms)
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "blocked_by_layer": dict(self.blocked_by_layer),
            "output_redactions": self.output_redactions,
            "egress_denied": self.egress_denied,
            "llm_errors": self.llm_errors,
            "latency_ms": {
                "count": len(lat),
                "avg": round(sum(lat) / len(lat), 2) if lat else None,
                "p50": round(lat[len(lat) // 2], 2) if lat else None,
                "max": round(lat[-1], 2) if lat else None,
            },
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
