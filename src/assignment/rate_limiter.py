"""
Assignment 11 — Rate Limiter.

Sliding-window, per-user rate limiting. Blocks abuse that other
guardrail layers do not address (flooding / cost attacks).

Author: Nguyen Tien Hung — 2A202603365
"""
from __future__ import annotations

from collections import defaultdict, deque
import time

from google.adk.plugins import base_plugin
from google.genai import types


class RateLimitPlugin(base_plugin.BasePlugin):
    """Block users who exceed max_requests within window_seconds."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60, clock=time.time):
        super().__init__(name="rate_limiter")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows: dict[str, deque] = defaultdict(deque)
        self.blocked_count = 0
        self.total_count = 0
        self._clock = clock  # injectable for deterministic tests

    def _block_response(self, message: str) -> types.Content:
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        )

    def remaining(self, user_id: str) -> int:
        """Requests still allowed for ``user_id`` in the current window."""
        self._evict(self.user_windows[user_id], self._clock())
        return max(0, self.max_requests - len(self.user_windows[user_id]))

    def _evict(self, window: deque, now: float) -> None:
        # Drop timestamps that fell out of the sliding window
        while window and window[0] <= now - self.window_seconds:
            window.popleft()

    async def on_user_message_callback(self, *, invocation_context, user_message):
        """Return Content to block, or None to allow."""
        self.total_count += 1
        user_id = getattr(invocation_context, "user_id", None) or "anonymous"
        now = self._clock()
        window = self.user_windows[user_id]

        # 1. Pop timestamps older than (now - window_seconds)
        self._evict(window, now)

        # 2. Over the limit → block without calling the LLM
        if len(window) >= self.max_requests:
            wait = max(0.0, self.window_seconds - (now - window[0]))
            self.blocked_count += 1
            return self._block_response(
                f"Rate limit exceeded ({self.max_requests} requests / "
                f"{self.window_seconds}s). Try again in {wait:.0f}s."
            )

        # 3. Under the limit → record and allow
        window.append(now)
        return None
