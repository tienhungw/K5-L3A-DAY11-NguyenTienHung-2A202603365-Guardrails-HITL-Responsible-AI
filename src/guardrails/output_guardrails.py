"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← optional (không chấm)

Author: Nguyen Tien Hung — 2A202603365
"""
import re
import textwrap
import unicodedata

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.config import DEMO_SECRETS
from core.utils import chat_with_agent


# ============================================================
# Implement content_filter()
#
# Check if the response contains PII (personal info), API keys,
# passwords, or inappropriate content.
#
# Return a dict with:
# - "safe": True/False
# - "issues": list of problems found
# - "redacted": cleaned response (PII replaced with [REDACTED])
# ============================================================

REDACTED = "[REDACTED]"

# Order matters: secrets/credentials first (longest, most specific), then PII.
PII_PATTERNS = {
    # sk-… style API keys
    "api_key": r"\bsk-[A-Za-z0-9_-]{6,}",
    # "password is X", "password=X", "mật khẩu: X" — the value must look like a
    # credential (contains a digit/symbol), so "password: go to Settings" is kept.
    "password": (
        r"(?:password|passwd|pwd|passcode|m[aậ]t\s*kh[aẩ]u)\s*"
        r"(?:is|was|l[aà]|:|=)\s*[\"'`]?(?=[^\s\"'`,;]*[\d!@#$%^&*_])[^\s\"'`,;]{4,}"
    ),
    # Known protected values + any *.internal host (with optional port)
    "secret": r"\badmin123\b|\b[\w.-]+\.internal(?::\d{2,5})?\b",
    # Email addresses
    "email": r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b",
    # VN phone: +84 / 0 followed by 9–10 digits (mobile 10 số, landline 11 số);
    # the public hotline 1900 xxx xxx does not start with 0 and is kept.
    "phone": r"(?<![\d])(?:\+84|0)(?:[\s.-]?\d){9,10}(?![\d])",
    # CCCD (12 digits) / CMND (9 digits)
    "national_id": r"(?<![\d])(?:\d{12}|\d{9})(?![\d])",
}
_COMPILED_PII = {k: re.compile(v, re.IGNORECASE) for k, v in PII_PATTERNS.items()}

# Issue types that mean *internal* data escaped (block, not just redact)
SECRET_ISSUES = {"api_key", "password", "secret", "obfuscated_secret"}

# Normalized needles for spotting obfuscated secrets ("a d m i n 1 2 3",
# "sk_vinbank_secret_2024", "db . vinbank . internal" …)
_SECRET_NEEDLES = sorted(
    {re.sub(r"[^a-z0-9]", "", s.lower()) for s in DEMO_SECRETS if s}
    | {"admin123", "skvinbanksecret2024", "dbvinbankinternal"},
    key=len,
    reverse=True,
)


def _contains_obfuscated_secret(text: str) -> bool:
    flat = re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKC", text or "").lower())
    return any(n and (n in flat or n[::-1] in flat) for n in _SECRET_NEEDLES)


def content_filter(response: str) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
        (plus 'issue_types' for programmatic checks).
    """
    issues = []
    issue_types = []
    redacted = response or ""

    for name, pattern in _COMPILED_PII.items():
        matches = pattern.findall(redacted)
        if matches:
            issues.append(f"{name}: {len(matches)} found")
            issue_types.append(name)
            redacted = pattern.sub(REDACTED, redacted)

    # Secrets that survived the regexes in a spaced / reversed / punctuated form
    if _contains_obfuscated_secret(redacted):
        issues.append("obfuscated_secret: protected value detected after normalization")
        issue_types.append("obfuscated_secret")

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "issue_types": issue_types,
        "redacted": redacted,
    }


# ============================================================
# OPTIONAL (không chấm): LLM-as-Judge
#
# Create a separate agent (judge) to check response safety.
# The judge classifies responses as SAFE or UNSAFE.
#
# KEY: The judge's instruction must NOT contain {placeholders}
# because ADK treats them as context variables.
# Instead, pass the content to evaluate as the user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""

# Optional judge is intentionally disabled (not graded; Blue model is fixed and
# deterministic regex filters are the graded boundary).
safety_judge_agent = None
judge_runner = None


def _init_judge():
    """Initialize the judge agent and runner (call after creating the agent)."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict.strip()}


# ============================================================
# Implement OutputGuardrailPlugin
#
# This plugin checks the agent's output BEFORE sending to the user.
# Uses after_model_callback to intercept LLM responses.
# Combines content_filter() and llm_safety_check().
#
# NOTE: after_model_callback uses keyword-only arguments.
#   - llm_response has a .content attribute (types.Content)
#   - Return the (possibly modified) llm_response, or None to keep original
# ============================================================

OUTPUT_BLOCK_MESSAGE = (
    "I'm sorry, I can't share internal system details. "
    "How else can I help with your VinBank account or banking needs?"
)


class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user.

    Policy:
      * customer PII (phone / email / ID)  → redact in place, keep the answer
      * internal secret (password, sk-key, *.internal host, obfuscated secret)
        → withhold the whole answer (fail-closed: partial context may still leak)
    """

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0
        self.last_action: str | None = None  # None | "redacted" | "blocked"
        self.last_issues: list[str] = []

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts or []:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    @staticmethod
    def _set_text(llm_response, text: str):
        llm_response.content = types.Content(
            role="model", parts=[types.Part.from_text(text=text)]
        )
        return llm_response

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1
        self.last_action = None
        self.last_issues = []

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        # 1. Deterministic PII / secret filter
        result = content_filter(response_text)
        if not result["safe"]:
            self.last_issues = result["issues"]
            if SECRET_ISSUES & set(result["issue_types"]):
                self.blocked_count += 1
                self.last_action = "blocked"
                return self._set_text(llm_response, OUTPUT_BLOCK_MESSAGE)
            self.redacted_count += 1
            self.last_action = "redacted"
            self._set_text(llm_response, result["redacted"])
            response_text = result["redacted"]

        # 2. Optional LLM-as-Judge (disabled by default)
        if self.use_llm_judge:
            verdict = await llm_safety_check(response_text)
            if not verdict["safe"]:
                self.blocked_count += 1
                self.last_action = "blocked"
                self.last_issues.append(f"judge: {verdict['verdict'][:80]}")
                return self._set_text(llm_response, OUTPUT_BLOCK_MESSAGE)

        # 3. Return (possibly modified) response
        return llm_response


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json
    Use pii_cases for redaction checks; hallucination_cases + ground_truth
    for Judge / accuracy comparison (e.g. savings 12m = 4.25%, not 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")

    # Dataset check: data/pii_hallucination_samples.json → pii_cases
    try:
        cases = load_lab_pii_dataset().get("pii_cases", [])
    except Exception as e:  # pragma: no cover
        print(f"  (dataset not loaded: {e})")
        return
    passed = 0
    for case in cases:
        result = content_filter(case["input_text"])
        ok_safe = result["safe"] == case["expect_safe"]
        ok_types = set(case.get("expect_issue_types", [])) <= set(result["issue_types"])
        ok_red = (REDACTED in result["redacted"]) == case["expect_contains_redacted"]
        ok = ok_safe and ok_types and ok_red
        passed += ok
        if not ok:
            print(f"  [FAIL] {case['id']}: got {result['issue_types']}")
    print(f"  PII dataset: {passed}/{len(cases)} cases match expectations")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()
