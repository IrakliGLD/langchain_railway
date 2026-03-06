"""
Stage-0 AI firewall for user query safety.

This module classifies incoming prompts before they enter the planning pipeline.
"""
from dataclasses import dataclass
import re
from typing import List


@dataclass
class FirewallDecision:
    """Outcome produced by the Stage-0 firewall."""

    action: str  # allow | warn | block
    sanitized_query: str
    reason: str
    matched_rules: List[str]
    risk_score: int


_BLOCK_RULES = [
    (
        "instruction_override",
        re.compile(
            r"(ignore|disregard|bypass).{0,80}(previous|above|system|developer).{0,80}(instructions?|rules?)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"(reveal|show|print|leak|expose).{0,120}(system prompt|developer prompt|hidden prompt|internal instructions?)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "role_hijack",
        re.compile(
            r"(you are now|act as|pretend to be).{0,80}(system|developer|root|dan|jailbreak)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "dangerous_sql_intent",
        re.compile(
            r"\b(?:"
            r"drop\s+(?:table|schema|database)\b|"
            r"truncate\s+table\b|"
            r"alter\s+(?:table|schema)\b|"
            r"create\s+(?:table|schema|database)\b|"
            r"insert\s+into\b|"
            r"delete\s+from\b|"
            r"update\s+[a-zA-Z_][a-zA-Z0-9_]*\s+set\b"
            r")",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
]

_WARN_RULES = [
    ("prompt_meta_reference", re.compile(r"\b(system prompt|developer message|hidden instruction)\b", re.IGNORECASE)),
    ("sql_comment_token", re.compile(r"(--|/\*)")),
    ("excessive_control_chars", re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")),
]


def _sanitize_query(raw_query: str) -> str:
    text = str(raw_query or "")
    # Remove markdown fences and control chars.
    text = re.sub(r"```(?:[a-zA-Z0-9_+-]+)?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", text)
    # Normalize whitespace so the downstream model sees stable input.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def inspect_query(query: str) -> FirewallDecision:
    """Analyze query and return allow/warn/block decision."""
    sanitized = _sanitize_query(query)
    matched_rules: List[str] = []
    risk_score = 0

    for rule_name, pattern in _BLOCK_RULES:
        if pattern.search(sanitized):
            matched_rules.append(rule_name)
            risk_score += 5

    if len(sanitized) > 1800:
        matched_rules.append("oversized_prompt")
        risk_score += 2

    if re.search(r"\b(select|union|where)\b.{0,20}\bfrom\b", sanitized, re.IGNORECASE):
        matched_rules.append("sql_like_prompt")
        risk_score += 1

    warn_matches = []
    for rule_name, pattern in _WARN_RULES:
        if pattern.search(sanitized):
            warn_matches.append(rule_name)

    if matched_rules and any(rule in {"instruction_override", "prompt_exfiltration", "role_hijack", "dangerous_sql_intent"} for rule in matched_rules):
        return FirewallDecision(
            action="block",
            sanitized_query=sanitized,
            reason="Prompt blocked by AI firewall policy.",
            matched_rules=matched_rules,
            risk_score=max(risk_score, 8),
        )

    if matched_rules or warn_matches:
        all_rules = matched_rules + [w for w in warn_matches if w not in matched_rules]
        return FirewallDecision(
            action="warn",
            sanitized_query=sanitized,
            reason="Prompt accepted with guardrail warning signals.",
            matched_rules=all_rules,
            risk_score=max(risk_score, 3),
        )

    return FirewallDecision(
        action="allow",
        sanitized_query=sanitized,
        reason="Prompt accepted by AI firewall.",
        matched_rules=[],
        risk_score=0,
    )


def build_safe_refusal_message(decision: FirewallDecision) -> str:
    """Return user-facing safe refusal with a recovery path."""
    reason = decision.reason or "Unsafe instruction pattern detected."
    return (
        "I can help with analytical questions on the available public energy datasets, "
        "but I cannot follow instruction-override or system-prompt extraction requests.\n\n"
        f"Reason: {reason}\n"
        f"Matched rules: {', '.join(decision.matched_rules) if decision.matched_rules else 'policy'}\n\n"
        "Try a safe query format, for example:\n"
        "- \"Show balancing price trend from 2021 to 2025 in GEL.\"\n"
        "- \"Compare Enguri and Gardabani tariffs in 2024.\"\n"
        "- \"What was renewable PPA share in balancing electricity in June 2024?\""
    )
