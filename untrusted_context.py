from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

MAX_DEPTH = 6
MAX_NODES = 256
MAX_TEXT_CHARS = 8_000
MAX_TOTAL_CHARS = 32_000

_BIDI_CONTROLS = {
    "\u061c",
    "\u200e",
    "\u200f",
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
    "\u2066",
    "\u2067",
    "\u2068",
    "\u2069",
}

_PROMPT_CONTROL_PATTERNS = (
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|override|bypass)\b.{0,64}"
            r"\b(?:previous|prior|above|system|developer|instructions?|polic(?:y|ies)|rules?)\b"
        ),
    ),
    (
        "privileged_prompt_reference",
        re.compile(r"\b(?:system|developer)\s+(?:prompt|message|instructions?)\b"),
    ),
    (
        "secret_extraction",
        re.compile(
            r"\b(?:reveal|print|exfiltrate|expose|show|return)\b.{0,64}"
            r"\b(?:system prompt|developer message|secrets?|credentials?|api keys?|tokens?)\b"
        ),
    ),
    (
        "tool_execution",
        re.compile(
            r"\b(?:call|invoke|run|execute|use)\b.{0,48}"
            r"\b(?:tool|function|command|shell)\b"
        ),
    ),
    (
        "prompt_boundary_marker",
        re.compile(
            r"(?:\b(?:begin|end)\b.{0,24}\b(?:system|developer)\b.{0,24}"
            r"\b(?:prompt|message)\b|<\s*/?\s*(?:system|developer)\s*>)"
        ),
    ),
)

_ROLE_MARKER = re.compile(r"(?m)^\s*(?:system|developer|assistant|tool)(?:\s+message)?\s*:")


@dataclass(frozen=True)
class ContextAuditFinding:
    """Privacy-reduced finding from scanning an untrusted evidence field."""

    code: str
    field_path: str
    rule: str


@dataclass
class _TraversalBudget:
    nodes: int = 0
    total_chars: int = 0


def _safe_key(key: object) -> str:
    text = str(key)
    if re.fullmatch(r"[A-Za-z0-9_-]{1,40}", text):
        return text
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"key_sha256_{digest}"


def _canonicalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    without_format_controls = "".join(
        character for character in normalized if unicodedata.category(character) != "Cf"
    )
    return "\n".join(
        re.sub(r"\s+", " ", line).strip()
        for line in without_format_controls.casefold().splitlines()
    )


def _scan_text(text: str, field_path: str) -> list[ContextAuditFinding]:
    findings: list[ContextAuditFinding] = []
    if any(character in _BIDI_CONTROLS for character in text):
        findings.append(
            ContextAuditFinding(
                code="untrusted_bidi_control",
                field_path=field_path,
                rule="bidirectional_display_control",
            )
        )

    canonical = _canonicalize(text)
    flattened = canonical.replace("\n", " ")
    if _ROLE_MARKER.search(canonical):
        findings.append(
            ContextAuditFinding(
                code="prompt_control_payload",
                field_path=field_path,
                rule="role_boundary_marker",
            )
        )
    for rule, pattern in _PROMPT_CONTROL_PATTERNS:
        if pattern.search(flattened):
            findings.append(
                ContextAuditFinding(
                    code="prompt_control_payload",
                    field_path=field_path,
                    rule=rule,
                )
            )
    return findings


def _walk(
    value: object,
    *,
    field_path: str,
    depth: int,
    budget: _TraversalBudget,
    findings: list[ContextAuditFinding],
) -> None:
    budget.nodes += 1
    if budget.nodes > MAX_NODES:
        findings.append(
            ContextAuditFinding(
                code="context_scan_budget_exceeded",
                field_path=field_path,
                rule="node_limit",
            )
        )
        return
    if depth > MAX_DEPTH:
        findings.append(
            ContextAuditFinding(
                code="context_scan_budget_exceeded",
                field_path=field_path,
                rule="depth_limit",
            )
        )
        return

    if isinstance(value, str):
        if len(value) > MAX_TEXT_CHARS:
            findings.append(
                ContextAuditFinding(
                    code="context_scan_budget_exceeded",
                    field_path=field_path,
                    rule="field_character_limit",
                )
            )
            return
        budget.total_chars += len(value)
        if budget.total_chars > MAX_TOTAL_CHARS:
            findings.append(
                ContextAuditFinding(
                    code="context_scan_budget_exceeded",
                    field_path=field_path,
                    rule="total_character_limit",
                )
            )
            return
        findings.extend(_scan_text(value, field_path))
        return

    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            _walk(
                value[key],
                field_path=f"{field_path}.{_safe_key(key)}",
                depth=depth + 1,
                budget=budget,
                findings=findings,
            )
            if budget.nodes > MAX_NODES or budget.total_chars > MAX_TOTAL_CHARS:
                return
        return

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, item in enumerate(value):
            _walk(
                item,
                field_path=f"{field_path}[{index}]",
                depth=depth + 1,
                budget=budget,
                findings=findings,
            )
            if budget.nodes > MAX_NODES or budget.total_chars > MAX_TOTAL_CHARS:
                return


def audit_untrusted_evidence(
    *,
    summary: str,
    attributes: Mapping[str, object],
) -> tuple[ContextAuditFinding, ...]:
    """Audit evidence text before it may cross an LLM or agent trust boundary.

    The audit is intentionally deterministic and bounded. It rejects known prompt-control
    shapes and ambiguous Unicode display controls; it does not attempt to prove that
    accepted natural language is safe or semantically benign.
    """

    budget = _TraversalBudget()
    findings: list[ContextAuditFinding] = []
    _walk(
        summary,
        field_path="summary",
        depth=0,
        budget=budget,
        findings=findings,
    )
    _walk(
        attributes,
        field_path="attributes",
        depth=0,
        budget=budget,
        findings=findings,
    )
    return tuple(dict.fromkeys(findings))
