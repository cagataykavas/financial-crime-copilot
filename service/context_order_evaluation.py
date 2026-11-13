from __future__ import annotations

import argparse
import json
import math
import os
import stat
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MAX_INPUT_BYTES = 4_000_000
MAX_TRIALS = 5_000
MAX_EVIDENCE_PER_TRIAL = 256


class EvaluationInputError(ValueError):
    """Raised when benchmark evidence cannot be interpreted safely."""


@dataclass(frozen=True)
class ContextOrderTrial:
    case_id: str
    trial_id: str
    is_baseline: bool
    evidence_order: tuple[str, ...]
    action: str
    cited_evidence: frozenset[str]


@dataclass(frozen=True)
class EvaluationIssue:
    code: str
    message: str
    trial_id: str | None = None


@dataclass(frozen=True)
class CaseEvaluation:
    case_id: str
    trials: int
    comparable_variants: int
    action_stability_rate: float | None
    minimum_citation_jaccard: float | None
    issues: tuple[EvaluationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ok"] = self.ok
        payload["issues"] = [asdict(issue) for issue in self.issues]
        return payload


@dataclass(frozen=True)
class ContextOrderReport:
    minimum_trials: int
    minimum_citation_jaccard: float
    cases: tuple[CaseEvaluation, ...]

    @property
    def ok(self) -> bool:
        return bool(self.cases) and all(case.ok for case in self.cases)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "minimum_trials": self.minimum_trials,
            "minimum_citation_jaccard": self.minimum_citation_jaccard,
            "total_cases": len(self.cases),
            "passed_cases": sum(case.ok for case in self.cases),
            "cases": [case.as_dict() for case in self.cases],
        }


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationInputError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _string(value: Any, field: str, line: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationInputError(f"line {line}: {field} must be a non-empty string")
    return value


def _unique_strings(value: Any, field: str, line: int, *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "an array" if allow_empty else "a non-empty array"
        raise EvaluationInputError(f"line {line}: {field} must be {qualifier}")
    if len(value) > MAX_EVIDENCE_PER_TRIAL:
        raise EvaluationInputError(
            f"line {line}: {field} exceeds {MAX_EVIDENCE_PER_TRIAL} entries"
        )
    parsed = tuple(_string(item, field, line) for item in value)
    if len(set(parsed)) != len(parsed):
        raise EvaluationInputError(f"line {line}: {field} must not contain duplicates")
    return parsed


def _parse_trial(payload: Any, line: int) -> ContextOrderTrial:
    if not isinstance(payload, dict):
        raise EvaluationInputError(f"line {line}: trial must be a JSON object")
    case_id = _string(payload.get("case_id"), "case_id", line)
    trial_id = _string(payload.get("trial_id"), "trial_id", line)
    is_baseline = payload.get("is_baseline")
    if not isinstance(is_baseline, bool):
        raise EvaluationInputError(f"line {line}: is_baseline must be a boolean")
    evidence_order = _unique_strings(
        payload.get("evidence_order"), "evidence_order", line, allow_empty=False
    )
    action = _string(payload.get("action"), "action", line)
    cited = _unique_strings(
        payload.get("cited_evidence"), "cited_evidence", line, allow_empty=True
    )
    unknown_citations = sorted(set(cited) - set(evidence_order))
    if unknown_citations:
        raise EvaluationInputError(
            f"line {line}: cited_evidence references absent evidence: "
            + ", ".join(unknown_citations)
        )
    return ContextOrderTrial(
        case_id=case_id,
        trial_id=trial_id,
        is_baseline=is_baseline,
        evidence_order=evidence_order,
        action=action,
        cited_evidence=frozenset(cited),
    )


def load_trials(path: str | Path) -> list[ContextOrderTrial]:
    source = Path(path)
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise EvaluationInputError(f"cannot inspect input: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise EvaluationInputError("input must be a regular, non-symlink file")
    if metadata.st_size > MAX_INPUT_BYTES:
        raise EvaluationInputError(f"input exceeds {MAX_INPUT_BYTES} bytes")

    trials: list[ContextOrderTrial] = []
    try:
        with source.open("rb") as handle:
            raw_bytes = handle.read(MAX_INPUT_BYTES + 1)
        if len(raw_bytes) > MAX_INPUT_BYTES:
            raise EvaluationInputError(f"input exceeds {MAX_INPUT_BYTES} bytes")
        raw_text = raw_bytes.decode("utf-8")
        for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
            if not raw_line.strip():
                raise EvaluationInputError(f"line {line_number}: blank lines are not allowed")
            if len(trials) >= MAX_TRIALS:
                raise EvaluationInputError(f"input exceeds {MAX_TRIALS} trials")
            try:
                payload = json.loads(
                    raw_line,
                    object_pairs_hook=_strict_object,
                    parse_constant=lambda value, current_line=line_number: (
                        _ for _ in ()
                    ).throw(
                        EvaluationInputError(
                            f"line {current_line}: non-finite number {value}"
                        )
                    ),
                )
            except json.JSONDecodeError as exc:
                raise EvaluationInputError(
                    f"line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            trials.append(_parse_trial(payload, line_number))
    except (OSError, UnicodeDecodeError) as exc:
        raise EvaluationInputError(f"cannot read UTF-8 input: {exc}") from exc
    if not trials:
        raise EvaluationInputError("input contains no trials")
    return trials


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def evaluate_context_order(
    trials: list[ContextOrderTrial],
    *,
    minimum_trials: int = 2,
    minimum_citation_jaccard: float = 1.0,
) -> ContextOrderReport:
    if isinstance(minimum_trials, bool) or not isinstance(minimum_trials, int):
        raise EvaluationInputError("minimum_trials must be an integer")
    if not 2 <= minimum_trials <= 100:
        raise EvaluationInputError("minimum_trials must be between 2 and 100")
    if (
        isinstance(minimum_citation_jaccard, bool)
        or not isinstance(minimum_citation_jaccard, (int, float))
        or not math.isfinite(minimum_citation_jaccard)
        or not 0 <= minimum_citation_jaccard <= 1
    ):
        raise EvaluationInputError("minimum_citation_jaccard must be between 0 and 1")

    grouped: dict[str, list[ContextOrderTrial]] = defaultdict(list)
    for trial in trials:
        grouped[trial.case_id].append(trial)

    case_results: list[CaseEvaluation] = []
    for case_id in sorted(grouped):
        case_trials = grouped[case_id]
        issues: set[EvaluationIssue] = set()
        trial_id_counts = Counter(trial.trial_id for trial in case_trials)
        duplicate_ids = sorted(
            trial_id for trial_id, count in trial_id_counts.items() if count > 1
        )
        for trial_id in duplicate_ids:
            issues.add(
                EvaluationIssue("trial.duplicate_id", "trial ID is duplicated within case", trial_id)
            )
        if len(case_trials) < minimum_trials:
            issues.add(
                EvaluationIssue(
                    "case.insufficient_trials",
                    f"expected at least {minimum_trials} trials, found {len(case_trials)}",
                )
            )

        baselines = [trial for trial in case_trials if trial.is_baseline]
        if len(baselines) != 1:
            issues.add(
                EvaluationIssue(
                    "case.baseline_count",
                    f"expected exactly one baseline, found {len(baselines)}",
                )
            )
            case_results.append(
                CaseEvaluation(case_id, len(case_trials), 0, None, None, _sorted_issues(issues))
            )
            continue

        baseline = baselines[0]
        baseline_evidence = set(baseline.evidence_order)
        variants = [trial for trial in case_trials if not trial.is_baseline]
        comparable: list[ContextOrderTrial] = []
        for variant in variants:
            if set(variant.evidence_order) != baseline_evidence:
                issues.add(
                    EvaluationIssue(
                        "trial.evidence_membership_changed",
                        "variant must contain exactly the baseline evidence IDs",
                        variant.trial_id,
                    )
                )
                continue
            comparable.append(variant)
            if variant.action != baseline.action:
                issues.add(
                    EvaluationIssue(
                        "trial.action_changed",
                        f"action changed from {baseline.action!r} to {variant.action!r}",
                        variant.trial_id,
                    )
                )
            citation_jaccard = _jaccard(baseline.cited_evidence, variant.cited_evidence)
            if citation_jaccard < minimum_citation_jaccard:
                issues.add(
                    EvaluationIssue(
                        "trial.citation_instability",
                        f"citation Jaccard {citation_jaccard:.4f} is below "
                        f"{minimum_citation_jaccard:.4f}",
                        variant.trial_id,
                    )
                )

        if comparable and all(
            variant.evidence_order == baseline.evidence_order for variant in comparable
        ):
            issues.add(
                EvaluationIssue(
                    "case.no_reordering",
                    "at least one comparable variant must change evidence order",
                )
            )

        action_rate = None
        minimum_jaccard = None
        if comparable:
            action_rate = sum(variant.action == baseline.action for variant in comparable) / len(
                comparable
            )
            minimum_jaccard = min(
                _jaccard(baseline.cited_evidence, variant.cited_evidence)
                for variant in comparable
            )
        case_results.append(
            CaseEvaluation(
                case_id=case_id,
                trials=len(case_trials),
                comparable_variants=len(comparable),
                action_stability_rate=round(action_rate, 4) if action_rate is not None else None,
                minimum_citation_jaccard=(
                    round(minimum_jaccard, 4) if minimum_jaccard is not None else None
                ),
                issues=_sorted_issues(issues),
            )
        )
    return ContextOrderReport(
        minimum_trials=minimum_trials,
        minimum_citation_jaccard=minimum_citation_jaccard,
        cases=tuple(case_results),
    )


def _sorted_issues(issues: set[EvaluationIssue]) -> tuple[EvaluationIssue, ...]:
    return tuple(
        sorted(issues, key=lambda issue: (issue.code, issue.trial_id or "", issue.message))
    )


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gate LLM/RAG decisions against context-order sensitivity."
    )
    parser.add_argument("input", type=Path, help="JSONL context-order trials")
    parser.add_argument("--minimum-trials", type=int, default=2)
    parser.add_argument("--minimum-citation-jaccard", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        report = evaluate_context_order(
            load_trials(args.input),
            minimum_trials=args.minimum_trials,
            minimum_citation_jaccard=args.minimum_citation_jaccard,
        ).as_dict()
    except EvaluationInputError as exc:
        report = {
            "ok": False,
            "total_cases": 0,
            "passed_cases": 0,
            "cases": [],
            "issues": [{"code": "input.invalid", "message": str(exc)}],
        }
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _write_report(args.output, report)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
