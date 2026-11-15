"""Held-out selective-answering gate for evidence-grounded LLM outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist

MAX_INPUT_BYTES = 10 * 1024 * 1024
MAX_OBSERVATIONS = 100_000


@dataclass(frozen=True)
class SelectiveAnswerObservation:
    """A reviewed candidate output from a held-out evaluation set."""

    case_id: str
    confidence: float
    answerable: bool
    answer_supported: bool


@dataclass(frozen=True)
class SelectiveAnswerPolicy:
    """Fixed release policy; thresholds must be chosen before evaluation."""

    confidence_threshold: float = 0.75
    confidence_level: float = 0.95
    minimum_observations: int = 200
    minimum_answerable: int = 50
    minimum_unanswerable: int = 50
    maximum_selective_risk: float = 0.05
    maximum_false_answer_rate: float = 0.05
    minimum_answerable_coverage: float = 0.80

    def __post_init__(self) -> None:
        probabilities = {
            "confidence_threshold": self.confidence_threshold,
            "maximum_selective_risk": self.maximum_selective_risk,
            "maximum_false_answer_rate": self.maximum_false_answer_rate,
            "minimum_answerable_coverage": self.minimum_answerable_coverage,
        }
        for name, value in probabilities.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")
        if (
            isinstance(self.confidence_level, bool)
            or not isinstance(self.confidence_level, (int, float))
            or not math.isfinite(self.confidence_level)
            or not 0.5 < self.confidence_level < 1.0
        ):
            raise ValueError("confidence_level must be finite and within (0.5, 1.0)")
        for name in ("minimum_observations", "minimum_answerable", "minimum_unanswerable"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class BinomialMetric:
    successes: int
    trials: int
    rate: float
    lower_bound: float
    upper_bound: float


@dataclass(frozen=True)
class SelectiveAnswerReport:
    decision: str
    reasons: tuple[str, ...]
    evidence_digest: str
    observations: int
    answered: int
    abstained: int
    answerable: int
    unanswerable: int
    policy: SelectiveAnswerPolicy
    selective_risk: BinomialMetric
    false_answer_rate: BinomialMetric
    answerable_coverage: BinomialMetric

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _wilson_interval(successes: int, trials: int, confidence_level: float) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
    rate = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    centre = rate + z_squared / (2.0 * trials)
    margin = z * math.sqrt(rate * (1.0 - rate) / trials + z_squared / (4.0 * trials * trials))
    return max(0.0, (centre - margin) / denominator), min(1.0, (centre + margin) / denominator)


def _metric(successes: int, trials: int, confidence_level: float) -> BinomialMetric:
    lower, upper = _wilson_interval(successes, trials, confidence_level)
    return BinomialMetric(
        successes=successes,
        trials=trials,
        rate=successes / trials if trials else 0.0,
        lower_bound=lower,
        upper_bound=upper,
    )


def _validate_observations(
    observations: Sequence[SelectiveAnswerObservation], policy: SelectiveAnswerPolicy
) -> tuple[SelectiveAnswerObservation, ...]:
    if isinstance(observations, (str, bytes)):
        raise TypeError("observations must be a sequence")
    if len(observations) < policy.minimum_observations:
        raise ValueError(f"at least {policy.minimum_observations} observations are required")
    if len(observations) > MAX_OBSERVATIONS:
        raise ValueError(f"observations exceed the {MAX_OBSERVATIONS} record limit")

    seen: set[str] = set()
    validated: list[SelectiveAnswerObservation] = []
    answerable = 0
    for index, observation in enumerate(observations):
        if not isinstance(observation, SelectiveAnswerObservation):
            raise TypeError(f"observations[{index}] has the wrong type")
        if not isinstance(observation.case_id, str):
            raise TypeError(f"observations[{index}].case_id must be a string")
        if not observation.case_id.strip() or len(observation.case_id) > 256:
            raise ValueError(f"observations[{index}].case_id must contain 1-256 characters")
        if observation.case_id in seen:
            raise ValueError(f"duplicate case_id: {observation.case_id}")
        seen.add(observation.case_id)
        if (
            isinstance(observation.confidence, bool)
            or not isinstance(observation.confidence, (int, float))
            or not math.isfinite(observation.confidence)
            or not 0.0 <= observation.confidence <= 1.0
        ):
            raise ValueError(f"observations[{index}].confidence must be within [0, 1]")
        if not isinstance(observation.answerable, bool) or not isinstance(
            observation.answer_supported, bool
        ):
            raise TypeError(f"observations[{index}] labels must be booleans")
        if not observation.answerable and observation.answer_supported:
            raise ValueError(f"observations[{index}] cannot mark an unanswerable case as supported")
        answerable += int(observation.answerable)
        validated.append(observation)

    unanswerable = len(validated) - answerable
    if answerable < policy.minimum_answerable:
        raise ValueError(f"at least {policy.minimum_answerable} answerable cases are required")
    if unanswerable < policy.minimum_unanswerable:
        raise ValueError(f"at least {policy.minimum_unanswerable} unanswerable cases are required")
    return tuple(validated)


def _evidence_digest(
    observations: Sequence[SelectiveAnswerObservation], policy: SelectiveAnswerPolicy
) -> str:
    payload = {
        "observations": [
            asdict(item) for item in sorted(observations, key=lambda item: item.case_id)
        ],
        "policy": asdict(policy),
        "schema_version": 1,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def evaluate_selective_answering(
    observations: Sequence[SelectiveAnswerObservation],
    policy: SelectiveAnswerPolicy | None = None,
) -> SelectiveAnswerReport:
    """Gate a fixed abstention threshold against reviewed held-out outcomes.

    Risk policies use Wilson upper bounds; the coverage policy uses the Wilson
    lower bound. This prevents small samples from passing on optimistic point
    estimates alone.
    """

    policy = policy or SelectiveAnswerPolicy()
    admitted = _validate_observations(observations, policy)
    answered = tuple(item for item in admitted if item.confidence >= policy.confidence_threshold)
    answerable = tuple(item for item in admitted if item.answerable)
    unanswerable = tuple(item for item in admitted if not item.answerable)

    selective_risk = _metric(
        sum(not item.answer_supported for item in answered),
        len(answered),
        policy.confidence_level,
    )
    false_answer_rate = _metric(
        sum(item.confidence >= policy.confidence_threshold for item in unanswerable),
        len(unanswerable),
        policy.confidence_level,
    )
    answerable_coverage = _metric(
        sum(item.confidence >= policy.confidence_threshold for item in answerable),
        len(answerable),
        policy.confidence_level,
    )

    reasons: list[str] = []
    if not answered:
        reasons.append("selective_risk_unobservable")
    if selective_risk.upper_bound > policy.maximum_selective_risk:
        reasons.append("selective_risk_upper_bound_exceeded")
    if false_answer_rate.upper_bound > policy.maximum_false_answer_rate:
        reasons.append("false_answer_rate_upper_bound_exceeded")
    if answerable_coverage.lower_bound < policy.minimum_answerable_coverage:
        reasons.append("answerable_coverage_lower_bound_below_minimum")

    return SelectiveAnswerReport(
        decision="fail" if reasons else "pass",
        reasons=tuple(reasons),
        evidence_digest=_evidence_digest(admitted, policy),
        observations=len(admitted),
        answered=len(answered),
        abstained=len(admitted) - len(answered),
        answerable=len(answerable),
        unanswerable=len(unanswerable),
        policy=policy,
        selective_risk=selective_risk,
        false_answer_rate=false_answer_rate,
        answerable_coverage=answerable_coverage,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_observations(path: Path) -> list[SelectiveAnswerObservation]:
    if path.is_symlink():
        raise ValueError(f"refusing symlink input: {path}")
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError(f"input exceeds {MAX_INPUT_BYTES} bytes: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    rows = payload.get("observations") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise TypeError("input must be an array or an object with an observations array")
    observations: list[SelectiveAnswerObservation] = []
    expected = {"case_id", "confidence", "answerable", "answer_supported"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError(f"observations[{index}] must contain exactly {sorted(expected)}")
        observations.append(SelectiveAnswerObservation(**row))
    return observations


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate selective LLM answering quality")
    parser.add_argument("input", type=Path, help="reviewed JSON observations")
    parser.add_argument("--output", type=Path, required=True, help="atomic JSON report path")
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--minimum-observations", type=int, default=200)
    parser.add_argument("--minimum-answerable", type=int, default=50)
    parser.add_argument("--minimum-unanswerable", type=int, default=50)
    parser.add_argument("--maximum-selective-risk", type=float, default=0.05)
    parser.add_argument("--maximum-false-answer-rate", type=float, default=0.05)
    parser.add_argument("--minimum-answerable-coverage", type=float, default=0.80)
    args = parser.parse_args(argv)

    try:
        policy = SelectiveAnswerPolicy(
            confidence_threshold=args.confidence_threshold,
            minimum_observations=args.minimum_observations,
            minimum_answerable=args.minimum_answerable,
            minimum_unanswerable=args.minimum_unanswerable,
            maximum_selective_risk=args.maximum_selective_risk,
            maximum_false_answer_rate=args.maximum_false_answer_rate,
            minimum_answerable_coverage=args.minimum_answerable_coverage,
        )
        report = evaluate_selective_answering(_load_observations(args.input), policy)
        _write_atomic(args.output, report.to_dict())
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"selective answering evaluation failed: {exc}", file=sys.stderr)
        return 3
    return 0 if report.decision == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
