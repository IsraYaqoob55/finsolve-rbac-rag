"""End-to-end answer evaluation: real retrieval, real LLM, LLM-as-judge.

    python -m eval.answer_eval                      # all cases, writes docs/evaluation.md
    python -m eval.answer_eval --only fin- hr-      # case ids starting with these prefixes
    python -m eval.answer_eval --no-judge --delay 2 # deterministic checks only

Per case it checks:
- facts:            required facts/numbers appear (deterministic)
- no_leak:          no canary value from a department outside the role (deterministic)
- in_scope_sources: every source used belongs to the role's departments (deterministic)
- route:            SQL was used when required (deterministic)
- behaviour:        answered / refused / clarified / not found, as expected (judge)
Faithfulness (every claim supported by the sources) is reported separately (judge).

The judge is a *different, smaller* model than the one answering, which
reduces "grading its own homework" bias and uses a separate rate limit.
"""

import argparse
import json
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

import yaml

from app.auth.user_store import User
from app.core.config import PROJECT_ROOT, Settings, get_settings
from app.core.logging import ensure_utf8_console, setup_logging
from app.llm.base import LLMError, LLMProvider, Message
from app.services.chat_service import ChatAnswer
from eval.scoring import CaseResult, JudgeVerdict, parse_judge, score_case, validate_canaries

GOLDEN = PROJECT_ROOT / "eval" / "answer_golden.yaml"
REPORT = PROJECT_ROOT / "docs" / "evaluation.md"
RAW_DIR = PROJECT_ROOT / "eval" / "reports"

JUDGE_PROMPT = """You are a strict evaluator of an internal company assistant.
You get the user's QUESTION, the SOURCES the assistant was given, and its ANSWER.
Return ONLY a JSON object:
{
  "behaviour": "answered" | "refused_access" | "asked_clarification" | "not_found",
  "faithful": true | false,
  "unsupported_claims": ["..."],
  "relevance": 1-5
}
Definitions:
- behaviour "answered": gives substantive information that addresses the question.
- behaviour "refused_access": says the user's role does not have access to the needed data.
- behaviour "asked_clarification": asks the user to clarify or choose between options \
(it may list the options).
- behaviour "not_found": says the information is not in the available documents.
- faithful: every factual claim about the company in the ANSWER is supported by the \
SOURCES. Statements about the user's access rights, about what the documents do not \
contain, and clarification questions are NOT factual claims, so a refusal, a \
"not found" answer or a clarification question is faithful.
- unsupported_claims: up to 3 claims not supported by the sources (empty if faithful).
- relevance: how well the answer addresses the question (5 = fully)."""


T = TypeVar("T")

# Free LLM tiers limit tokens per *minute*, and one case can need several
# thousand tokens. On a rate-limit error we wait for the window to reset and
# retry, so quota pressure is never scored as a wrong answer.
RATE_LIMIT_WAITS = (60, 60, 90)


def patiently(call: Callable[[], T]) -> tuple[T, int]:
    """Run ``call``, retrying retryable LLM errors after a wait. Returns (result, ms)."""
    for wait in (*RATE_LIMIT_WAITS, None):
        started = time.perf_counter()
        try:
            return call(), int((time.perf_counter() - started) * 1000)
        except LLMError as exc:
            if not exc.retryable or wait is None:
                raise
            print(f"(rate limited, waiting {wait}s)", end=" ", flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


def judge_verdict(judge: LLMProvider, question: str, answer: ChatAnswer) -> JudgeVerdict:
    """Ask the judge model. LLM errors propagate so ``patiently`` can retry them."""
    sources = "\n\n".join(
        f"[{s.number}] {s.title} > {s.section}\n{s.content[:1500]}" for s in answer.sources
    )
    user = f"QUESTION:\n{question}\n\nSOURCES:\n{sources or '(none)'}\n\nANSWER:\n{answer.answer}"
    return parse_judge(judge.complete([Message("system", JUDGE_PROMPT), Message("user", user)]))


def run(args: argparse.Namespace) -> list[CaseResult]:
    from app.llm.openai_compatible import OpenAICompatibleProvider
    from app.services.container import build_services

    golden = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    canaries: dict[str, list[str]] = golden["canaries"]
    cases: list[dict[str, Any]] = [
        c for c in golden["cases"] if not args.only or c["id"].startswith(tuple(args.only))
    ]
    base = get_settings()
    problems = validate_canaries(canaries, base.resolve(base.data_dir))
    if problems:
        raise SystemExit("Invalid canaries:\n  " + "\n  ".join(problems))

    # More patient retries: the free LLM tier rate-limits bursts of requests.
    settings: Settings = base.model_copy(update={"llm_max_retries": 6})
    services = build_services(settings)
    judge = (
        None
        if args.no_judge
        else OpenAICompatibleProvider.from_settings(
            settings.model_copy(update={"llm_model": args.judge_model})
        )
    )
    results: list[CaseResult] = []
    try:
        for i, case in enumerate(cases, start=1):
            scope = services.policy.scope_for(case["role"])
            user = User(f"eval_{case['role']}", f"Eval {scope.display_name}", case["role"])
            history = [Message(m["role"], m["content"]) for m in case.get("history", [])]
            print(f"[{i:>2}/{len(cases)}] {case['id']:28}", end=" ", flush=True)
            try:
                answer, latency = patiently(
                    partial(services.chat.answer, case["question"], user, scope, history)
                )
            except LLMError as exc:
                result = score_case(case, "", "error", 0, [], scope.departments, canaries, None, 0)
                result.error = str(exc)
                results.append(result)
                print("ERROR", exc)
                continue
            verdict = (
                patiently(partial(judge_verdict, judge, case["question"], answer))[0]
                if judge
                else None
            )
            result = score_case(
                case,
                answer.answer,
                answer.route,
                len(answer.cited_sources),
                [s.department for s in answer.sources],
                scope.departments,
                canaries,
                verdict,
                latency,
            )
            results.append(result)
            status = "PASS" if result.passed else "FAIL " + "; ".join(result.failure_reasons)
            print(f"{latency / 1000:5.1f}s  {status}")
            time.sleep(args.delay)
    finally:
        services.close()
    return results


# ---------------------------------------------------------------- reporting


def _pct(part: int, whole: int) -> str:
    return f"{part / whole:.0%}" if whole else "n/a"


def write_report(results: list[CaseResult], judge_model: str | None, path: Path) -> str:
    judged = [r for r in results if r.verdict and r.verdict.behaviour != "unknown"]
    answered = [r for r in judged if r.verdict and r.verdict.behaviour == "answered"]
    latencies = sorted(r.latency_ms for r in results if r.error is None)
    by_category: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)

    total = len(results)
    passed = sum(r.passed for r in results)
    facts_ok = sum(r.checks["facts"] for r in results)
    leaks = sum(len(r.leaks) for r in results)
    out_of_scope = sum(not r.checks["in_scope_sources"] for r in results)
    behaviour_ok = sum(r.checks.get("behaviour", False) for r in judged)
    faithful = sum(bool(r.verdict and r.verdict.faithful) for r in judged)
    cited = sum(r.cited_count > 0 for r in answered)
    relevance = statistics.mean(r.verdict.relevance for r in judged if r.verdict) if judged else 0

    lines = [
        "# Evaluation report",
        "",
        f"_Generated {datetime.now(tz=UTC):%Y-%m-%d %H:%M} UTC by `python -m eval.answer_eval`. "
        f"Judge model: `{judge_model or 'none'}`._",
        "",
        "## Summary",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Cases passed (all checks) | **{passed} / {total}** ({_pct(passed, total)}) |",
        f"| Required facts present | {_pct(facts_ok, total)} |",
        f"| **Data leaks (canary values from restricted departments)** | **{leaks}** |",
        f"| Sources outside the user's departments | {out_of_scope} |",
        f"| Expected behaviour (judge) | {_pct(behaviour_ok, len(judged))} |",
        f"| Faithful: every claim supported by sources (judge) | {_pct(faithful, len(judged))} |",
        f"| Substantive answers citing at least one source | {_pct(cited, len(answered))} |",
        f"| Mean relevance, 1-5 (judge) | {relevance:.2f} |",
    ]
    if latencies:
        median = statistics.median(latencies) / 1000
        p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] / 1000
        lines.append(f"| Latency median / p95 | {median:.1f}s / {p95:.1f}s |")
    lines += ["", "## By category", "", "| Category | Passed |", "|---|---|"]
    for category, items in sorted(by_category.items()):
        lines.append(f"| {category} | {sum(r.passed for r in items)} / {len(items)} |")

    failures = [r for r in results if not r.passed]
    lines += ["", "## Failures", ""]
    if not failures:
        lines.append("None.")
    else:
        lines += ["| Case | Role | Reason |", "|---|---|---|"]
        lines += [f"| `{r.id}` | {r.role} | {'; '.join(r.failure_reasons)} |" for r in failures]

    unfaithful = [r for r in judged if r.verdict and not r.verdict.faithful]
    if unfaithful:
        lines += ["", "## Possibly unsupported claims (judge)", ""]
        for r in unfaithful:
            claims = "; ".join(r.verdict.unsupported_claims) if r.verdict else ""
            lines.append(f"- `{r.id}`: {claims or '(no details)'}")

    lines += [
        "",
        "## All cases",
        "",
        "| Case | Category | Role | Route | Behaviour | Faithful | Latency | Result |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        behaviour = r.verdict.behaviour if r.verdict else "-"
        faithful_label = ("yes" if r.verdict.faithful else "no") if r.verdict else "-"
        lines.append(
            f"| `{r.id}` | {r.category} | {r.role} | {r.route} | {behaviour} | {faithful_label} | "
            f"{r.latency_ms / 1000:.1f}s | {'PASS' if r.passed else 'FAIL'} |"
        )
    report = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate end-to-end answers.")
    parser.add_argument("--only", nargs="+", help="run cases whose id starts with these prefixes")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    parser.add_argument("--judge-model", default="openai/gpt-oss-20b")
    parser.add_argument("--delay", type=float, default=6.0, help="seconds between cases")
    parser.add_argument("--out", type=Path, default=REPORT, help="markdown report path")
    args = parser.parse_args()

    ensure_utf8_console()
    setup_logging("WARNING")
    results = run(args)
    write_report(results, None if args.no_judge else args.judge_model, args.out)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw = RAW_DIR / f"answers_{datetime.now(tz=UTC):%Y%m%d_%H%M%S}.json"
    raw.write_text(json.dumps([asdict(r) for r in results], indent=2, default=str), "utf-8")
    passed = sum(r.passed for r in results)
    print(f"\n{passed}/{len(results)} passed. Report: {args.out}  Raw results: {raw}")


if __name__ == "__main__":
    main()
