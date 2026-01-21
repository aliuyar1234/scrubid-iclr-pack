from __future__ import annotations

from typing import Any


def compute_sss(replicate_circuits: list[set[str]], canonical: dict[str, Any]) -> dict[str, Any]:
    thr_stable = float(canonical["DIAGNOSTICS"]["SSS_THRESHOLD_STABLE"])
    thr_border = float(canonical["DIAGNOSTICS"]["SSS_THRESHOLD_BORDERLINE"])

    def jaccard(a: set[str], b: set[str]) -> float:
        u = a | b
        if not u:
            return 1.0
        return len(a & b) / len(u)

    if len(replicate_circuits) < 2:
        sss = 0.0
    else:
        vals: list[float] = []
        for i in range(len(replicate_circuits)):
            for j in range(i + 1, len(replicate_circuits)):
                vals.append(jaccard(replicate_circuits[i], replicate_circuits[j]))
        sss = sum(vals) / len(vals) if vals else 0.0

    if sss >= thr_stable:
        verdict = canonical["ENUMS"]["VERDICTS"]["VERDICT_PASS"]
    elif sss >= thr_border:
        verdict = canonical["ENUMS"]["VERDICTS"]["VERDICT_WARN"]
    else:
        verdict = canonical["ENUMS"]["VERDICTS"]["VERDICT_FAIL"]

    return {"SSS": float(sss), "SSS_verdict": verdict}
