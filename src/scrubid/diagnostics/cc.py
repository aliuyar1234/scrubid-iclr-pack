from __future__ import annotations

from typing import Any


def compute_cc(near_optimal_records: list[dict[str, Any]], canonical: dict[str, Any]) -> dict[str, Any]:
    thr_good = float(canonical["DIAGNOSTICS"]["CC_THRESHOLD_GOOD"])
    thr_mod = float(canonical["DIAGNOSTICS"]["CC_THRESHOLD_MODERATE"])

    circuits = [set(r["components"]) for r in near_optimal_records]
    necessity_maps: list[dict[str, bool]] = [dict(r.get("necessity", {})) for r in near_optimal_records]

    union: set[str] = set()
    for c in circuits:
        union |= c
    if not union:
        cc = 0.0
    else:
        contradictory = 0
        for v in union:
            necessary_somewhere = False
            contradicts_somewhere = False
            for c, nm in zip(circuits, necessity_maps, strict=True):
                if v in c and nm.get(v, False):
                    necessary_somewhere = True
            if necessary_somewhere:
                for c, nm in zip(circuits, necessity_maps, strict=True):
                    if v not in c:
                        contradicts_somewhere = True
                        break
                    if v in c and not nm.get(v, False):
                        contradicts_somewhere = True
                        break
            if necessary_somewhere and contradicts_somewhere:
                contradictory += 1
        cc = contradictory / len(union)

    if cc <= thr_good:
        verdict = canonical["ENUMS"]["VERDICTS"]["VERDICT_PASS"]
    elif cc <= thr_mod:
        verdict = canonical["ENUMS"]["VERDICTS"]["VERDICT_WARN"]
    else:
        verdict = canonical["ENUMS"]["VERDICTS"]["VERDICT_FAIL"]

    return {"CC": float(cc), "CC_verdict": verdict}
