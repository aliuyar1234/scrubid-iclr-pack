from __future__ import annotations

from typing import Any


def compute_rr(candidate_records: list[dict[str, Any]], canonical: dict[str, Any]) -> dict[str, Any]:
    rel_frac = float(canonical["DIAGNOSTICS"]["RR_NEAR_OPTIMAL_MDL_REL_FRAC"])
    max_set = int(canonical["DIAGNOSTICS"]["RR_NUM_CIRCUITS_SET"])
    thr_low = float(canonical["DIAGNOSTICS"]["RR_THRESHOLD_LOW"])
    thr_high = float(canonical["DIAGNOSTICS"]["RR_THRESHOLD_HIGH"])

    def key(rec: dict[str, Any]):
        comps = tuple(sorted(rec["components"]))
        return (float(rec["mdl"]), int(len(comps)), comps)

    faithful = [r for r in candidate_records if bool(r.get("faithful", False))]
    faithful_sorted = sorted(faithful, key=key)
    if not faithful_sorted:
        return {
            "RR": 0.0,
            "RR_verdict": canonical["ENUMS"]["VERDICTS"]["VERDICT_FAIL"],
            "c_star": None,
            "s_near": [],
        }

    c_star = faithful_sorted[0]
    mdl_star = float(c_star["mdl"])
    s_near = [r for r in faithful_sorted if float(r["mdl"]) <= (1.0 + rel_frac) * mdl_star]
    s_near = sorted(s_near, key=key)[:max_set]

    def jaccard(a: set[str], b: set[str]) -> float:
        u = a | b
        if not u:
            return 1.0
        return len(a & b) / len(u)

    rr = 0.0
    circuits = [set(r["components"]) for r in s_near]
    for i in range(len(circuits)):
        for j in range(i + 1, len(circuits)):
            d = 1.0 - jaccard(circuits[i], circuits[j])
            if d > rr:
                rr = d

    if rr < thr_low:
        verdict = canonical["ENUMS"]["VERDICTS"]["VERDICT_PASS"]
    elif rr < thr_high:
        verdict = canonical["ENUMS"]["VERDICTS"]["VERDICT_WARN"]
    else:
        verdict = canonical["ENUMS"]["VERDICTS"]["VERDICT_FAIL"]

    return {
        "RR": float(rr),
        "RR_verdict": verdict,
        "c_star": c_star,
        "s_near": s_near,
    }
