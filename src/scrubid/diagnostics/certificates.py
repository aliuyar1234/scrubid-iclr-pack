from __future__ import annotations

from typing import Any


def build_certificate(payload: dict[str, Any], canonical: dict[str, Any]) -> dict[str, Any]:
    reason_codes = payload.get("reason_codes")
    if not isinstance(reason_codes, list) or not reason_codes or not all(isinstance(x, str) and x for x in reason_codes):
        raise ValueError("certificate payload must include non-empty reason_codes: list[str]")

    codes_obj = canonical.get("ENUMS", {}).get("CERTIFICATE_REASON_CODES", {})
    if isinstance(codes_obj, dict):
        allowed = {v for v in codes_obj.values() if isinstance(v, str) and v}
        if allowed and any(rc not in allowed for rc in reason_codes):
            raise ValueError(f"Unknown reason_code(s) in certificate payload: {reason_codes}")

        rr = codes_obj.get("REASON_RR_FAIL")
        sss = codes_obj.get("REASON_SSS_FAIL")
        cc = codes_obj.get("REASON_CC_FAIL")
        if all(isinstance(x, str) and x for x in [rr, sss, cc]):
            rank = {str(rr): 0, str(sss): 1, str(cc): 2}
            if reason_codes != sorted(reason_codes, key=lambda x: rank.get(str(x), 999)):
                raise ValueError("certificate reason_codes must be in deterministic order (RR, SSS, CC)")

    # The spec defines required contents; we keep a minimal deterministic structure.
    return {
        "project_id": canonical["PROJECT_ID"],
        "project_version": canonical["PROJECT_VERSION"],
        **payload,
    }
