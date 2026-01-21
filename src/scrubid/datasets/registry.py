from __future__ import annotations

from typing import Any, Callable

from scrubid.canonical import get_canonical
from scrubid.datasets.real_greaterthan import build_greaterthan_suite, build_greaterthan_yn_suite
from scrubid.datasets.real_induction import build_induction_suite
from scrubid.datasets.real_ioi import build_ioi_suite
from scrubid.datasets.synthetic import build_synth_suite


def get_suite(suite_id: str) -> Callable[..., dict[str, Any]]:
    # suite_id values are canonical strings like "SUITE_SYNTH_V1"
    mapping = {
        "SUITE_SYNTH_V1": build_synth_suite,
        "SUITE_REAL_IOI_V1": build_ioi_suite,
        "SUITE_REAL_GREATERTHAN_V1": build_greaterthan_suite,
        "SUITE_REAL_GREATERTHAN_YN_V1": build_greaterthan_yn_suite,
        "SUITE_REAL_INDUCTION_V1": build_induction_suite,
    }
    if suite_id not in mapping:
        raise KeyError(f"Unknown suite_id: {suite_id}")
    return mapping[suite_id]
