from __future__ import annotations


from scrubid.canonical import get_canonical, load_canonical
from scrubid.config import load_config


def test_paper_allowed_interventions() -> None:
    canonical = load_canonical(".")
    cfg = load_config("configs/interventions.yaml", canonical)
    ids = [str(x.get("id")) for x in cfg.get("intervention_families", [])]
    assert ids == [str(get_canonical(canonical, "IDS.INTERVENTION_FAMILY_IDS.I_ACTPATCH"))]


def test_paper_repro_error_message_is_canonical() -> None:
    canonical = load_canonical(".")
    assert str(get_canonical(canonical, "ERRORS.NOT_IMPLEMENTED_FOR_PAPER_REPRO")) == "NOT_IMPLEMENTED_FOR_PAPER_REPRO"

