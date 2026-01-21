from __future__ import annotations

from typing import Any


def make_hooks(model: Any, canonical: dict[str, Any]) -> dict[str, Any]:
    # We patch at the attention z hookpoint (pre-W_O), so we do not enable use_attn_result.
    n_layers = int(getattr(getattr(model, "cfg", None), "n_layers", None) or getattr(model, "n_layers"))
    n_heads = int(getattr(getattr(model, "cfg", None), "n_heads", None) or getattr(model, "n_heads"))
    head_tpl = str(canonical["HOOKS"]["TL_HOOK_HEAD_RESULT_TEMPLATE"])
    mlp_tpl = str(canonical["HOOKS"]["TL_HOOK_MLP_OUT_TEMPLATE"])

    head_hooknames = [head_tpl.format(layer=i) for i in range(n_layers)]
    mlp_hooknames = [mlp_tpl.format(layer=i) for i in range(n_layers)]
    return {
        "backend": str(canonical["HOOKS"]["BACKEND_DEFAULT"]),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_hooknames": head_hooknames,
        "mlp_hooknames": mlp_hooknames,
    }
