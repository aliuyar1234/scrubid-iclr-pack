from __future__ import annotations

from typing import Any, Callable


def _parse_component_id(cid: str) -> tuple[str, int, int | None]:
    if cid.startswith("H"):
        # H{layer}:{head}
        layer_s, head_s = cid[1:].split(":", 1)
        return ("H", int(layer_s), int(head_s))
    if cid.startswith("M"):
        return ("M", int(cid[1:]), None)
    raise ValueError(f"Invalid component id: {cid}")


def apply_actpatch(
    model: Any,
    circuit: dict[str, Any],
    hooks: dict[str, Any],
    reference_cache: dict[str, Any],
) -> Callable[[Any], Any]:
    components = set(circuit.get("components", []))
    n_layers = int(hooks["n_layers"])
    n_heads = int(hooks["n_heads"])
    head_hooknames: list[str] = list(hooks["head_hooknames"])
    mlp_hooknames: list[str] = list(hooks["mlp_hooknames"])

    keep_heads_by_layer: dict[int, set[int]] = {l: set() for l in range(n_layers)}
    keep_mlp: set[int] = set()
    for cid in components:
        kind, layer, head = _parse_component_id(str(cid))
        if kind == "H":
            if head is None or head < 0 or head >= n_heads:
                raise ValueError(f"Invalid head id: {cid}")
            keep_heads_by_layer[layer].add(int(head))
        elif kind == "M":
            keep_mlp.add(int(layer))

    patch_heads_by_layer: dict[int, list[int]] = {}
    for l in range(n_layers):
        patch_heads_by_layer[l] = [h for h in range(n_heads) if h not in keep_heads_by_layer[l]]

    fwd_hooks = []
    for layer in range(n_layers):
        head_name = head_hooknames[layer]
        mlp_name = mlp_hooknames[layer]
        patch_heads = patch_heads_by_layer[layer]
        keep_mlp_layer = layer in keep_mlp

        def _make_head_hook(name: str, patch_heads_local: list[int]):
            def head_hook(act, hook):
                if not patch_heads_local:
                    return act
                ref = reference_cache[name]
                out = act.clone()
                out[:, :, patch_heads_local, :] = ref[:, :, patch_heads_local, :]
                return out

            return head_hook

        def _make_mlp_hook(name: str, keep_layer: bool):
            def mlp_hook(act, hook):
                if keep_layer:
                    return act
                ref = reference_cache[name]
                return ref

            return mlp_hook

        fwd_hooks.append((head_name, _make_head_hook(head_name, patch_heads)))
        fwd_hooks.append((mlp_name, _make_mlp_hook(mlp_name, keep_mlp_layer)))

    def scrubbed(tokens):
        return model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

    return scrubbed
