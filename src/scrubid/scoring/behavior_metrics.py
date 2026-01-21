from __future__ import annotations

from typing import Any

import torch


def compute_metric(suite_id: str, model: Any, batch: dict[str, Any]) -> list[float]:
    tokens: torch.Tensor = batch["tokens"]
    with torch.no_grad():
        logits = batch.get("logits")
        if logits is None:
            logits = model(tokens)
        # logits shape: (batch, seq, vocab)
        last = logits[:, -1, :]

    if suite_id == "SUITE_REAL_IOI_V1":
        t_corr = torch.tensor(batch["token_correct"], device=last.device, dtype=torch.long)
        t_inc = torch.tensor(batch["token_incorrect"], device=last.device, dtype=torch.long)
        vals = last.gather(1, t_corr[:, None]).squeeze(1) - last.gather(1, t_inc[:, None]).squeeze(1)
        return [float(x) for x in vals.detach().cpu().tolist()]

    if suite_id == "SUITE_REAL_GREATERTHAN_YN_V1":
        t_corr = torch.tensor(batch["token_correct"], device=last.device, dtype=torch.long)
        t_inc = torch.tensor(batch["token_incorrect"], device=last.device, dtype=torch.long)
        vals = last.gather(1, t_corr[:, None]).squeeze(1) - last.gather(1, t_inc[:, None]).squeeze(1)
        return [float(x) for x in vals.detach().cpu().tolist()]

    if suite_id == "SUITE_REAL_GREATERTHAN_V1":
        good_ids: list[list[int]] = batch["good_token_ids"]
        bad_ids: list[list[int]] = batch["bad_token_ids"]
        out: list[float] = []
        for i in range(last.shape[0]):
            g = torch.tensor(good_ids[i], device=last.device, dtype=torch.long)
            b = torch.tensor(bad_ids[i], device=last.device, dtype=torch.long)
            g_lse = torch.logsumexp(last[i, g], dim=0) if g.numel() > 0 else torch.tensor(float("-inf"), device=last.device)
            b_lse = torch.logsumexp(last[i, b], dim=0) if b.numel() > 0 else torch.tensor(float("-inf"), device=last.device)
            out.append(float((g_lse - b_lse).detach().cpu().item()))
        return out

    if suite_id == "SUITE_REAL_INDUCTION_V1":
        t_corr = torch.tensor(batch["token_correct"], device=last.device, dtype=torch.long)
        t_dis = torch.tensor(batch["token_distract"], device=last.device, dtype=torch.long)
        vals = last.gather(1, t_corr[:, None]).squeeze(1) - last.gather(1, t_dis[:, None]).squeeze(1)
        return [float(x) for x in vals.detach().cpu().tolist()]

    raise KeyError(f"Unknown suite_id for metric: {suite_id}")
