from __future__ import annotations
import numpy as np
import tinker


def build_datum(
    prompt_tokens: list[int],
    completion_tokens: list[int],
    completion_logprobs: list[float],
    advantage: float,
) -> tinker.Datum:
    """Build an importance_sampling Datum.

    Prompt positions are masked by setting their advantage to 0 (the
    importance_sampling loss has no separate ``weights`` arg — a 0 advantage
    contributes no policy-gradient signal). Loss-fn inputs are therefore
    ``target_tokens``, ``advantages``, ``logprobs`` (the sampling logprobs).
    """
    full = list(prompt_tokens) + list(completion_tokens)
    model_input = tinker.ModelInput.from_ints(full[:-1])
    target_tokens = full[1:]                      # length L = len(full) - 1
    L = len(target_tokens)
    n_comp = len(completion_tokens)

    advantages = [0.0] * L
    logprobs = [0.0] * L
    # completion-target positions are the final n_comp positions of the shifted target
    for j in range(n_comp):
        i = L - n_comp + j
        advantages[i] = float(advantage)
        logprobs[i] = float(completion_logprobs[j])

    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": np.array(target_tokens, dtype=np.int64),
            "advantages": np.array(advantages, dtype=np.float32),
            "logprobs": np.array(logprobs, dtype=np.float32),
        },
    )
