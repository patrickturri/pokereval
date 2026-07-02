from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class RLConfig:
    base_model: str = "Qwen/Qwen3.5-4B"
    lora_rank: int = 32
    variant: str = "leduc"
    num_steps: int = 60
    batch_spots: int = 16          # distinct spots sampled per step
    group_size: int = 8            # completions sampled per spot
    learning_rate: float = 1e-5
    temperature: float = 1.0
    max_tokens: int = 32          # enough for "ACTION: <action>"; terse system prompt
    refresh_every: int = 1         # steps between sampler refreshes
    eval_every: int = 10
    eval_split: float = 0.2
    seed: int = 0
    iterations: int = 2000         # CFR iterations for synth labels
    reward_mode: str = "kl_nash"   # "kl_nash" (rewards mixing) or "nash_prob" (collapses)
    eval_samples: int = 16         # samples/node for mixed-policy exploitability
    # Self-play (Phase 3)
    deals_per_step: int = 8         # distinct deals sampled per step
    hands_per_deal: int = 8         # hands per deal = GRPO group size (deal-conditioned)
    opponent_refresh_every: int = 10  # steps between frozen-opponent refreshes


PRESETS: dict[str, RLConfig] = {
    "smoke": RLConfig(num_steps=2, batch_spots=2, group_size=2, eval_every=1, max_tokens=32,
                      eval_samples=4, deals_per_step=2, hands_per_deal=2,
                      opponent_refresh_every=1),
    "real": RLConfig(num_steps=60, batch_spots=16, group_size=8, eval_every=10,
                     deals_per_step=8, hands_per_deal=8, opponent_refresh_every=10),
    # Credential-free TABULAR self-play (`rl-selfplay --tabular`). The policy
    # actually learns offline, so these sizes target a real exploitability drop
    # rather than a mechanics check. `tabular` is the headline run (~30s);
    # `tabular_smoke` just exercises the loop fast.
    "tabular_smoke": RLConfig(num_steps=200, deals_per_step=8, hands_per_deal=6,
                              learning_rate=0.4, opponent_refresh_every=10,
                              eval_samples=12, seed=0),
    "tabular": RLConfig(num_steps=800, deals_per_step=16, hands_per_deal=10,
                        learning_rate=0.3, opponent_refresh_every=10,
                        eval_samples=32, seed=0),
}


def get_preset(name: str) -> RLConfig:
    return PRESETS[name]
