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


PRESETS: dict[str, RLConfig] = {
    "smoke": RLConfig(num_steps=2, batch_spots=2, group_size=2, eval_every=1, max_tokens=32),
    "real": RLConfig(num_steps=60, batch_spots=16, group_size=8, eval_every=10),
}


def get_preset(name: str) -> RLConfig:
    return PRESETS[name]
