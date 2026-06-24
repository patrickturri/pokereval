"""One-off live Tinker probe: validates the TinkerBackend path end-to-end cheaply.

Not part of the test suite. Run manually with billing active:
    set -a; . ./.env; set +a; python scripts/live_probe.py
"""
import time
from pokereval.interface.types import GameVariant
from pokereval.synth.generator import build_labeled_spots
from pokereval.interface.prompts import render_state
from pokereval.rl.config import RLConfig
from pokereval.rl.backend import TinkerBackend
from pokereval.rl.reward import spot_reward
from pokereval.rl.advantage import group_advantages
from pokereval.rl.datum import build_datum


def main():
    cfg = RLConfig(base_model="Qwen/Qwen3.5-4B", max_tokens=32, lora_rank=32)
    t0 = time.time()
    spots = build_labeled_spots(GameVariant.LEDUC, iterations=200)
    print(f"[{time.time()-t0:.1f}s] built {len(spots)} spots")

    t = time.time()
    backend = TinkerBackend(cfg)
    print(f"[{time.time()-t:.1f}s] TinkerBackend created (LoRA client + tokenizer)")

    t = time.time()
    sampler = backend.sampler()
    print(f"[{time.time()-t:.1f}s] got sampler")

    spot = spots[0]
    prompt = render_state(spot.state)
    t = time.time()
    comps = sampler.sample(prompt, n=4, temperature=1.0)
    dt = time.time() - t
    print(f"[{dt:.1f}s] sampled {len(comps)} completions ({dt/len(comps):.2f}s each)")
    for c in comps:
        r, dec = spot_reward(spot, c.text)
        print(f"    reward={r:.3f} act={dec.action.value if dec else None} ntok={len(c.tokens)} nlp={len(c.logprobs)} text={c.text[:60]!r}")

    rewards = [spot_reward(spot, c.text)[0] for c in comps]
    advs = group_advantages(rewards)
    ptoks = backend.encode(prompt)
    data = [build_datum(ptoks, c.tokens, c.logprobs, a) for c, a in zip(comps, advs)]
    t = time.time()
    out = backend.train_step(data, lr=1e-5)
    print(f"[{time.time()-t:.1f}s] train_step done: {out}")
    print("PROBE_OK")


if __name__ == "__main__":
    main()
