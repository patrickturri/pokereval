from __future__ import annotations
from typing import Protocol
from pokereval.rl.config import RLConfig
from pokereval.rl.sampler import SamplingClient, Completion


# Terse system instruction: poker decision models (esp. reasoning models like
# Qwen3) otherwise emit long chain-of-thought and rarely reach a parseable ACTION
# line within the token budget, yielding all-zero rewards. This forces a single
# parseable line so the verifiable reward (and thus the RL signal) is nonzero.
SYSTEM_PROMPT = (
    "You are a poker solver. Reply with ONLY one line, exactly: "
    "ACTION: <action>  (for example 'ACTION: raise'). "
    "No explanation, no reasoning, no extra text."
)


def _encode_chat(tokenizer, text: str) -> list[int]:
    """Encode (system + user) turn to token ids, normalizing across tokenizers.

    `apply_chat_template(..., tokenize=True)` returns a flat list on some
    tokenizers and a BatchEncoding ({'input_ids': [...]}) on others; coerce both
    to a plain list[int]. Thinking mode is disabled where supported.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    kwargs = dict(add_generation_prompt=True, tokenize=True)
    try:
        out = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:  # tokenizer doesn't accept enable_thinking
        out = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(out, "input_ids"):  # BatchEncoding
        out = out["input_ids"]
    if out and isinstance(out[0], (list, tuple)):  # batched nesting
        out = out[0]
    return [int(t) for t in out]


def _seq_to_completion(seq, tokenizer) -> Completion:
    toks = list(seq.tokens_np.tolist()) if seq.tokens_np is not None else list(seq._tokens_list)
    lps = list(seq.logprobs_np.tolist()) if seq.logprobs_np is not None else list(seq._logprobs_list)
    return Completion(text=tokenizer.decode(toks), tokens=toks, logprobs=lps)


class Backend(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def sampler(self) -> SamplingClient: ...
    def train_step(self, data: list, lr: float) -> dict: ...
    def snapshot(self) -> SamplingClient: ...


class FakeBackend:
    """Offline backend: records train_step calls, no network."""

    def __init__(self, sampling_client: SamplingClient):
        self._sc = sampling_client
        self.calls: list[tuple[int, float]] = []
        self.snapshot_calls = 0

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def sampler(self) -> SamplingClient:
        return self._sc

    def train_step(self, data: list, lr: float) -> dict:
        self.calls.append((len(data), lr))
        return {"loss": 0.0}

    def snapshot(self) -> SamplingClient:
        """A frozen sampler. The mock policy is fixed, so this returns the same
        client; the counter lets the self-play loop assert refresh cadence."""
        self.snapshot_calls += 1
        return self._sc


class _TinkerSampler:
    """Wraps a tinker SamplingClient as our SamplingClient protocol."""

    def __init__(self, sc, tokenizer, max_tokens: int, chunk: int = 32):
        self._sc = sc
        self._tok = tokenizer
        self._max_tokens = max_tokens
        self._chunk = chunk

    def _params(self, temperature: float):
        import tinker
        return tinker.SamplingParams(max_tokens=self._max_tokens, temperature=temperature)

    def sample(self, prompt: str, n: int, temperature: float) -> list[Completion]:
        import tinker
        ids = _encode_chat(self._tok, prompt)
        model_input = tinker.ModelInput.from_ints(ids)
        resp = self._sc.sample(
            prompt=model_input, num_samples=n, sampling_params=self._params(temperature)
        ).result()
        return [_seq_to_completion(seq, self._tok) for seq in resp.sequences]

    def sample_many(
        self, prompts: list[str], n: int, temperature: float
    ) -> list[list[Completion]]:
        """Fire sampling requests concurrently in chunks (Tinker sample() returns
        a future), so 900+ eval prompts don't run end-to-end sequentially."""
        import tinker
        params = self._params(temperature)
        results: list[list[Completion]] = []
        for start in range(0, len(prompts), self._chunk):
            chunk = prompts[start:start + self._chunk]
            futures = []
            for p in chunk:
                mi = tinker.ModelInput.from_ints(_encode_chat(self._tok, p))
                futures.append(self._sc.sample(prompt=mi, num_samples=n, sampling_params=params))
            for fut in futures:
                resp = fut.result()
                results.append([_seq_to_completion(seq, self._tok) for seq in resp.sequences])
        return results


class TinkerBackend:
    """Live Tinker LoRA backend. Imports tinker lazily so the module loads without it."""

    def __init__(self, config: RLConfig):
        import tinker
        self._tinker = tinker
        self._config = config
        self._service = tinker.ServiceClient()
        self._train = self._service.create_lora_training_client(
            base_model=config.base_model, rank=config.lora_rank
        )
        self._tok = self._train.get_tokenizer()
        self._sampler_seq = 0

    def encode(self, text: str) -> list[int]:
        return _encode_chat(self._tok, text)

    def sampler(self) -> SamplingClient:
        self._sampler_seq += 1
        sc = self._train.save_weights_and_get_sampling_client(name=f"poker-rl-{self._sampler_seq}")
        return _TinkerSampler(sc, self._tok, self._config.max_tokens)

    def train_step(self, data: list, lr: float) -> dict:
        tinker = self._tinker
        self._train.forward_backward(data, loss_fn="importance_sampling").result()
        self._train.optim_step(tinker.AdamParams(learning_rate=lr)).result()
        return {"loss": float("nan")}

    def snapshot(self) -> SamplingClient:
        """Frozen opponent snapshot. ``sampler()`` already persists current
        weights via save_weights_and_get_sampling_client, so the returned client
        is bound to the weights at call time and unaffected by later train steps."""
        return self.sampler()
