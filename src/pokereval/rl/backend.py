from __future__ import annotations
from typing import Protocol
from pokereval.rl.config import RLConfig
from pokereval.rl.sampler import SamplingClient, Completion


class Backend(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def sampler(self) -> SamplingClient: ...
    def train_step(self, data: list, lr: float) -> dict: ...


class FakeBackend:
    """Offline backend: records train_step calls, no network."""

    def __init__(self, sampling_client: SamplingClient):
        self._sc = sampling_client
        self.calls: list[tuple[int, float]] = []

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def sampler(self) -> SamplingClient:
        return self._sc

    def train_step(self, data: list, lr: float) -> dict:
        self.calls.append((len(data), lr))
        return {"loss": 0.0}


class _TinkerSampler:
    """Wraps a tinker SamplingClient as our SamplingClient protocol."""

    def __init__(self, sc, tokenizer, max_tokens: int):
        self._sc = sc
        self._tok = tokenizer
        self._max_tokens = max_tokens

    def sample(self, prompt: str, n: int, temperature: float) -> list[Completion]:
        import tinker
        ids = self._tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
        )
        model_input = tinker.ModelInput.from_ints(ids)
        params = tinker.SamplingParams(max_tokens=self._max_tokens, temperature=temperature)
        resp = self._sc.sample(prompt=model_input, num_samples=n, sampling_params=params).result()
        out: list[Completion] = []
        for seq in resp.sequences:
            toks = list(seq.tokens_np.tolist()) if seq.tokens_np is not None else list(seq._tokens_list)
            lps = list(seq.logprobs_np.tolist()) if seq.logprobs_np is not None else list(seq._logprobs_list)
            text = self._tok.decode(toks)
            out.append(Completion(text=text, tokens=toks, logprobs=lps))
        return out


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
        return list(self._tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
        ))

    def sampler(self) -> SamplingClient:
        self._sampler_seq += 1
        sc = self._train.save_weights_and_get_sampling_client(name=f"poker-rl-{self._sampler_seq}")
        return _TinkerSampler(sc, self._tok, self._config.max_tokens)

    def train_step(self, data: list, lr: float) -> dict:
        tinker = self._tinker
        self._train.forward_backward(data, loss_fn="importance_sampling").result()
        self._train.optim_step(tinker.AdamParams(learning_rate=lr)).result()
        return {"loss": float("nan")}
