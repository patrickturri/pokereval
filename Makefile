.PHONY: install test test-fast leaderboard leaderboard-json metric-divergence synth rl-smoke rl-train selfplay-smoke selfplay-train demo demo-build clean

# Editable install with all extras (OpenSpiel, model SDKs, RL stack, demo).
# The `all` extra pulls in everything the offline test suite and smoke loops
# need — no API keys or Tinker account required to run them.
install:
	python -m pip install -e ".[all]"

# Full test suite (OpenSpiel-marked tests run only if pyspiel is installed).
test:
	python -m pytest -q

# Pure-Python tests only (skip anything needing OpenSpiel).
test-fast:
	python -m pytest -q -m "not openspiel"

# Reproduce the credential-free leaderboard for every variant.
# Override MODEL/PROVIDER to run a real frontier model, e.g.:
#   make leaderboard PROVIDER=anthropic MODEL=claude-opus-4-8
PROVIDER ?= fake
MODEL ?=
ITERATIONS ?= 2000
MODEL_FLAG := $(if $(MODEL),--model $(MODEL),)

leaderboard:
	@for v in kuhn leduc holdem; do \
		echo "## $$v"; \
		python -m pokereval.cli run --variant $$v --iterations $(ITERATIONS) \
			--provider $(PROVIDER) $(MODEL_FLAG); \
		echo ""; \
	done

# Emit machine-readable JSON for a single variant (default leduc) for archival.
VARIANT ?= leduc
leaderboard-json:
	python -m pokereval.cli run --variant $(VARIANT) --iterations $(ITERATIONS) \
		--provider $(PROVIDER) $(MODEL_FLAG) --format json

# Action-match vs. exploitability: a credential-free policy panel showing the two
# metrics disagree (and move together the wrong way). No model, no API keys.
metric-divergence:
	python -m pokereval.cli metric-divergence --variant $(VARIANT) --iterations $(ITERATIONS)

# Generate CFR/Nash-labeled synthetic spots (Phase 2 flywheel).
# e.g. make synth VARIANT=leduc TAG=mixed OUT=data/leduc_mixed.jsonl
TAG ?=
OUT ?= data/$(VARIANT)_synth.jsonl
TAG_FLAG := $(if $(TAG),--tag $(TAG),)
synth:
	python -m pokereval.cli synth --variant $(VARIANT) --iterations $(ITERATIONS) \
		--out $(OUT) $(TAG_FLAG)

# RLVR (Phase 2). rl-smoke runs the loop fully offline (mock backend, no account).
# rl-train runs the live Tinker training run (needs Tinker billing + W&B).
rl-smoke:
	python -m pokereval.cli rl-train --variant leduc --preset smoke

rl-train:
	python -m pokereval.cli rl-train --variant leduc --preset real --live --wandb

# Phase 3 self-play. selfplay-smoke runs fully offline (mock backend, no account).
selfplay-smoke:
	python -m pokereval.cli rl-selfplay --variant leduc --preset smoke

selfplay-train:
	python -m pokereval.cli rl-selfplay --variant leduc --preset real --live --wandb

# Phase 3 demo: serve the Phase-2 finding as a local web app (offline, no keys).
# First run computes the CFR analysis and caches it; subsequent runs are instant.
demo:
	python -m pokereval.cli demo --port 8000

# Force-recompute the cached solver analysis (e.g. after solver changes).
demo-build:
	python -m pokereval.cli demo --rebuild --port 8000

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
