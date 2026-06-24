.PHONY: install test test-fast leaderboard leaderboard-json synth rl-smoke rl-train selfplay-smoke selfplay-train clean

# Editable install with all extras (OpenSpiel, model SDKs).
install:
	python -m pip install -e .

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

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
