.PHONY: install test test-fast eval-kuhn eval-leduc eval-holdem lint

install:
	pip install -e ".[dev]"

# Full test suite (OpenSpiel tests skip automatically if pyspiel is absent).
test:
	python -m pytest -q

# Pure-Python tests only — no OpenSpiel required.
test-fast:
	python -m pytest -q -m "not openspiel"

eval-kuhn:
	python -m pokereval.cli run --variant kuhn

eval-leduc:
	python -m pokereval.cli run --variant leduc

eval-holdem:
	python -m pokereval.cli run --variant holdem
