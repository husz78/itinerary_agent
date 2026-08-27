.PHONY: install run test eval clean

install:
	uv sync

run:
	uv run python -m src.main

test:
	uv run pytest -v

eval:
	uv run python evals/run_evals.py

clean:
	rm -rf .pytest_cache **/__pycache__ data/*.db
