.PHONY: test experiments figures all smoke
test:
	PYTHONPATH=src python -m pytest -q tests
experiments:
	python scripts/run_experiments.py --seeds 5
smoke:
	python scripts/run_experiments.py --seeds 1 --epochs 2 --out-dir results/smoke
figures:
	python scripts/make_figures.py
all: test experiments figures
