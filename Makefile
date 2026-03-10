PYTHON ?= python3
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate

.PHONY: help venv install compile smoke-help clean

help:
	@printf "Targets:\n"
	@printf "  make venv        Create local virtual environment\n"
	@printf "  make install     Install package in editable mode\n"
	@printf "  make compile     Byte-compile source tree\n"
	@printf "  make smoke-help  Check CLI help surfaces\n"
	@printf "  make clean       Remove local caches\n"

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(ACTIVATE) && pip install --upgrade pip && pip install -e .

compile:
	PYTHONPATH=src $(PYTHON) -m compileall src

smoke-help:
	PYTHONPATH=src $(PYTHON) -m divesensei.cli --help
	PYTHONPATH=src $(PYTHON) -m divesensei.cli detect --help

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

