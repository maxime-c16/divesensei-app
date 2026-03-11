PYTHON ?= python3
VENV ?= .venv
BUN ?= bun
DESKTOP_DIR ?= apps/desktop
ACTIVATE = . $(VENV)/bin/activate

.PHONY: help venv install compile smoke-help desktop-setup desktop-check desktop-build electron-dev electron-start electron-prepare clean

help:
	@printf "Targets:\n"
	@printf "  make venv        Create local virtual environment\n"
	@printf "  make install     Install package in editable mode\n"
	@printf "  make compile     Byte-compile source tree\n"
	@printf "  make smoke-help  Check CLI help surfaces\n"
	@printf "  make desktop-setup  Install Bun dependencies for the Astro/Electron app\n"
	@printf "  make desktop-check  Run Astro type/config checks\n"
	@printf "  make desktop-build  Build the Astro desktop app\n"
	@printf "  make electron-dev   Run the Electron shell against the local dev app\n"
	@printf "  make electron-start Run the built Electron app locally\n"
	@printf "  make electron-prepare Patch Astro dist for packaged Electron runtime\n"
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

desktop-setup:
	cd $(DESKTOP_DIR) && $(BUN) install

desktop-check:
	cd $(DESKTOP_DIR) && $(BUN) run check

desktop-build:
	cd $(DESKTOP_DIR) && $(BUN) run build

electron-dev:
	cd $(DESKTOP_DIR) && $(BUN) run electron:dev

electron-start:
	cd $(DESKTOP_DIR) && $(BUN) run electron:start

electron-prepare:
	cd $(DESKTOP_DIR) && $(BUN) run electron:prepare

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
