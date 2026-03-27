SHELL := /bin/bash

PYTHON ?= python3
VENV ?= .venv
DESKTOP_DIR ?= apps/desktop
DESKTOP_DIR_ABS := $(abspath $(DESKTOP_DIR))
RUN_DIR ?= .run
LOCK_DIR ?= $(RUN_DIR)/locks
APP_NAME ?= divesensei-desktop
APP_PORT ?= 5173
APP_HOST ?= 0.0.0.0
APP_URL ?= http://127.0.0.1:$(APP_PORT)
APP_PID_FILE ?= $(abspath $(RUN_DIR)/desktop-preview.pid)
APP_LOG_FILE ?= $(abspath $(RUN_DIR)/desktop-preview.log)
SETUP_STAMP_FILE ?= $(abspath $(RUN_DIR)/desktop-setup.stamp)
BUILD_STAMP_FILE ?= $(abspath $(RUN_DIR)/desktop-build.stamp)
ACTIVATE = . $(VENV)/bin/activate
BUN ?= $(shell command -v bun 2>/dev/null || printf '%s' "$(HOME)/.bun/bin/bun")
NODE ?= $(shell command -v node)
NPM ?= $(shell command -v npm)
CURL ?= curl

.PHONY: help venv install compile smoke-help desktop-setup desktop-check desktop-build \
	electron-dev electron-start electron-prepare status wait up down restart re logs \
	clean clean-runtime clean-build clean-python clean-app

help:
	@printf "Targets:\n"
	@printf "  make venv           Create the local Python virtualenv\n"
	@printf "  make install        Install the Python package in editable mode\n"
	@printf "  make compile        Byte-compile the Python source tree\n"
	@printf "  make smoke-help     Check the CLI help surfaces\n"
	@printf "  make desktop-setup  Install Bun dependencies for the Astro app\n"
	@printf "  make desktop-check  Run Astro type/config checks\n"
	@printf "  make desktop-build  Build the Astro desktop app\n"
	@printf "  make up             Build and start the desktop app preview server in the background\n"
	@printf "  make down           Stop the background app server\n"
	@printf "  make re             Restart the background app server\n"
	@printf "  make status         Show preview server status\n"
	@printf "  make logs           Tail the preview server log\n"
	@printf "  make clean          Remove caches, build output, and runtime logs\n"

$(RUN_DIR):
	@mkdir -p "$(RUN_DIR)"

$(LOCK_DIR): | $(RUN_DIR)
	@mkdir -p "$(LOCK_DIR)"
	@for lock in app.lock build.lock setup.lock; do \
		if [[ -d "$(LOCK_DIR)/$$lock" ]]; then rmdir "$(LOCK_DIR)/$$lock"; fi; \
		touch "$(LOCK_DIR)/$$lock"; \
	done

venv:
	$(PYTHON) -m venv "$(VENV)"

install: venv
	$(ACTIVATE) && pip install --upgrade pip && pip install -e .

compile:
	PYTHONPATH=src $(PYTHON) -m compileall src

smoke-help:
	PYTHONPATH=src $(PYTHON) -m divesensei.cli --help
	PYTHONPATH=src $(PYTHON) -m divesensei.cli detect --help

desktop-setup: | $(RUN_DIR) $(LOCK_DIR)
	@set -euo pipefail; \
	test -x "$(BUN)" || { echo "bun not found at $(BUN)"; exit 1; }; \
	test -x "$(NODE)" || { echo "node not found at $(NODE)"; exit 1; }; \
	test -x "$(NPM)" || { echo "npm not found at $(NPM)"; exit 1; }; \
	flock -o "$(LOCK_DIR)/setup.lock" bash -lc 'set -euo pipefail; \
		compute_setup_sig() { \
			{ \
				printf "node=%s\n" "$$("$(NODE)" -v)"; \
				printf "bun=%s\n" "$$("$(BUN)" --version)"; \
				cd "$(DESKTOP_DIR_ABS)"; \
				sha256sum package.json bun.lock; \
			} | sha256sum | cut -d" " -f1; \
		}; \
		signature="$$(compute_setup_sig)"; \
		addon_path="$(DESKTOP_DIR_ABS)/node_modules/better-sqlite3/build/Release/better_sqlite3.node"; \
		if [[ -f "$(SETUP_STAMP_FILE)" && -f "$$addon_path" && "$$(cat "$(SETUP_STAMP_FILE)")" == "$$signature" ]]; then \
			echo "desktop dependencies already up to date"; \
			exit 0; \
		fi; \
		cd "$(DESKTOP_DIR_ABS)" && "$(BUN)" install; \
		export PATH="$(dir $(NODE)):$$PATH"; \
		cd "$(DESKTOP_DIR_ABS)/node_modules/better-sqlite3"; \
		"$(NODE)" ../node-gyp/bin/node-gyp.js rebuild --release; \
		printf "%s\n" "$$signature" > "$(SETUP_STAMP_FILE)"'

desktop-check: desktop-setup
	cd "$(DESKTOP_DIR_ABS)" && "$(NODE)" ./node_modules/astro/astro.js check

desktop-build: desktop-setup | $(RUN_DIR) $(LOCK_DIR)
	@set -euo pipefail; \
	flock -o "$(LOCK_DIR)/build.lock" bash -lc 'set -euo pipefail; \
		compute_build_sig() { \
			{ \
				printf "node=%s\n" "$$("$(NODE)" -v)"; \
				cd "$(DESKTOP_DIR_ABS)"; \
				find src public electron -type f 2>/dev/null | sort | xargs -r sha256sum; \
				sha256sum package.json astro.config.mjs tsconfig.json bun.lock; \
			} | sha256sum | cut -d" " -f1; \
		}; \
		signature="$$(compute_build_sig)"; \
		if [[ -f "$(BUILD_STAMP_FILE)" && -d "$(DESKTOP_DIR_ABS)/dist" && "$$(cat "$(BUILD_STAMP_FILE)")" == "$$signature" ]]; then \
			echo "desktop build already up to date"; \
			exit 0; \
		fi; \
		rm -rf "$(DESKTOP_DIR_ABS)/dist" "$(DESKTOP_DIR_ABS)/release"; \
		cd "$(DESKTOP_DIR_ABS)" && "$(NODE)" ./node_modules/astro/astro.js build; \
		printf "%s\n" "$$signature" > "$(BUILD_STAMP_FILE)"'

electron-dev: desktop-setup
	cd "$(DESKTOP_DIR_ABS)" && "$(BUN)" run electron:dev

electron-start: desktop-build
	cd "$(DESKTOP_DIR_ABS)" && "$(BUN)" run electron:start

electron-prepare: desktop-build
	cd "$(DESKTOP_DIR_ABS)" && "$(BUN)" run electron:prepare

status: | $(RUN_DIR)
	@set -euo pipefail; \
	if [[ -f "$(APP_PID_FILE)" ]]; then \
		pid="$$(cat "$(APP_PID_FILE)")"; \
		if kill -0 "$$pid" 2>/dev/null; then \
			echo "$(APP_NAME) is running (pid $$pid) on $(APP_URL)"; \
			exit 0; \
		fi; \
		echo "$(APP_NAME) has a stale pid file ($$pid)"; \
		exit 1; \
	fi; \
	echo "$(APP_NAME) is stopped"; \
	exit 1

wait:
	@set -euo pipefail; \
	for _ in $$(seq 1 60); do \
		if "$(CURL)" -fsS "$(APP_URL)" >/dev/null 2>&1; then \
			echo "$(APP_NAME) is ready at $(APP_URL)"; \
			exit 0; \
		fi; \
		sleep 1; \
	done; \
	echo "Timed out waiting for $(APP_NAME) at $(APP_URL)"; \
	exit 1

up: desktop-setup | $(RUN_DIR) $(LOCK_DIR)
	@set -euo pipefail; \
	flock -o "$(LOCK_DIR)/app.lock" bash -lc 'set -euo pipefail; \
		if [[ -f "$(APP_PID_FILE)" ]]; then \
			pid="$$(cat "$(APP_PID_FILE)")"; \
			if kill -0 "$$pid" 2>/dev/null; then \
				echo "$(APP_NAME) already running (pid $$pid) at $(APP_URL)"; \
				exit 0; \
			fi; \
			rm -f "$(APP_PID_FILE)"; \
		fi; \
		compute_build_sig() { \
			{ \
				printf "node=%s\n" "$$("$(NODE)" -v)"; \
				cd "$(DESKTOP_DIR_ABS)"; \
				find src public electron -type f 2>/dev/null | sort | xargs -r sha256sum; \
				sha256sum package.json astro.config.mjs tsconfig.json bun.lock; \
			} | sha256sum | cut -d" " -f1; \
		}; \
		signature="$$(compute_build_sig)"; \
		if [[ ! -f "$(BUILD_STAMP_FILE)" || ! -d "$(DESKTOP_DIR_ABS)/dist" || "$$(cat "$(BUILD_STAMP_FILE)")" != "$$signature" ]]; then \
			rm -rf "$(DESKTOP_DIR_ABS)/dist" "$(DESKTOP_DIR_ABS)/release"; \
			cd "$(DESKTOP_DIR_ABS)"; \
			"$(NODE)" ./node_modules/astro/astro.js build >"$(APP_LOG_FILE)" 2>&1; \
			printf "%s\n" "$$signature" > "$(BUILD_STAMP_FILE)"; \
		else \
			: >"$(APP_LOG_FILE)"; \
		fi; \
		setsid bash -lc '"'"'cd "$(DESKTOP_DIR_ABS)"; exec "$(NODE)" ./node_modules/astro/astro.js preview --host "$(APP_HOST)" --port "$(APP_PORT)"'"'"' >>"$(APP_LOG_FILE)" 2>&1 < /dev/null & \
		pid="$$!"; \
		echo "$$pid" >"$(APP_PID_FILE)"; \
		echo "Started $(APP_NAME) (pid $$pid), waiting for $(APP_URL)"; \
		for _ in $$(seq 1 60); do \
			if ! kill -0 "$$pid" 2>/dev/null; then \
				echo "$(APP_NAME) exited during startup"; \
				tail -n 50 "$(APP_LOG_FILE)" || true; \
				rm -f "$(APP_PID_FILE)"; \
				exit 1; \
			fi; \
			if "$(CURL)" -fsS "$(APP_URL)" >/dev/null 2>&1; then \
				echo "$(APP_NAME) is ready at $(APP_URL)"; \
				exit 0; \
			fi; \
			sleep 1; \
		done; \
		echo "Timed out waiting for $(APP_NAME) at $(APP_URL)"; \
		tail -n 50 "$(APP_LOG_FILE)" || true; \
		exit 1'

down: | $(RUN_DIR) $(LOCK_DIR)
	@set -euo pipefail; \
	flock -o "$(LOCK_DIR)/app.lock" bash -lc 'set -euo pipefail; \
		if [[ ! -f "$(APP_PID_FILE)" ]]; then \
			echo "$(APP_NAME) is already stopped"; \
			exit 0; \
		fi; \
		pid="$$(cat "$(APP_PID_FILE)")"; \
		if kill -0 "$$pid" 2>/dev/null; then \
			kill "$$pid" 2>/dev/null || true; \
			for _ in $$(seq 1 20); do \
				if ! kill -0 "$$pid" 2>/dev/null; then break; fi; \
				sleep 0.25; \
			done; \
			if kill -0 "$$pid" 2>/dev/null; then \
				kill -9 "$$pid" 2>/dev/null || true; \
			fi; \
			echo "Stopped $(APP_NAME) (pid $$pid)"; \
		else \
			echo "Removed stale pid file for $(APP_NAME) (pid $$pid)"; \
		fi; \
		rm -f "$(APP_PID_FILE)"'

restart:
	@$(MAKE) down
	@$(MAKE) up

re:
	@$(MAKE) down
	@$(MAKE) up

logs: | $(RUN_DIR)
	@touch "$(APP_LOG_FILE)"
	@tail -n 80 -f "$(APP_LOG_FILE)"

clean: clean-python clean-build clean-runtime

clean-python:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-build:
	rm -rf "$(DESKTOP_DIR)/dist"
	rm -rf "$(DESKTOP_DIR)/release"

clean-runtime: down
	rm -f "$(APP_LOG_FILE)"
	rm -f "$(APP_PID_FILE)"
	rm -f "$(SETUP_STAMP_FILE)"
	rm -f "$(BUILD_STAMP_FILE)"
	rm -rf "$(LOCK_DIR)"

clean-app:
	rm -rf "$(DESKTOP_DIR_ABS)/node_modules"
