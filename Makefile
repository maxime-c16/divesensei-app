SHELL := /bin/bash


PYTHON ?= python3
VENV ?= .venv
DESKTOP_DIR ?= apps/desktop
DESKTOP_DIR_ABS := $(abspath $(DESKTOP_DIR))
MOBILE_DIR ?= apps/mobile
MOBILE_DIR_ABS := $(abspath $(MOBILE_DIR))
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
NODE ?= $(shell if [[ -x /usr/local/opt/node@22/bin/node ]]; then printf '%s' "/usr/local/opt/node@22/bin/node"; else command -v node; fi)
NPM ?= $(shell if [[ -x /usr/local/opt/node@22/bin/npm ]]; then printf '%s' "/usr/local/opt/node@22/bin/npm"; else command -v npm; fi)
FLOCK ?= $(shell command -v flock 2>/dev/null || command -v gflock 2>/dev/null || printf '%s' "/usr/local/opt/util-linux/bin/flock")
CURL ?= curl
DEVELOPER_DIR ?= /Applications/Xcode.app/Contents/Developer
CAP_SYNC_ENV = RUBYOPT='-EUTF-8:UTF-8' LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
IOS_SCHEME ?= App
IOS_WORKSPACE ?= $(MOBILE_DIR_ABS)/ios/App/App.xcworkspace
IOS_SIMULATOR_ID ?= 10E4E9DA-CA56-45C7-B846-3A63D534BAF1
IOS_APP_BUNDLE ?= /Users/$(USER)/Library/Developer/Xcode/DerivedData/App-fprmnnhpsvflfgcrzusflztskrms/Build/Products/Debug-iphonesimulator/App.app
IOS_BUNDLE_ID ?= com.divesensei.mobile

.PHONY: help venv install compile smoke-help desktop-setup desktop-check desktop-build \
	electron-dev electron-start electron-prepare mobile-setup mobile-build mobile-sync-ios \
	mobile-open-ios mobile-xcode-build mobile-web-refresh mobile-fast mobile-sim-install mobile-sim-launch mobile-sim-relaunch mobile-sim-reinstall \
	mobile-sim-screenshot mobile-review-reset review-session review-session-open status wait up down restart re logs clean \
	clean-runtime clean-build clean-python clean-app

help:
	@printf "Targets:\n"
	@printf "  make venv           Create the local Python virtualenv\n"
	@printf "  make install        Install the Python package in editable mode\n"
	@printf "  make compile        Byte-compile the Python source tree\n"
	@printf "  make smoke-help     Check the CLI help surfaces\n"
	@printf "  make desktop-setup  Install Bun dependencies for the Astro app\n"
	@printf "  make desktop-check  Run Astro type/config checks\n"
	@printf "  make desktop-build  Build the Astro desktop app\n"
	@printf "  make mobile-setup   Install Bun dependencies for the Capacitor mobile app\n"
	@printf "  make mobile-build   Build the mobile web bundle\n"
	@printf "  make mobile-sync-ios Sync the Capacitor iOS project\n"
	@printf "  make mobile-open-ios Open the iOS workspace in Xcode\n"
	@printf "  make mobile-xcode-build Build the iOS app for the configured simulator\n"
	@printf "  make mobile-web-refresh Build + sync mobile web assets into the Xcode project\n"
	@printf "  make mobile-sim-relaunch Relaunch the installed app on the simulator\n"
	@printf "  make mobile-fast    Incremental iOS rebuild + reinstall for fast UI iteration\n"
	@printf "  make mobile-sim-install Install the built app on the configured simulator\n"
	@printf "  make mobile-sim-launch Launch the app on the configured simulator\n"
	@printf "  make mobile-sim-reinstall Rebuild, install, and relaunch the app on the simulator\n"
	@printf "  make mobile-sim-screenshot Capture a simulator screenshot to /tmp/divesensei-sim.png\n"
	@printf "  make mobile-review-reset Clear simulator decision state for the selected session\n"
	@printf "  make review-session VIDEO_PATH=/abs/video.mov Prepare an evaluation session, export review artifacts, and print the Review URL\n"
	@printf "  make review-session-open VIDEO_PATH=/abs/video.mov Prepare the session and open the desktop Review URL in the browser\n"
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
	"$(FLOCK)" -o "$(LOCK_DIR)/setup.lock" bash -lc 'set -euo pipefail; \
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
		addon_ok() { \
			cd "$(DESKTOP_DIR_ABS)" && "$(NODE)" -e "require(\"better-sqlite3\");" >/dev/null 2>&1; \
		}; \
		if [[ -f "$(SETUP_STAMP_FILE)" && -f "$$addon_path" && "$$(cat "$(SETUP_STAMP_FILE)")" == "$$signature" ]] && addon_ok; then \
			echo "desktop dependencies already up to date"; \
			exit 0; \
		fi; \
		cd "$(DESKTOP_DIR_ABS)" && "$(BUN)" install; \
		export PATH="$(dir $(NODE)):$$PATH"; \
		cd "$(DESKTOP_DIR_ABS)/node_modules/better-sqlite3"; \
		"$(NODE)" ../node-gyp/bin/node-gyp.js rebuild --release; \
		addon_ok || { echo "better-sqlite3 native module ABI check failed after rebuild"; exit 1; }; \
		printf "%s\n" "$$signature" > "$(SETUP_STAMP_FILE)"'

desktop-check: desktop-setup
	cd "$(DESKTOP_DIR_ABS)" && "$(NODE)" ./node_modules/astro/astro.js check

desktop-build: desktop-setup | $(RUN_DIR) $(LOCK_DIR)
	@set -euo pipefail; \
	"$(FLOCK)" -o "$(LOCK_DIR)/build.lock" bash -lc 'set -euo pipefail; \
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

mobile-setup:
	@set -euo pipefail; \
	test -x "$(BUN)" || { echo "bun not found at $(BUN)"; exit 1; }; \
	cd "$(MOBILE_DIR_ABS)" && "$(BUN)" install

mobile-build: mobile-setup
	cd "$(MOBILE_DIR_ABS)" && "$(BUN)" run build

mobile-sync-ios: mobile-build
	cd "$(MOBILE_DIR_ABS)" && env $(CAP_SYNC_ENV) npx cap sync ios

mobile-open-ios: mobile-sync-ios
	cd "$(MOBILE_DIR_ABS)" && env $(CAP_SYNC_ENV) npx cap open ios

mobile-xcode-build: mobile-sync-ios
	DEVELOPER_DIR="$(DEVELOPER_DIR)" xcodebuild \
		-workspace "$(IOS_WORKSPACE)" \
		-scheme "$(IOS_SCHEME)" \
		-configuration Debug \
		-destination 'id=$(IOS_SIMULATOR_ID)' \
		build

mobile-web-refresh: mobile-build
	cd "$(MOBILE_DIR_ABS)" && env $(CAP_SYNC_ENV) npx cap sync ios

mobile-fast: mobile-xcode-build
	xcrun simctl terminate "$(IOS_SIMULATOR_ID)" "$(IOS_BUNDLE_ID)" || true
	xcrun simctl install "$(IOS_SIMULATOR_ID)" "$(IOS_APP_BUNDLE)"
	xcrun simctl launch "$(IOS_SIMULATOR_ID)" "$(IOS_BUNDLE_ID)"

mobile-sim-install:
	xcrun simctl install "$(IOS_SIMULATOR_ID)" "$(IOS_APP_BUNDLE)"

mobile-sim-launch:
	xcrun simctl launch "$(IOS_SIMULATOR_ID)" "$(IOS_BUNDLE_ID)"

mobile-sim-relaunch:
	xcrun simctl terminate "$(IOS_SIMULATOR_ID)" "$(IOS_BUNDLE_ID)" || true
	xcrun simctl launch "$(IOS_SIMULATOR_ID)" "$(IOS_BUNDLE_ID)"

mobile-sim-reinstall: mobile-xcode-build
	xcrun simctl terminate "$(IOS_SIMULATOR_ID)" "$(IOS_BUNDLE_ID)" || true
	xcrun simctl uninstall "$(IOS_SIMULATOR_ID)" "$(IOS_BUNDLE_ID)" || true
	xcrun simctl install "$(IOS_SIMULATOR_ID)" "$(IOS_APP_BUNDLE)"
	xcrun simctl launch "$(IOS_SIMULATOR_ID)" "$(IOS_BUNDLE_ID)"

mobile-sim-screenshot:
	xcrun simctl io "$(IOS_SIMULATOR_ID)" screenshot /tmp/divesensei-sim.png
	@printf "Saved /tmp/divesensei-sim.png\n"

mobile-review-reset:
	@set -euo pipefail; \
	test -n "$${SESSION_ID:-}" || { echo "Set SESSION_ID=<session-id> to clear review decisions"; exit 1; }; \
	container="$$(xcrun simctl get_app_container "$(IOS_SIMULATOR_ID)" "$(IOS_BUNDLE_ID)" data)"; \
	printf '[]' > "$$container/Library/Application Support/DiveSenseiMobile/decisions/$${SESSION_ID}.json"; \
	printf "Cleared review decisions for %s\n" "$${SESSION_ID}"

review-session: up
	@set -euo pipefail; \
	test -n "$${VIDEO_PATH:-}" || { echo "Set VIDEO_PATH=/abs/path/to/video"; exit 1; }; \
	test -f "$${VIDEO_PATH}" || { echo "Video not found: $${VIDEO_PATH}"; exit 1; }; \
	PROFILE="$${PROFILE:-long-session}"; \
	DETECTOR_ID="$${DETECTOR_ID:-audio_v2_pcen_classifier}"; \
	video_abs="$$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$${VIDEO_PATH}")"; \
	base_name="$$(python3 -c 'from pathlib import Path; import re, sys; name=Path(sys.argv[1]).stem; print(re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "session")' "$$video_abs")"; \
	marker="$$(mktemp)"; \
	touch "$$marker"; \
	PYTHONPATH=src .venv/bin/python -m divesensei.cli evaluate-session "$$video_abs" --profile "$$PROFILE" --detector-id "$$DETECTOR_ID"; \
	session_dir="$$(find outputs -maxdepth 1 -type d -name 'evaluation_*' -newer "$$marker" -print | sort | tail -n 1)"; \
	rm -f "$$marker"; \
	if [[ -z "$$session_dir" ]]; then session_dir="$$(ls -dt outputs/evaluation_* 2>/dev/null | head -n 1)"; fi; \
	test -n "$$session_dir" || { echo "Unable to determine session output dir for $$video_abs"; exit 1; }; \
	PYTHONPATH=src .venv/bin/python -m divesensei.cli export-evaluation-review "$$session_dir"; \
	PYTHONPATH=src .venv/bin/python -m divesensei.cli export-event-review-support "$$session_dir"; \
	session_id="$$(basename "$$session_dir")"; \
	printf "SESSION_DIR=%s\n" "$$session_dir"; \
	printf "REVIEW_URL=%s/?session=%s&tab=1\n" "$(APP_URL)" "$$session_id"

review-session-open: review-session
	@set -euo pipefail; \
	test -n "$${VIDEO_PATH:-}" || { echo "Set VIDEO_PATH=/abs/path/to/video"; exit 1; }; \
	video_abs="$$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$${VIDEO_PATH}")"; \
	base_name="$$(python3 -c 'from pathlib import Path; import re, sys; name=Path(sys.argv[1]).stem; print(re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "session")' "$$video_abs")"; \
	session_dir="$$(ls -dt outputs/evaluation_"$$base_name"_* 2>/dev/null | head -n 1)"; \
	test -n "$$session_dir" || { echo "No prepared session found for $$video_abs"; exit 1; }; \
	session_id="$$(basename "$$session_dir")"; \
	open "$(APP_URL)/?session=$$session_id&tab=1"

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
	"$(FLOCK)" -o "$(LOCK_DIR)/app.lock" bash -lc 'set -euo pipefail; \
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
		cd "$(DESKTOP_DIR_ABS)"; \
		nohup "$(NODE)" ./node_modules/astro/astro.js preview --host "$(APP_HOST)" --port "$(APP_PORT)" >>"$(APP_LOG_FILE)" 2>&1 < /dev/null & \
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
	"$(FLOCK)" -o "$(LOCK_DIR)/app.lock" bash -lc 'set -euo pipefail; \
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
