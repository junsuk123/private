# Convenience wrapper around the Raspberry Pi bootstrap/run scripts.
# Run from the repo root:  make -C packaging/raspberrypi install
#
# The single command that installs dependencies AND builds everything:
#     make -C packaging/raspberrypi install

HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: install install-no-apt build-native run verify service-install help

## install: one command — OS deps + venv + Python deps + optional native build + verify
install:
	bash "$(HERE)bootstrap.sh"

## install-no-apt: same as install but skip apt (system packages already present)
install-no-apt:
	bash "$(HERE)bootstrap.sh" --no-apt

## build-native: install a Rust toolchain (if needed) and build the native screening core
build-native:
	bash "$(HERE)bootstrap.sh" --no-apt --with-rust

## run: launch the app (CPU-only, headless)
run:
	bash "$(HERE)run.sh"

## verify: re-run the CPU-only runtime verification
verify:
	PYTHONPATH=src ONTOLOGY_ACCELERATOR=CPU ../../.venv-pi/bin/python "$(HERE)verify_pi.py"

## service-install: install + enable the systemd service (needs sudo)
service-install:
	sudo cp "$(HERE)personal-investment.service" /etc/systemd/system/
	sudo systemctl daemon-reload
	sudo systemctl enable --now personal-investment.service
	@echo "Service enabled. Logs: journalctl -u personal-investment.service -f"

## help: list targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## //'
