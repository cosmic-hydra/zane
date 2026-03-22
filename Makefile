PYTHON ?= python
GO ?= go

.PHONY: test lint format build-go-fastsearch setup-workspace setup-venv test-cerebras bootstrap-dashboard

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check drug_discovery tests

format:
	$(PYTHON) -m black drug_discovery tests

build-go-fastsearch:
	mkdir -p tools/bin
	cd tools/go/fastsearch && $(GO) build -o ../../bin/zane-fastsearch .

setup-workspace:
	bash scripts/setup_workspace.sh

setup-venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

test-cerebras:
	$(PYTHON) scripts/cerebras_chat_test.py

bootstrap-dashboard:
	bash scripts/bootstrap_and_dashboard.sh
