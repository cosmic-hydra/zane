PYTHON ?= python
GO ?= go

.PHONY: test lint format build-go-fastsearch setup-workspace setup-venv \
        test-cerebras bootstrap-dashboard install check ci clean \
        type-check test-coverage docker-build

# --- Development ---

install:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) -m pip install -r requirements-ci.txt

test:
	$(PYTHON) -m pytest -q

test-coverage:
	$(PYTHON) -m pytest --cov=drug_discovery --cov-report=term --cov-report=html -q

lint:
	$(PYTHON) -m ruff check drug_discovery tests scripts

format:
	$(PYTHON) -m black drug_discovery tests scripts

type-check:
	$(PYTHON) -m mypy drug_discovery --ignore-missing-imports

check: lint type-check test
	@echo "All checks passed."

ci: lint test
	@echo "CI checks complete."

# --- Build ---

build-go-fastsearch:
	mkdir -p tools/bin
	cd tools/go/fastsearch && $(GO) build -o ../../bin/zane-fastsearch .

docker-build:
	docker build -t zane-drug-discovery:latest .

# --- Setup ---

setup-workspace:
	bash scripts/setup_workspace.sh

setup-venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .
	@echo "Virtual environment ready. Activate with: source .venv/bin/activate"

# --- Dashboard & Testing ---

test-cerebras:
	$(PYTHON) scripts/cerebras_chat_test.py

bootstrap-dashboard:
	bash scripts/bootstrap_and_dashboard.sh

dashboard:
	$(PYTHON) -m drug_discovery.cli dashboard --static

dashboard-live:
	$(PYTHON) -m drug_discovery.cli dashboard --theme neon

# --- Cleanup ---

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .mypy_cache dist build *.egg-info
	@echo "Cleaned build artifacts."
