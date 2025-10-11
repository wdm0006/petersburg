.PHONY: help install test examples case-studies clean

help:
	@echo "Petersburg Framework - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install        - Install the package in development mode"
	@echo "  test           - Run tests"
	@echo "  examples       - Run all example scripts"
	@echo "  case-studies   - Run all case study analyses"
	@echo "  clean          - Remove build artifacts and caches"

install:
	uv venv
	uv pip install -e .

test:
	.venv/bin/python3 -m pytest tests/

examples:
	@echo "Running example scripts..."
	@echo ""
	@echo "=== St. Petersburg Paradox ==="
	.venv/bin/python3 examples/stpetersburg.py
	@echo ""
	@echo "=== Two Envelope Problem ==="
	.venv/bin/python3 examples/two_envelope_problem.py
	@echo ""
	@echo "=== Necktie Paradox ==="
	.venv/bin/python3 examples/necktie_paradox.py

case-studies:
	@echo "Running case study analyses..."
	@echo ""
	@echo "=== Drug Development ==="
	.venv/bin/python3 examples/case_studies/drug_development.py
	@echo ""
	@echo "=== Startup Funding ==="
	.venv/bin/python3 examples/case_studies/startup_funding.py
	@echo ""
	@echo "=== Product Launch ==="
	.venv/bin/python3 examples/case_studies/product_launch.py
	@echo ""
	@echo "=== Litigation Strategy ==="
	.venv/bin/python3 examples/case_studies/litigation_strategy.py

clean:
	rm -rf .venv
	rm -rf dist
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
