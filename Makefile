.PHONY: clean test format check

clean: format check clean-cache

format:
	uv run ruff format .

check:
	uv run ruff check . --fix

clean-cache:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete 2>/dev/null || true

test:
	uv run python -m pytest .
