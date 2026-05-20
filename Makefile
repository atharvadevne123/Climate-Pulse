.PHONY: install test lint run diagram clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short 2>&1 | tail -60

lint:
	ruff check .

lint-fix:
	ruff check . --fix
	ruff check . --fix --unsafe-fixes

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

diagram:
	python scripts/generate_diagram.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache dist build *.egg-info
