.PHONY: install test test-cov lint lint-fix format type-check security run diagram clean docker-build benchmark retrain db-migrate profile audit

install:
	pip install -r requirements.txt
	pip install pytest-cov mypy bandit ruff

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short \
		--cov=app \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-fail-under=75

lint:
	ruff check .
	ruff format --check .

lint-fix:
	ruff check . --fix
	ruff check . --fix --unsafe-fixes
	ruff format .

format:
	ruff format .

type-check:
	mypy app/ --ignore-missing-imports --no-error-summary

security:
	bandit -r app/ pipelines/ -ll --exit-zero

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

diagram:
	python scripts/generate_diagram.py

docker-build:
	docker build -t climate-pulse:latest .

benchmark:
	python scripts/load_test.py

retrain:
	curl -s -X POST http://localhost:8000/api/v1/retrain | python3 -m json.tool

db-migrate:
	alembic upgrade head

profile:
	python -m cProfile -s cumulative scripts/evaluate.py 2>&1 | head -30

audit:
	pip-audit --desc

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache dist build *.egg-info htmlcov .coverage coverage.xml models/*.joblib models/metrics.json
