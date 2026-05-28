# Contributing to Climate-Pulse

## Development Setup

```bash
git clone https://github.com/atharvadevne123/Climate-Pulse
cd Climate-Pulse
pip install -r requirements.txt
pip install pytest-cov mypy bandit ruff
cp .env.example .env
make test
```

## Code Standards

- Python 3.11+, type annotations on all public functions
- Google-style docstrings on all classes and public methods
- `ruff check . && ruff format --check .` must pass before submitting a PR
- All new features require tests with ≥ 75% coverage on the `app/` package
- Use `from __future__ import annotations` in all source files

## Running Tests

```bash
# All tests
make test

# With coverage report
make test-cov

# Single module
pytest tests/test_api.py -v
```

## Type Checking

```bash
make type-check   # mypy app/
```

## Security Scanning

```bash
make security     # bandit -r app/ pipelines/
```

## Pull Request Process

1. Fork the repo and create a feature branch (`git checkout -b feat/my-feature`)
2. Write tests for all new functionality
3. Run `make lint && make test-cov` — both must pass with coverage ≥ 75%
4. Submit a PR with a clear description referencing any related issues

## Adding a New Endpoint

1. Add the route to `app/main.py` with a Pydantic response model
2. Add corresponding business logic to the appropriate module under `app/`
3. Write at least 3 tests: happy path, edge case, and error case
4. Update `README.md` API Reference section
5. Add a CHANGELOG entry under the next version

## Reporting Issues

Open a GitHub Issue with:
- Reproduction steps (minimal code snippet if possible)
- Expected vs actual behaviour
- Python version, OS, and dependency versions (`pip freeze`)
