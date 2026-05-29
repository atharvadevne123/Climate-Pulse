# Contributing to Climate-Pulse

We welcome bug reports, feature requests, and pull requests. This document describes the development workflow, coding standards, and PR process.

## Development Setup

```bash
git clone https://github.com/atharvadevne123/Climate-Pulse
cd Climate-Pulse
pip install -r requirements.txt
pip install pytest-cov mypy bandit ruff
cp .env.example .env   # fill in DB credentials if using PostgreSQL
make test              # run the full test suite
```

### Using Docker (recommended for full-stack dev)

```bash
docker-compose up --build
# API at http://localhost:8000, PostgreSQL at localhost:5432
```

---

## Code Standards

| Rule | Detail |
|------|--------|
| Python | 3.11+ |
| Annotations | Type annotations on **all** public functions and class attributes |
| Docstrings | Google-style on all classes and public methods |
| Linting | `ruff check .` must exit 0 |
| Formatting | `ruff format --check .` must exit 0 |
| Coverage | New code must keep `app/` coverage ≥ 75% |
| Imports | `from __future__ import annotations` in all source files |

---

## Running Tests

```bash
# Full test suite
make test

# With coverage report (HTML + terminal)
make test-cov

# Single module
pytest tests/test_api.py -v
pytest tests/test_features.py -v

# Only fast unit tests (exclude retrain)
pytest tests/ -v -k "not retrain"
```

---

## Type Checking and Security

```bash
make type-check   # mypy app/ --ignore-missing-imports
make security     # bandit -r app/ pipelines/ -ll
```

---

## Adding a New Feature

1. **New endpoint**: Add route to `app/main.py` with Pydantic response model
2. **Business logic**: Add to the appropriate module (`app/monitoring.py`, `app/model.py`, etc.)
3. **Tests**: Write ≥ 3 tests — happy path, edge case (boundary values), error case
4. **Docs**: Update `README.md` API Reference and `CHANGELOG.md`
5. **Migration** (if DB schema changes): Create a new Alembic revision in `alembic/versions/`

---

## Adding a New Feature Transformer

1. Subclass `BaseEstimator, TransformerMixin` in `app/features.py`
2. Implement `fit()` (return self) and `transform()` (return modified DataFrame copy)
3. Add as a new stage in `build_feature_pipeline()` — update its docstring
4. Write tests in `tests/test_features.py` using the `sample_df` fixture
5. Update the pipeline table in `README.md`

---

## Pull Request Process

1. Fork the repo and create a branch: `git checkout -b feat/my-feature`
2. Implement the change with tests
3. Run `make lint && make test-cov` — both must pass
4. Submit a PR with:
   - A clear one-line title (`feat:`, `fix:`, `docs:`, `chore:`)
   - Description of what changed and why
   - Link to any related issues

---

## Commit Message Convention

Use the Angular commit format:

```
<type>(<scope>): <short description>

Types: feat, fix, refactor, test, docs, chore, perf, ci
Example: feat(monitoring): add get_station_stats aggregation function
```

---

## Reporting Issues

Open a GitHub Issue with:
- Reproduction steps (minimal code snippet if possible)
- Expected vs actual behaviour
- Python version, OS, and output of `pip show climate-pulse`
