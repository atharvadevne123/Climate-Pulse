# Contributing to Climate-Pulse

## Development Setup

```bash
git clone https://github.com/atharvadevne123/Climate-Pulse
cd Climate-Pulse
pip install -r requirements.txt
cp .env.example .env
make test
```

## Code Standards

- Python 3.11+, type annotations on all public functions
- Google-style docstrings
- `ruff check .` must pass before submitting a PR
- All new features require tests with ≥ 80% coverage

## Pull Request Process

1. Fork the repo and create a feature branch
2. Write tests for all new functionality
3. Run `make lint && make test` — both must pass
4. Submit a PR with a clear description of the change

## Reporting Issues

Open a GitHub Issue with reproduction steps, expected vs actual behaviour, and your environment details.
