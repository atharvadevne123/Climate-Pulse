# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.2.x   | Yes       |
| 1.1.x   | No        |
| 1.0.x   | No        |

## Reporting a Vulnerability

Please email **devneatharva@gmail.com** with the subject "SECURITY: Climate-Pulse" and include:

- A description of the vulnerability
- Steps to reproduce
- Impact assessment
- Any suggested mitigation

We aim to respond within **72 hours** and coordinate disclosure after a fix is available.
Expect a resolution timeline within **14 business days** for high-severity issues.

## Security Measures

- Input validation via Pydantic on all API endpoints
- Rate limiting middleware (200 req/min per IP)
- No secrets committed to the repository (use `.env`)
- SQL injection protection via SQLAlchemy ORM
- Dependency pinning in `requirements.txt`

## Additional Security Controls

### Station ID Validation
- Station IDs are limited to 64 printable ASCII characters
- Empty or whitespace-only station IDs are rejected with 422

### Dependency Auditing
```bash
pip install pip-audit
pip-audit
```

### Static Analysis
```bash
# Security scanning with bandit
make security

# Lint checks with ruff
make lint
```

### CORS Configuration
The default `allow_origins=["*"]` is suitable for public read-only APIs.
For production deployments with write endpoints, restrict origins:
```bash
CORS_ORIGINS=https://your-frontend.example.com uvicorn app.main:app
```

### Connection Pool Security
Database connections use `pool_pre_ping=True` to detect stale connections.
Configurable via `DB_POOL_SIZE`, `DB_MAX_OVERFLOW`, `DB_POOL_TIMEOUT` env vars.
