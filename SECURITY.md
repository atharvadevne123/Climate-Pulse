# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | Yes       |

## Reporting a Vulnerability

Please email **devneatharva@gmail.com** with the subject "SECURITY: Climate-Pulse" and include:

- A description of the vulnerability
- Steps to reproduce
- Impact assessment
- Any suggested mitigation

We aim to respond within 72 hours and will coordinate disclosure after a fix is available.

## Security Measures

- Input validation via Pydantic on all API endpoints
- Rate limiting middleware (200 req/min per IP)
- No secrets committed to the repository (use `.env`)
- SQL injection protection via SQLAlchemy ORM
- Dependency pinning in `requirements.txt`
