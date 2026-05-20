# Deployment Guide

## Local Development

```bash
cp .env.example .env
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Docker (Recommended)

```bash
docker-compose up --build -d
docker-compose logs -f api
```

Access the API at `http://localhost:8000/docs`

## Database Migrations

```bash
# Apply migrations
alembic upgrade head

# Rollback one step
alembic downgrade -1
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./climate_pulse.db` | DB connection string |
| `MODEL_DIR` | `./models` | Directory to store trained models |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `RATE_LIMIT_PER_MINUTE` | `200` | Max requests per client IP per minute |

## Production Checklist

- [ ] Set `DATABASE_URL` to a PostgreSQL connection string
- [ ] Mount a persistent volume for `MODEL_DIR`
- [ ] Set `LOG_LEVEL=WARNING` in production
- [ ] Configure HTTPS/TLS termination (nginx or load balancer)
- [ ] Enable Airflow DAG for weekly retraining
- [ ] Set up alerting on drift detection endpoint

## Health Check

```bash
curl http://localhost:8000/api/v1/health
# {"status": "ok", "version": "1.0.0"}
```
