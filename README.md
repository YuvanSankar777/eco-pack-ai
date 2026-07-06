---
title: EcoPack AI
emoji: 🌿
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# 🌿 EcoPack AI

AI-driven smart packaging recommender. Describe a product in plain words and
EcoPack AI detects its category and fragility (TF-IDF NLP), then ranks the most
sustainable packaging materials with an XGBoost model scored on strength,
recyclability, biodegradability, cost and CO₂.

## Tech stack

Python · Flask · XGBoost · scikit-learn · pandas · gunicorn · (optional) PostgreSQL

## Run locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py            # http://127.0.0.1:5001
```

## Configuration

| Env var        | Purpose                                                            |
|----------------|-------------------------------------------------------------------|
| `PORT`         | Port to serve on (default `5001` locally, `7860` in the container) |
| `DATABASE_URL` | Optional Postgres URL to enable saved product history. If unset, the save step no-ops and recommendations still work. e.g. `postgresql://user:pass@host:5432/eco_pack` |
| `FLASK_DEBUG`  | Set to `1` to enable the Flask debug reloader locally             |

If `DATABASE_URL` is set, the app expects a table:

```sql
CREATE TABLE user_products (
    product_name TEXT PRIMARY KEY,
    category     TEXT,
    fragility    TEXT
);
```

## Deploy (Docker / Hugging Face Spaces)

The included `Dockerfile` builds a production image served by gunicorn on
port `7860`. On Hugging Face Spaces the front-matter above configures a Docker
Space automatically.

```bash
docker build -t ecopack-ai .
docker run -p 7860:7860 ecopack-ai   # http://127.0.0.1:7860
```
