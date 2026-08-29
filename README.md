# Movie Recommendation

Mood-based movie recommender for **Telugu, Hindi, English, and Kannada** films.

## Why Streamlit was replaced

Streamlit reloaded a ~25k-row model on each session and felt slow. The app is now:

- **FastAPI** backend (model loaded once, cached on disk)
- **Static frontend** (same layout: mood tab + similar tab)

## Quick start

```bash
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:8000**

First launch builds a cache under `.cache/recommender/` (~few seconds). Later starts are much faster.

## Project layout

| Path | Role |
|------|------|
| `app.py` | FastAPI server + static UI host |
| `recommender.py` | Recommendation engine (cached features) |
| `static/` | Frontend (HTML / CSS / JS) |
| `model_selection.ipynb` | Data pipeline + evaluation |
| `movies_features.csv` | Feature matrix |

## API

| Endpoint | Purpose |
|----------|---------|
| `GET /api/health` | Ready / starting / shutting down |
| `GET /api/info` | Dataset stats |
| `POST /api/recommend/mood` | Mood recommendations |
| `POST /api/recommend/similar` | Similar movies |
| `GET /api/suggest?q=` | Title suggestions |

Requests are abort-safe: navigating away or clicking again cancels the previous call. Server shutdown sets a flag so mid-flight work stops cleanly.

## Rebuild data (optional)

Run `model_selection.ipynb` (keep `RUN_TMDB_ENRICH = False` unless re-fetching). Then delete `.cache/recommender/` so the app rebuilds its fast cache.

## Notes

- `.env` / `.cache/` are gitignored
- UI matches the previous Streamlit structure (tabs, filters, results table)
