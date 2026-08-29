"""Fast movie recommender with cached feature matrix."""

from __future__ import annotations

import json
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, normalize

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "movies_features.csv"
CACHE_DIR = ROOT / ".cache" / "recommender"
CACHE_META = CACHE_DIR / "meta.json"
CACHE_FEATURES = CACHE_DIR / "features.npy"
CACHE_TABLE = CACHE_DIR / "movies.pkl"

STANDARD_GENRES = [
    "action", "adventure", "animation", "comedy", "crime", "documentary", "drama",
    "family", "fantasy", "history", "horror", "music", "mystery", "romance",
    "science fiction", "thriller", "tv movie", "war", "western",
]

MOOD_ALIASES = {
    "happy": ["joy", "amusement", "excitement", "optimism", "approval"],
    "sad": ["sadness", "grief", "disappointment", "remorse"],
    "angry": ["anger", "annoyance", "disapproval"],
    "scared": ["fear", "nervousness", "surprise"],
    "fearful": ["fear", "nervousness"],
    "romantic": ["love", "desire"],
    "loving": ["love", "caring", "desire"],
    "curious": ["curiosity", "realization"],
    "confused": ["confusion", "realization"],
    "excited": ["excitement", "joy", "surprise"],
    "surprised": ["surprise", "realization"],
    "tired": ["neutral", "relief"],
    "bored": ["neutral", "disapproval"],
    "relaxed": ["neutral", "relief", "approval"],
    "neutral": ["neutral", "approval"],
    "inspired": ["admiration", "optimism", "approval"],
    "nostalgic": ["sadness", "admiration", "gratitude"],
}

DISPLAY_COLS = ["Title", "Genres", "Moods", "Rating", "release_year", "Language"]


def pretty_title(text: object) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    return " ".join(part.capitalize() for part in value.split())


def _csv_signature(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "mtime": stat.st_mtime, "size": stat.st_size}


class MovieRecommender:
    """Content-based recommender. Loads from disk cache when possible."""

    def __init__(self, path: Path = DATA_PATH, use_cache: bool = True):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Missing {self.path}. Run model_selection.ipynb to build movies_features.csv."
            )

        loaded = False
        if use_cache:
            loaded = self._try_load_cache()
        if not loaded:
            self._build_from_csv()
            if use_cache:
                self._save_cache()

        self.titles_lower = self.df["Title"].astype(str).str.lower().tolist()
        self.available_moods = sorted(
            {
                col[5:]
                for col in self.mood_cols
                if col in self.df.columns and int(self.df[col].sum()) > 0
            }
        )
        self.languages = sorted(
            {
                str(v).lower()
                for v in self.df["Language"].dropna().unique()
                if str(v).strip()
            }
        )
        self.ready = True

    def _try_load_cache(self) -> bool:
        if not (CACHE_META.exists() and CACHE_FEATURES.exists() and CACHE_TABLE.exists()):
            return False
        try:
            meta = json.loads(CACHE_META.read_text(encoding="utf-8"))
            if meta.get("signature") != _csv_signature(self.path):
                return False
            self.features = np.load(CACHE_FEATURES)
            self.df = pd.read_pickle(CACHE_TABLE)
            self.mood_cols = meta["mood_cols"]
            self.genre_matrix = self.df[STANDARD_GENRES].to_numpy(dtype=np.float32)
            self.lang_codes = self.df["Language"].astype(str).str.lower().to_numpy()
            return True
        except Exception:
            return False

    def _save_cache(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_feat = CACHE_FEATURES.with_suffix(".tmp.npy")
        np.save(tmp_feat, self.features)
        tmp_feat.replace(CACHE_FEATURES)
        self.df.to_pickle(CACHE_TABLE)
        CACHE_META.write_text(
            json.dumps(
                {
                    "signature": _csv_signature(self.path),
                    "mood_cols": self.mood_cols,
                    "n_movies": len(self.df),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _build_from_csv(self) -> None:
        raw = pd.read_csv(self.path)
        raw["Moods"] = raw["Moods"].fillna("neutral")
        raw["Title"] = raw["Title"].fillna("").astype(str)
        raw["Genres"] = raw["Genres"].fillna("")
        if "Rating" not in raw.columns:
            raw["Rating"] = 5.0
        raw["Rating"] = pd.to_numeric(raw["Rating"], errors="coerce").fillna(5.0)
        if "Language" not in raw.columns:
            raw["Language"] = "unknown"

        for genre in STANDARD_GENRES:
            if genre not in raw.columns:
                raw[genre] = 0
            raw[genre] = pd.to_numeric(raw[genre], errors="coerce").fillna(0).astype(np.int8)

        mood_tokens = sorted(
            {
                t.strip().lower()
                for moods in raw["Moods"]
                for t in str(moods).split(",")
                if t.strip()
            }
        )
        self.mood_cols = [f"mood_{m}" for m in mood_tokens]
        for col in self.mood_cols:
            if col not in raw.columns:
                mood = col[5:]
                raw[col] = raw["Moods"].apply(lambda text, mood=mood: int(mood in str(text).lower()))
            raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0).astype(np.int8)

        raw = raw[raw[STANDARD_GENRES].sum(axis=1) > 0].reset_index(drop=True)

        keep = [c for c in DISPLAY_COLS if c in raw.columns] + STANDARD_GENRES + self.mood_cols
        keep = list(dict.fromkeys(keep))
        self.df = raw[keep].copy()

        binary_genre = self.df[STANDARD_GENRES].to_numpy(dtype=np.float32) * 4.0
        binary_mood = self.df[self.mood_cols].to_numpy(dtype=np.float32)
        numeric_cols = [
            c
            for c in [
                "has_release_year",
                "release_year",
                "Popularity_Scaled",
                "Vote_Count_Scaled",
                "Rating_Scaled",
            ]
            if c in raw.columns
        ]
        if numeric_cols:
            numeric = MinMaxScaler().fit_transform(
                raw[numeric_cols].fillna(0).to_numpy(dtype=np.float32)
            ).astype(np.float32)
            feats = np.hstack([binary_genre, binary_mood, numeric])
        else:
            feats = np.hstack([binary_genre, binary_mood])

        # L2-normalize once → similarity becomes a fast matrix-vector product
        self.features = normalize(feats, norm="l2", axis=1).astype(np.float32)
        self.genre_matrix = self.df[STANDARD_GENRES].to_numpy(dtype=np.float32)
        self.lang_codes = self.df["Language"].astype(str).str.lower().to_numpy()

    def _scores(self, idx: int) -> np.ndarray:
        return self.features @ self.features[idx]

    def resolve_mood(self, user_mood: str) -> list[str]:
        query = user_mood.strip().lower()
        if not query:
            return []
        direct = get_close_matches(query, self.available_moods, n=1, cutoff=0.75)
        if direct:
            return direct
        alias = get_close_matches(query, list(MOOD_ALIASES.keys()), n=1, cutoff=0.7)
        if alias:
            labels = [
                m
                for m in MOOD_ALIASES[alias[0]]
                if f"mood_{m}" in self.df.columns and self.df[f"mood_{m}"].sum() > 0
            ]
            if labels:
                return labels
        return get_close_matches(query, self.available_moods, n=3, cutoff=0.55)

    def _format_rows(self, indices: list[int]) -> list[dict]:
        rows = []
        for i in indices:
            row = self.df.iloc[i]
            year = row.get("release_year", 0)
            try:
                year_out: int | str = int(year) if float(year) > 1900 else "Unknown"
            except (TypeError, ValueError):
                year_out = "Unknown"
            rows.append(
                {
                    "Title": pretty_title(row["Title"]),
                    "Genres": str(row.get("Genres", "")).title(),
                    "Moods": str(row.get("Moods", "")).title(),
                    "Rating": round(float(row.get("Rating", 0)), 1),
                    "Language": str(row.get("Language", "")).title(),
                    "Year": year_out,
                }
            )
        return rows

    def recommend_by_mood(
        self,
        user_mood: str,
        n: int = 10,
        language: str | None = None,
        min_rating: float = 0.0,
    ) -> tuple[list[str], list[dict]]:
        mapped = self.resolve_mood(user_mood)
        if not mapped:
            return [], []

        mood_score = np.zeros(len(self.df), dtype=np.float32)
        for i, mood in enumerate(mapped):
            col = f"mood_{mood}"
            if col in self.df.columns:
                mood_score += self.df[col].to_numpy(dtype=np.float32) * (len(mapped) - i)

        mask = mood_score > 0
        if language:
            mask &= self.lang_codes == language.lower()
        if min_rating > 0:
            mask &= self.df["Rating"].to_numpy(dtype=np.float32) >= min_rating
        if not mask.any():
            return mapped, []

        candidate_pos = np.flatnonzero(mask)
        # Seed = best mood hit then rating
        ratings = self.df["Rating"].to_numpy(dtype=np.float32)
        seed_local = int(
            np.lexsort(
                (-ratings[candidate_pos], -mood_score[candidate_pos])
            )[-1]
        )
        seed_pos = int(candidate_pos[seed_local])
        scores = self._scores(seed_pos)

        order = sorted(
            candidate_pos.tolist(),
            key=lambda i: (float(scores[i]), float(mood_score[i]), float(ratings[i])),
            reverse=True,
        )[:n]
        return mapped, self._format_rows(order)

    def resolve_title_index(self, title: str) -> int | None:
        query = title.strip().lower()
        if not query:
            return None

        titles = self.titles_lower
        # exact
        exact = [i for i, t in enumerate(titles) if t == query]
        if exact:
            return max(exact, key=lambda i: float(self.df.iloc[i]["Rating"]))

        def match_rank(t: str) -> float:
            if t == query:
                return 1000.0
            if t.startswith(query + " ") or t.startswith(query + "-") or t.startswith(query + ":"):
                return 900.0
            if t.startswith(query):
                nxt = t[len(query) : len(query) + 1]
                if nxt.isalpha():
                    return 50.0
                return 850.0
            tokens = query.split()
            if tokens and all(tok in t for tok in tokens):
                return 700.0 + (10.0 if t.startswith(tokens[0]) else 0.0)
            if query in t:
                return 400.0
            return -1.0

        ranks = [(i, match_rank(t)) for i, t in enumerate(titles)]
        strong = [(i, r) for i, r in ranks if r >= 400]
        if not strong:
            strong = [(i, r) for i, r in ranks if r > 0]
        if strong:
            strong.sort(key=lambda x: (x[1], float(self.df.iloc[x[0]]["Rating"])), reverse=True)
            return strong[0][0]

        close = get_close_matches(query, titles, n=8, cutoff=0.65)
        close = [c for c in close if len(c) >= max(4, len(query) - 1) and len(c) >= len(query) * 0.8]
        if not close:
            return None
        for i, t in enumerate(titles):
            if t == close[0]:
                return i
        return None

    def recommend_similar(self, title: str, n: int = 10) -> list[dict]:
        pos = self.resolve_title_index(title)
        if pos is None:
            return []

        scores = self._scores(pos).copy()
        seed_lang = self.lang_codes[pos]
        seed_genres = self.genre_matrix[pos]
        genre_norm = float(seed_genres.sum()) or 1.0
        overlap = (self.genre_matrix @ seed_genres) / genre_norm
        scores += 0.05 * overlap
        scores += 0.08 * (self.lang_codes == seed_lang).astype(np.float32)
        scores[pos] = -1.0

        order = np.argpartition(-scores, min(n, len(scores) - 1))[:n]
        order = order[np.argsort(-scores[order])]
        return self._format_rows(order.tolist())

    def search_titles(self, query: str, limit: int = 8) -> list[str]:
        idx = self.resolve_title_index(query)
        if idx is None:
            # fallback: contains scan
            q = query.strip().lower()
            hits = [
                pretty_title(self.df.iloc[i]["Title"])
                for i, t in enumerate(self.titles_lower)
                if q in t
            ][:limit]
            return hits

        # Return top ranked matches from resolve ranking logic
        q = query.strip().lower()

        def rank(t: str) -> float:
            if t == q:
                return 1000
            if t.startswith(q + " ") or t.startswith(q + "-") or t.startswith(q + ":"):
                return 900
            if t.startswith(q) and (len(t) == len(q) or not t[len(q) : len(q) + 1].isalpha()):
                return 850
            if all(tok in t for tok in q.split()):
                return 700
            if q in t:
                return 400
            return -1

        scored = [(i, rank(t)) for i, t in enumerate(self.titles_lower) if rank(t) >= 400]
        scored.sort(key=lambda x: (x[1], float(self.df.iloc[x[0]]["Rating"])), reverse=True)
        return [pretty_title(self.df.iloc[i]["Title"]) for i, _ in scored[:limit]]

    def info(self) -> dict:
        return {
            "movies": len(self.df),
            "languages": [lang.title() for lang in self.languages],
            "moods": self.available_moods,
            "ready": True,
        }


def build_cache(path: Path = DATA_PATH) -> MovieRecommender:
    """Force-rebuild disk cache (safe to interrupt; writes atomically)."""
    return MovieRecommender(path=path, use_cache=False)
