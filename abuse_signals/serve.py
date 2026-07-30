"""Minimal scoring API: look up an account's features, apply the rules layer, return
the verdict. In a real system this would sit behind the event pipeline and feed
enforcement; here it demonstrates the serving-path shape.

    uvicorn abuse_signals.serve:app
    curl -s localhost:8000/score -X POST -H 'content-type: application/json' \
         -d '{"account_id": 42}'
"""

from __future__ import annotations

import os
import sqlite3

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .rules import DEFAULT_RULES_PATH, load_rules

DB_PATH = os.environ.get("ABUSE_SIGNALS_DB", "data/abuse.db")

app = FastAPI(title="abuse-signals", version="0.1.0")
_ruleset = load_rules(DEFAULT_RULES_PATH)


class ScoreRequest(BaseModel):
    account_id: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "rules": len(_ruleset.rules)}


@app.post("/score")
def score(request: ScoreRequest) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM account_features WHERE account_id = ?",
            (request.account_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="unknown account_id")
    verdict = _ruleset.evaluate(dict(row))
    verdict["action"] = verdict["action"] or "none"
    return verdict
