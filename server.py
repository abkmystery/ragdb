from __future__ import annotations
import os
from typing import Any, Dict
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from .core import RAGdb


def _get_db() -> RAGdb:
    return RAGdb(os.getenv("RAGDB_PATH", "RAGdb.ragdb"))


def create_app():
    app = FastAPI(title="RAGdb API", version="1.0.3")
    security = HTTPBearer(auto_error=False)

    def require_auth(creds: HTTPAuthorizationCredentials | None = Depends(security)):
        api_key = os.getenv("RAGDB_API_KEY")
        if api_key and (not creds or creds.credentials != api_key):
            raise HTTPException(403, "Invalid API Key")

    @app.post("/ingest")
    def ingest(payload: Dict[str, Any], _=Depends(require_auth)):
        path = payload.get("path")
        if not path: raise HTTPException(400, "Path required")
        db = _get_db()
        if os.path.isdir(path):
            db.ingest_folder(path)
        else:
            db.ingest_file(path)
        return {"status": "ingested", "path": path}

    @app.get("/search")
    def search(q: str, k: int = 5, _=Depends(require_auth)):
        db = _get_db()
        results = db.search(q, top_k=k)

        # FIX: Manually convert the SearchResult objects to a JSON-compatible list
        return [
            {
                "path": r.path,
                "score": r.score,
                "media_type": r.media_type,
                "preview": r.preview,
                "metadata": r.metadata
            }
            for r in results
        ]

    return app