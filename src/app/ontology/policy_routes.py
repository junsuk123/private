"""Read-only cached policy diagnostics and explicit off-tick RDF audit export."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Literal

from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import Response

from app.ontology.policy_evidence import SCHEMA_PATH


@lru_cache(maxsize=1)
def _schema_text() -> str:
    return SCHEMA_PATH.read_text(encoding="utf-8")


def create_policy_router(runtime_provider: Callable[[], Any]) -> APIRouter:
    router = APIRouter(prefix="/api/ontology/policy", tags=["ontology-policy"])

    @router.get("")
    def policy_snapshot() -> dict[str, Any]:
        runtime = runtime_provider()
        if runtime is None:
            return {"available": False, "policy_count": 0, "policies": {}}
        return {"available": True, **runtime.snapshot()}

    @router.get("/schema")
    def policy_schema() -> Response:
        return Response(_schema_text(), media_type="text/turtle",
                        headers={"Content-Disposition": 'attachment; filename="obaits-policy-ontology.ttl"'})

    @router.get("/{market}/{symbol}")
    def policy_graph(market: Literal["KR", "US"], symbol: str = Path(pattern=r"^[A-Za-z0-9._-]{1,32}$")) -> Response:
        # A sync endpoint runs in FastAPI's worker pool, outside both the event
        # receiver and the decision loop. No resolve() or broker read is called.
        runtime = runtime_provider()
        if runtime is None:
            raise HTTPException(status_code=503, detail="ONTOLOGY_POLICY_RUNTIME_UNAVAILABLE")
        try:
            graph = runtime.evidence_graph(symbol, market)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="ONTOLOGY_POLICY_NOT_OBSERVED") from exc
        return Response(graph.serialize(format="turtle"), media_type="text/turtle",
                        headers={"Content-Disposition": f'attachment; filename="policy-{market}-{symbol}.ttl"',
                                 "Cache-Control": "no-store"})

    return router
