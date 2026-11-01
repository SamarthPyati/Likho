"""HTTP API over `likho.pipeline`. Run with `make api` (FastAPI on :8000).

Endpoints are plain `def`, so FastAPI runs them in its threadpool; `_lock`
then serialises model access, because each model is one in-process instance
and generation on it is not thread-safe.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI
from pydantic import BaseModel

from likho.config import settings
from likho.logging import get_logger
from likho.pipeline import assist as run_pipeline
from likho.registry import warm_models
from likho.router import route as run_router
from likho.schemas import AssistRequest, AssistResponse, Route

logger = get_logger(__name__)

_lock = Lock()
_models: dict[str, bool] = {}


class RouteRequest(BaseModel):
    text: str


class Health(BaseModel):
    status: str  # "ok", or "degraded" if a model failed to load
    mock_models: bool
    models: dict[str, bool]  # model name → loaded


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Load every model before serving (NFR3), so no request pays the load cost.
    _models.update(warm_models())
    logger.info("models loaded: %s", _models)
    yield


app = FastAPI(title="Likho", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> Health:
    status = "ok" if all(_models.values()) else "degraded"
    return Health(status=status, mock_models=settings.mock_models, models=_models)


@app.post("/assist")
def assist(request: AssistRequest) -> AssistResponse:
    with _lock:
        return run_pipeline(request)


@app.post("/route")
def route(request: RouteRequest) -> Route:
    with _lock:
        return run_router(request.text)
