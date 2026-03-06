"""
FastAPI app for ESCI multi-task learning reranker: load model at startup,
expose POST /rerank and GET /health. Model path from env MODEL_PATH
(default data/multi_task_reranker).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.constants import CHECKPOINTS_DIR, DEFAULT_RERANKER_MODEL
from src.models.multi_task_reranker import MultiTaskReranker, load_multi_task_reranker

logger = logging.getLogger(__name__)

# Global model reference; set in lifespan.
reranker_instance: MultiTaskReranker | None = None


def get_model_path() -> str:
    """Resolve model path from env; default to checkpoints/multi_task_reranker."""
    return os.environ.get(
        "MODEL_PATH", str(CHECKPOINTS_DIR / "multi_task_reranker")
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load multi-task learning reranker once at startup; release on shutdown."""
    global reranker_instance
    path = get_model_path()
    model_name = os.environ.get("MODEL_NAME", DEFAULT_RERANKER_MODEL)
    logger.info(
        "Loading reranker from path=%s (fallback model_name=%s)", path, model_name
    )
    reranker_instance = load_multi_task_reranker(model_path=path, model_name=model_name)
    yield
    reranker_instance = None


app = FastAPI(
    title="ESCI Reranker API",
    description=(
        "Rerank product candidates for a query; returns score, ESCI class, "
        "and substitute flag per product."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------


class CandidateItem(BaseModel):
    """One candidate product for reranking."""

    product_id: str = Field(..., description="Unique product identifier.")
    text: str = Field(
        ...,
        description="Product text (e.g. title + description) to score against the query.",
    )


class RerankRequest(BaseModel):
    """Request body for POST /rerank."""

    query: str = Field(..., description="User search query.")
    candidates: list[CandidateItem] = Field(
        ..., description="List of product_id and text to rerank."
    )


class RankedItem(BaseModel):
    """One item in the reranked list: product_id, score, ESCI class,
    substitute probability.
    """

    product_id: str = Field(..., description="Product identifier.")
    score: float = Field(..., description="Relevance score (higher = more relevant).")
    esci_class: str = Field(..., description="Predicted ESCI class: E, S, C, or I.")
    is_substitute: float = Field(
        ...,
        description=(
            "Probability that the product is a substitute (multi-task learning "
            "Task 3: substitute identification)."
        ),
    )


class RerankResponse(BaseModel):
    """Response for POST /rerank: list of ranked products with scores
    and ESCI outputs.
    """

    ranked: list[RankedItem] = Field(
        ..., description="Candidates sorted by score descending."
    )


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = Field(default="ok", description="Service status.")
    model_loaded: bool = Field(..., description="Whether the reranker model is loaded.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Health check for load balancers and Docker.
    Returns 200 with status and model_loaded flag.
    """
    return HealthResponse(status="ok", model_loaded=reranker_instance is not None)


@app.post("/rerank", response_model=RerankResponse)
def rerank(body: RerankRequest) -> RerankResponse:
    """
    Rerank candidates for a single query.
    Request: query string and list of { product_id, text }.
    Response: same candidates sorted by relevance score with score,
    esci_class, and is_substitute per item.
    """
    if reranker_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not body.candidates:
        return RerankResponse(ranked=[])

    candidates_tuples = [(c.product_id, c.text) for c in body.candidates]
    ranked_tuples = reranker_instance.rerank(
        body.query,
        candidates_tuples,
        batch_size=32,
    )
    ranked = _to_ranked_items(ranked_tuples)
    return RerankResponse(ranked=ranked)


def _to_ranked_items(
    ranked_tuples: list[tuple[str, float, str, float]],
) -> list[RankedItem]:
    """Convert raw reranker outputs into RankedItem models."""
    return [
        RankedItem(
            product_id=pid,
            score=score,
            esci_class=esci_class,
            is_substitute=is_sub,
        )
        for pid, score, esci_class, is_sub in ranked_tuples
    ]
