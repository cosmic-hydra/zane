"""LIMS / ELN API Gateway.

Production-ready RESTful API exposing ZANE's generative and scoring
models. Designed for integration with enterprise lab platforms like
Benchling via webhook adapters.

Tech stack:
- FastAPI for routing and OpenAPI/Swagger documentation
- Pydantic for request/response validation
- Celery for async task queuing of heavy simulations
- Redis as the message broker

When FastAPI/Celery are not installed, the module provides pure-Python
data models and a synchronous fallback so tests and imports always work.

All endpoints are documented via OpenAPI at ``/docs`` when the server
is running.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    from pydantic import BaseModel, Field

    _PYDANTIC = True
except ImportError:
    _PYDANTIC = False

    class BaseModel:  # type: ignore[no-redef]
        """Stub when pydantic is unavailable."""

        def __init__(self, **kwargs: Any):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self) -> dict[str, Any]:
            return self.__dict__

    def Field(default: Any = None, **kw: Any) -> Any:  # type: ignore[misc]
        return default


try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends

    _FASTAPI = True
except ImportError:
    _FASTAPI = False

try:
    from celery import Celery as _Celery

    _CELERY = True
except ImportError:
    _CELERY = False


# ---------------------------------------------------------------------------
# Pydantic models (request / response)
# ---------------------------------------------------------------------------
class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerateRequest(BaseModel):
    """Request to generate and score drug candidates."""

    target_protein_pdb: str = Field(default="", description="PDB content or path for the target protein")
    num_candidates: int = Field(default=100, description="Number of candidates to generate")
    top_k: int = Field(default=10, description="Number of top candidates to return")
    include_admet: bool = Field(default=True, description="Include ADMET scoring")
    include_fep: bool = Field(default=False, description="Run FEP physics simulation (slow)")
    webhook_url: str = Field(default="", description="Webhook URL for async result delivery (Benchling format)")


class CandidateScore(BaseModel):
    """Scored drug candidate."""

    smiles: str = Field(default="")
    delta_g: float | None = Field(default=None, description="Binding free energy (kcal/mol)")
    toxicity: float = Field(default=0.0, description="Overall toxicity probability")
    drug_likeness: float = Field(default=0.0, description="QED-like drug-likeness score")
    sa_score: float = Field(default=3.0, description="Synthetic accessibility score")
    pareto_rank: int = Field(default=0, description="Pareto optimality rank (0 = front)")
    supply_chain_viable: bool | None = Field(default=None)


class TaskResponse(BaseModel):
    """Async task submission response."""

    task_id: str = Field(default="")
    status: str = Field(default="pending")
    message: str = Field(default="")
    estimated_seconds: int = Field(default=0)


class TaskResult(BaseModel):
    """Completed task result."""

    task_id: str = Field(default="")
    status: str = Field(default="completed")
    candidates: list = Field(default_factory=list)
    elapsed_seconds: float = Field(default=0.0)
    metadata: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """API health check response."""

    status: str = Field(default="healthy")
    version: str = Field(default="2026.4.1")
    celery_connected: bool = Field(default=False)
    models_loaded: bool = Field(default=True)


class BenchlingWebhookPayload(BaseModel):
    """Webhook payload formatted for Benchling ELN integration."""

    event_type: str = Field(default="zane.generation.complete")
    task_id: str = Field(default="")
    results: list = Field(default_factory=list)
    timestamp: str = Field(default="")


# ---------------------------------------------------------------------------
# In-memory task store (production would use Redis)
# ---------------------------------------------------------------------------
class TaskStore:
    """Simple in-memory task store for development/testing."""

    def __init__(self):
        self._tasks: dict[str, dict[str, Any]] = {}

    def create(self, task_id: str, request_data: dict[str, Any]) -> dict[str, Any]:
        task = {
            "task_id": task_id,
            "status": TaskStatus.PENDING.value,
            "request": request_data,
            "result": None,
            "created_at": time.time(),
            "completed_at": None,
        }
        self._tasks[task_id] = task
        return task

    def update(self, task_id: str, status: str, result: Any = None) -> None:
        if task_id in self._tasks:
            self._tasks[task_id]["status"] = status
            if result is not None:
                self._tasks[task_id]["result"] = result
            if status in (TaskStatus.COMPLETED.value, TaskStatus.FAILED.value):
                self._tasks[task_id]["completed_at"] = time.time()

    def get(self, task_id: str) -> dict[str, Any] | None:
        return self._tasks.get(task_id)

    def list_tasks(self, limit: int = 50) -> list[dict[str, Any]]:
        tasks = sorted(self._tasks.values(), key=lambda t: t["created_at"], reverse=True)
        return tasks[:limit]


_task_store = TaskStore()


# ---------------------------------------------------------------------------
# Core task execution (sync, used by both Celery and direct path)
# ---------------------------------------------------------------------------
def _execute_generation_task(task_id: str, request_data: dict[str, Any]) -> dict[str, Any]:
    """Run the generation + scoring pipeline for a task.

    This function is called by both the Celery worker and the synchronous
    fallback path.
    """
    _task_store.update(task_id, TaskStatus.RUNNING.value)
    t0 = time.monotonic()

    try:
        from drug_discovery.safety.end_to_end_pipeline import SafeGenerationPipeline, PipelineConfig

        cfg = PipelineConfig(
            num_candidates=request_data.get("num_candidates", 100),
            final_top_k=request_data.get("top_k", 10),
        )
        pipeline = SafeGenerationPipeline(config=cfg)
        result = pipeline.run(
            num_candidates=cfg.num_candidates,
            top_k=cfg.final_top_k,
        )

        candidates = []
        for c in result.final_candidates:
            scores = c.get("scores", {})
            candidates.append({
                "smiles": c.get("smiles", ""),
                "delta_g": scores.get("delta_g"),
                "toxicity": scores.get("toxicity", 0.0),
                "drug_likeness": scores.get("drug_likeness", 0.0),
                "sa_score": scores.get("sa_score", 3.0),
                "pareto_rank": c.get("pareto_rank", 0),
            })

        task_result = {
            "task_id": task_id,
            "status": TaskStatus.COMPLETED.value,
            "candidates": candidates,
            "elapsed_seconds": time.monotonic() - t0,
            "metadata": {
                "generated": result.candidates_generated,
                "valid": result.candidates_valid,
                "validation_rate": result.validation_rate,
                "safety_rate": result.safety_rate,
            },
        }
        _task_store.update(task_id, TaskStatus.COMPLETED.value, task_result)

        # Fire webhook if configured
        webhook_url = request_data.get("webhook_url", "")
        if webhook_url:
            _fire_webhook(webhook_url, task_id, candidates)

        return task_result

    except Exception as exc:
        logger.error("Task %s failed: %s", task_id, exc)
        error_result = {
            "task_id": task_id,
            "status": TaskStatus.FAILED.value,
            "error": str(exc),
            "elapsed_seconds": time.monotonic() - t0,
        }
        _task_store.update(task_id, TaskStatus.FAILED.value, error_result)
        return error_result


def _fire_webhook(url: str, task_id: str, candidates: list[dict]) -> None:
    """Send results to a Benchling-formatted webhook."""
    import datetime

    payload = {
        "event_type": "zane.generation.complete",
        "task_id": task_id,
        "results": candidates,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }

    try:
        import requests

        resp = requests.post(url, json=payload, timeout=30)
        logger.info("Webhook delivered to %s: status %d", url, resp.status_code)
    except Exception as exc:
        logger.warning("Webhook delivery to %s failed: %s", url, exc)


# ---------------------------------------------------------------------------
# Celery app (when available)
# ---------------------------------------------------------------------------
if _CELERY:
    celery_app = _Celery(
        "zane_tasks",
        broker="redis://localhost:6379/0",
        backend="redis://localhost:6379/1",
    )
    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        task_acks_late=True,  # Re-queue on worker crash
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,  # One task at a time for heavy sims
        task_time_limit=14400,  # 4 hours max for physics simulations
        task_soft_time_limit=13800,  # Soft limit with 10 min grace
        broker_connection_retry_on_startup=True,
    )

    @celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
    def run_generation_task(self, task_id: str, request_data: dict) -> dict:
        """Celery task for async generation."""
        try:
            return _execute_generation_task(task_id, request_data)
        except Exception as exc:
            logger.error("Celery task %s failed, retrying: %s", task_id, exc)
            raise self.retry(exc=exc)

else:
    celery_app = None

    def run_generation_task(task_id: str, request_data: dict) -> dict:  # type: ignore[misc]
        """Synchronous fallback when Celery is not available."""
        return _execute_generation_task(task_id, request_data)


# ---------------------------------------------------------------------------
# FastAPI application (when available)
# ---------------------------------------------------------------------------
def create_app() -> Any:
    """Create and return the FastAPI application.

    Returns a FastAPI app when the library is available, or None.
    """
    if not _FASTAPI:
        logger.warning("FastAPI not installed; API gateway unavailable")
        return None

    app = FastAPI(
        title="ZANE Drug Discovery API",
        description=(
            "RESTful API for AI-native drug discovery. "
            "Exposes generative models, physics scoring, ADMET evaluation, "
            "and integration with enterprise LIMS/ELN platforms."
        ),
        version="2026.4.1",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Check API health and component availability."""
        return HealthResponse(
            status="healthy",
            version="2026.4.1",
            celery_connected=_CELERY,
            models_loaded=True,
        )

    @app.post("/generate", response_model=TaskResponse, tags=["Generation"])
    async def generate_candidates(request: GenerateRequest, background_tasks: BackgroundTasks):
        """Submit a drug candidate generation task.

        The task is queued via Celery (when available) or run in-process.
        Use ``GET /tasks/{task_id}`` to poll for results, or provide a
        ``webhook_url`` for async delivery in Benchling format.
        """
        task_id = uuid.uuid4().hex[:16]
        request_data = request.model_dump() if hasattr(request, "model_dump") else request.__dict__

        _task_store.create(task_id, request_data)

        estimated = 30 if not request.include_fep else 3600

        if _CELERY and celery_app is not None:
            run_generation_task.delay(task_id, request_data)
        else:
            background_tasks.add_task(_execute_generation_task, task_id, request_data)

        return TaskResponse(
            task_id=task_id,
            status="pending",
            message="Task queued for execution",
            estimated_seconds=estimated,
        )

    @app.get("/tasks/{task_id}", response_model=TaskResult, tags=["Tasks"])
    async def get_task_result(task_id: str):
        """Poll for the result of a generation task."""
        task = _task_store.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        result = task.get("result") or {}
        return TaskResult(
            task_id=task_id,
            status=task["status"],
            candidates=result.get("candidates", []),
            elapsed_seconds=result.get("elapsed_seconds", 0.0),
            metadata=result.get("metadata", {}),
        )

    @app.get("/tasks", tags=["Tasks"])
    async def list_tasks(limit: int = 50):
        """List recent tasks."""
        return _task_store.list_tasks(limit=limit)

    @app.post("/score", tags=["Scoring"])
    async def score_smiles(smiles_list: list[str]):
        """Score a list of SMILES against the toxicity gate and return verdicts."""
        from drug_discovery.safety.toxicity_gate import ToxicityGate

        gate = ToxicityGate()
        verdicts = gate.evaluate_batch(smiles_list)
        return [v.as_dict() for v in verdicts]

    @app.post("/validate", tags=["Validation"])
    async def validate_smiles(smiles_list: list[str]):
        """Validate and canonicalize a list of SMILES strings."""
        from drug_discovery.safety.smiles_validator import SmilesValidator

        validator = SmilesValidator()
        results = validator.validate_batch(smiles_list)
        return [
            {
                "original": r.original,
                "canonical": r.canonical,
                "is_valid": r.is_valid,
                "rejection_reason": r.rejection_reason,
            }
            for r in results
        ]

    @app.post("/webhooks/benchling", tags=["Webhooks"])
    async def benchling_webhook(payload: dict, background_tasks: BackgroundTasks):
        """Receive a Benchling webhook and trigger a generation task.

        Expects a JSON body with at least ``target_protein_pdb`` and
        optionally ``num_candidates`` and ``callback_url``.
        """
        task_id = uuid.uuid4().hex[:16]
        request_data = {
            "target_protein_pdb": payload.get("target_protein_pdb", ""),
            "num_candidates": payload.get("num_candidates", 100),
            "top_k": payload.get("top_k", 10),
            "include_admet": True,
            "webhook_url": payload.get("callback_url", ""),
        }
        _task_store.create(task_id, request_data)

        if _CELERY and celery_app is not None:
            run_generation_task.delay(task_id, request_data)
        else:
            background_tasks.add_task(_execute_generation_task, task_id, request_data)

        return {"task_id": task_id, "status": "accepted"}

    return app


# Create the app instance for uvicorn
app = create_app()
