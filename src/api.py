import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .agent import run_agent_iwa, run_agent_local
from .schemas import ActRequest, IwaActRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

app = FastAPI(title="Minimal SN36 Local Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.api_route("/health", methods=["GET", "HEAD"])
def health() -> dict:
    return {"status": "ok"}


_api_log = logging.getLogger("api")


@app.post("/act")
def act(payload: dict[str, Any]) -> dict[str, Any]:
    # task_id arrives from ApifiedWebAgent (apified_iterative_agent.py:83, act_protocol.py:19)
    task_id = str(payload.get("task_id") or "")
    _api_log.info("[API] /act task_id=%r web_project_id=%r step=%s", task_id, payload.get("web_project_id"), payload.get("step_index"))

    if "task" in payload:
        local_payload = ActRequest.model_validate(payload)
        return run_agent_local(task_text=local_payload.task, context=local_payload.context).model_dump()

    if "prompt" in payload and "step_index" in payload:
        iwa_payload = IwaActRequest.model_validate(payload)
        return run_agent_iwa(iwa_payload, task_id=task_id).model_dump()

    raise HTTPException(
        status_code=422,
        detail="Unsupported /act payload format. Use local format {'task': ...} or IWA format {'prompt': ..., 'step_index': ...}.",
    )
