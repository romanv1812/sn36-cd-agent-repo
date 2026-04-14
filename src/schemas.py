from typing import Any

from pydantic import BaseModel, Field


class Action(BaseModel):
    type: str
    target: str | None = None
    value: Any | None = None
    reason: str | None = None


class ActRequest(BaseModel):
    task: str = Field(..., min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)


class IwaActRequest(BaseModel):
    task_id: str | None = None
    prompt: str = Field(..., min_length=1)
    url: str | None = None
    snapshot_html: str = ""
    screenshot: str | None = None
    step_index: int = 0
    web_project_id: str | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    # Optional IWA-style constraint rows: [{"field":"name","operator":"equals","value":"X"}, ...]
    constraints: Any | None = None


class ActResponse(BaseModel):
    status: str
    actions: list[Action] = Field(default_factory=list)
    reason: str | None = None
    intent: str | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)


class IwaActResponse(BaseModel):
    actions: list[dict[str, Any]] = Field(default_factory=list)
