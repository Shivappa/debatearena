"""
DebateArenaEnv — FastAPI HTTP server.

Client <-> Server contract:
  POST /reset      body: {"topic_id": "easy"}          -> {"observation": {...}}
  POST /step       body: {"tool": "...", "params": {}}  -> {"observation": {...}, "reward": 0.0, "done": false}
  GET  /state                                           -> {"topic": {...}, "state": {...}}
  GET  /health                                          -> {"status": "ok"}
  GET  /manifest                                        -> MCP tool manifest
  GET  /curriculum                                      -> adaptive curriculum summary
  GET  /docs                                            -> Swagger UI
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from server.tools import TOOL_REGISTRY, MCP_MANIFEST, tool_new_episode, tool_curriculum
from server.env import get_env

app = FastAPI(
    title="DebateArenaEnv",
    description="Multi-agent epistemic debate RL environment.",
    version="1.0.0",
)


class ResetRequest(BaseModel):
    topic_id: Optional[str] = None


class StepRequest(BaseModel):
    tool:   str
    params: Dict[str, Any] = {}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "DebateArenaEnv"}


@app.get("/manifest")
def manifest() -> Dict[str, Any]:
    return MCP_MANIFEST


@app.post("/reset")
def reset(req: ResetRequest) -> Dict[str, Any]:
    return tool_new_episode(req.topic_id)


@app.get("/state")
def state() -> Dict[str, Any]:
    return get_env().state()


@app.post("/step")
def step(req: StepRequest) -> Dict[str, Any]:
    """
    Execute one agent action.

    Available tools:
    | Tool              | Params                                              |
    |-------------------|-----------------------------------------------------|
    | verify_fact       | fact_key (str), role (str)                          |
    | submit_argument   | argument (str), facts_cited (list)                  |
    | submit_rebuttal   | rebuttal (str), facts_cited (list), expose_fallacy  |
    | refine_position   | refined_claim (str), reason (str)                   |
    | concede_point     | sub_point (str), maintain_main (str)                |
    | end_debate        | closing_statement (str), role (str)                 |
    """
    fn = TOOL_REGISTRY.get(req.tool)
    if fn is None:
        raise HTTPException(status_code=404, detail=f"Unknown tool: '{req.tool}'. Check GET /manifest.")
    try:
        result = fn(**req.params)
    except TypeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return result


@app.get("/curriculum")
def curriculum() -> Dict[str, Any]:
    return tool_curriculum()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)
