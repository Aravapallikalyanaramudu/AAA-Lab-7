import os
import json
import asyncio
import logging
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from deep_research_agent.orchestrator import DeepResearchOrchestrator
from deep_research_agent.llm import LLMClient
from deep_research_agent.search import SearchEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deep_research_agent.web")

app = FastAPI(title="Deep Research Agent API", version="1.0.0")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(CURRENT_DIR, "static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ResearchRequestPayload(BaseModel):
    topic: str
    depth: str = "deep"
    provider: str = "auto"
    api_key: Optional[str] = None
    model: Optional[str] = None
    additional_instructions: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Deep Research Agent Web Dashboard</h2><p>Static index.html not found.</p>")


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "Deep Research Agent", "version": "1.0.0"}


@app.post("/api/research/run")
async def run_research_direct(payload: ResearchRequestPayload):
    """Executes the research workflow and returns the complete final report JSON."""
    llm = LLMClient(
        provider=payload.provider,
        api_key=payload.api_key,
        model=payload.model,
    )
    orchestrator = DeepResearchOrchestrator(llm=llm)
    try:
        report = await orchestrator.run(
            topic=payload.topic,
            depth=payload.depth,
            additional_instructions=payload.additional_instructions,
        )
        return JSONResponse(content=report.model_dump())
    except Exception as e:
        logger.exception("Research run error")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/research/stream")
async def stream_research(
    topic: str,
    depth: str = "deep",
    provider: str = "auto",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
):
    """
    Server-Sent Events (SSE) streaming endpoint providing real-time workflow events.
    """
    llm = LLMClient(provider=provider, api_key=api_key, model=model)
    orchestrator = DeepResearchOrchestrator(llm=llm)

    async def event_generator():
        try:
            async for event in orchestrator.stream_workflow(topic=topic, depth=depth):
                event_json = json.dumps(event.model_dump())
                yield f"data: {event_json}\n\n"
                await asyncio.sleep(0.05)  # smooth SSE transmission
        except Exception as e:
            err_data = json.dumps({"stage": "error", "status": "error", "message": str(e)})
            yield f"data: {err_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("deep_research_agent.web.server:app", host="0.0.0.0", port=8000, reload=True)
