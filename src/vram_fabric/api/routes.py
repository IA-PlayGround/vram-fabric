from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vram_fabric.core.fabric import VRAMFabric

app = FastAPI(title="VRAM Fabric API", version="1.0.0")

_fabric: VRAMFabric | None = None


class QueryRequest(BaseModel):
    prompt: str
    k: int = 10


class SearchRequest(BaseModel):
    query: str
    k: int = 10


class IndexRequest(BaseModel):
    documents: list[str]
    model: str = "bge-large"


class AgentRegisterRequest(BaseModel):
    name: str
    system_prompt: str
    model: str | None = None


class SchedulerPolicyRequest(BaseModel):
    llm: float = 0.70
    vector: float = 0.15
    agent: float = 0.10
    cache: float = 0.05


class AgentTask(BaseModel):
    agent_name: str
    prompt: str
    task_id: str = ""
    dependencies: list[str] = []


class WorkflowRequest(BaseModel):
    tasks: list[AgentTask]


def get_fabric() -> VRAMFabric:
    global _fabric
    if _fabric is None:
        _fabric = VRAMFabric()
    return _fabric


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/query")
async def query(req: QueryRequest) -> dict:
    fabric = get_fabric()
    response = fabric.query(req.prompt)
    return {"response": response}


@app.post("/search")
async def search(req: SearchRequest) -> dict:
    fabric = get_fabric()
    results = fabric.search(req.query, k=req.k)
    return {
        "results": [{"id": r.id, "score": r.score} for r in results],
    }


@app.post("/index")
async def index_documents(req: IndexRequest) -> dict:
    fabric = get_fabric()
    count = fabric.index_documents(req.documents, model=req.model)
    return {"indexed": count}


@app.post("/agents/register")
async def register_agent(req: AgentRegisterRequest) -> dict:
    fabric = get_fabric()
    handle = fabric.register_agent(req.name, req.system_prompt, req.model)
    return {"name": handle.name, "model": handle.model}


@app.post("/agents/task")
async def run_task(req: AgentTask) -> dict:
    from vram_fabric.core.types import Task

    fabric = get_fabric()
    task = Task(agent_name=req.agent_name, prompt=req.prompt, task_id=req.task_id, dependencies=req.dependencies)
    result = fabric.agent_runtime.run_task(task)
    return {"result": result}


@app.post("/workflow")
async def run_workflow(req: WorkflowRequest) -> dict:
    from vram_fabric.core.types import Task, Workflow

    fabric = get_fabric()
    tasks = [
        Task(agent_name=t.agent_name, prompt=t.prompt, task_id=t.task_id, dependencies=t.dependencies)
        for t in req.tasks
    ]
    workflow = Workflow(tasks=tasks)
    result = fabric.run(workflow)
    return {
        "results": result.results,
        "errors": result.errors,
        "total_time_ms": result.total_time_ms,
    }


@app.get("/cache/stats")
async def cache_stats() -> dict:
    s = get_fabric().cache_stats()
    return {
        "entries": s.current_entries,
        "max_entries": s.max_entries,
        "hits": s.hits,
        "misses": s.misses,
        "hit_ratio": s.hit_ratio,
        "evictions": s.evictions,
        "vram_used_mb": s.vram_used_mb,
    }


@app.get("/scheduler/stats")
async def scheduler_stats() -> dict:
    s = get_fabric().scheduler_stats()
    return {
        "shares": s.shares,
        "occupancy": s.occupancy,
        "adjustments": s.adjustments,
        "throttled": s.throttled,
    }


@app.post("/scheduler/policy")
async def set_scheduler_policy(req: SchedulerPolicyRequest) -> dict:
    get_fabric().set_scheduler_policy({
        "llm": req.llm, "vector": req.vector, "agent": req.agent, "cache": req.cache,
    })
    return {"status": "ok"}


@app.get("/memory/stats")
async def memory_stats() -> dict:
    s = get_fabric().memory_stats()
    return {
        "vram_total_mb": s.vram_total_mb,
        "vram_used_mb": s.vram_used_mb,
        "vram_free_mb": s.vram_free_mb,
        "ram_spillover_mb": s.ram_spillover_mb,
        "ssd_spillover_mb": s.ssd_spillover_mb,
        "spillover_count": s.spillover_count,
    }


@app.get("/telemetry")
async def telemetry_snapshot() -> dict:
    return get_fabric().telemetry_snapshot()


@app.post("/shutdown")
async def shutdown() -> dict:
    global _fabric
    if _fabric is not None:
        _fabric.shutdown()
        _fabric = None
    return {"status": "shutdown"}


@app.get("/stats")
async def all_stats() -> dict:
    import dataclasses
    fabric = get_fabric()
    return {
        "vector": fabric.vector_engine.stats(),
        "cache": dataclasses.asdict(fabric.cache_stats()),
        "scheduler": dataclasses.asdict(fabric.scheduler_stats()),
        "memory": dataclasses.asdict(fabric.memory_stats()),
    }
