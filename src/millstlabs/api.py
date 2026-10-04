"""Optional loopback HTTP interface to the simulator (not a paid language-model API)."""
from dataclasses import asdict
from threading import RLock
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import EnvironmentConfig
from .env import ACTION_NAMES, PopulationEnv

app = FastAPI(title="Mill Street Labs Sandbox", version="0.1.0")
sessions = {}
lock = RLock()
MAX_SESSIONS = 16


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


class CreateRequest(BaseModel):
    seed: int = 0
    reproduction: bool = True
    social_preference: float = Field(default=0, ge=-2, le=2)


class StepRequest(BaseModel):
    actions: dict[str, int]
    expected_tick: int = Field(ge=0)


@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(sessions)}


@app.get("/schema")
def schema():
    return {"actions": dict(enumerate(ACTION_NAMES)), "defaults": asdict(EnvironmentConfig()),
            "local_codes": {"-1": "unobserved", "0": "empty", "1": "wall", "2": "food", "3": "prey", "4": "predator", "5": "alert prey"}}


@app.post("/environments", status_code=201)
def create(request: CreateRequest):
    with lock:
        if len(sessions) >= MAX_SESSIONS:
            raise HTTPException(429, "Session limit reached; delete unused environments")
        env = PopulationEnv(reproduction=request.reproduction, social_preference=request.social_preference)
        observations, _ = env.reset(seed=request.seed)
        i = str(uuid4())
        sessions[i] = env
        return {"id": i, "tick": 0, "agents": env.agents, "observations": serializable(observations)}


def get_env(i):
    if i not in sessions:
        raise HTTPException(404, "Unknown environment")
    return sessions[i]


@app.post("/environments/{i}/step")
def step(i: str, request: StepRequest):
    with lock:
        env = get_env(i)
        if env.tick != request.expected_tick:
            raise HTTPException(409, "Tick changed; do not replay a completed action request")
        try:
            obs, rewards, terminated, truncated, infos = env.step(request.actions)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return serializable({"tick": env.tick, "agents": env.agents, "observations": obs,
                             "rewards": rewards, "terminated": terminated, "truncated": truncated, "infos": infos})


@app.get("/environments/{i}")
def inspect(i: str):
    with lock:
        env = get_env(i)
        return serializable({"tick": env.tick, "agents": env.agents,
                             "observations": {a: env.observe(a) for a in env.agents}})


@app.get("/environments/{i}/render")
def render(i: str):
    """Privileged operator diagnostics. Never include this in policy input."""
    with lock:
        env = get_env(i)
        return {"ansi": env.render(), "metrics": env.metrics()}


@app.delete("/environments/{i}", status_code=204)
def delete(i: str):
    with lock:
        get_env(i).close()
        del sessions[i]
