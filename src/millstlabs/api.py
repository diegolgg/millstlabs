"""Optional loopback HTTP interface to the simulator (not a paid language-model API)."""
import copy
from dataclasses import asdict
from threading import RLock
from typing import Literal
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import EnvironmentConfig, TrainingConfig
from .env import ACTION_NAMES, PopulationEnv
from .knowledge import TOOL_NAMES, execute_tools, make_corpus, next_observations
from .notes import publish_notes
from .tactics import record_experience

app = FastAPI(title="Mill Street Labs Sandbox", version="0.1.0")
sessions = {}
corpora = {}
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
    corpus_mode: Literal["off", "private", "shared"] = "off"
    corpus_interface: Literal["tools", "notes"] = "tools"
    note_style: Literal["grounded", "prose"] = "grounded"
    note_memory: Literal["map_facts", "tactics"] = "map_facts"


class StepRequest(BaseModel):
    actions: dict[str, int]
    expected_tick: int = Field(ge=0)
    knowledge_actions: dict[str, Literal["none", "put_terrain", "put_food", "get_terrain", "get_food"]] | None = None
    notes: dict[str, str] | None = None


@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(sessions)}


@app.get("/schema")
def schema():
    return {"actions": dict(enumerate(ACTION_NAMES)), "defaults": asdict(EnvironmentConfig()),
            "knowledge_tools": dict(enumerate(TOOL_NAMES)), "joint_action_encoding": "physical_action + 7 * knowledge_tool",
            "notes_interface": "Create with corpus_interface=notes and corpus_mode=shared. Step with physical actions 0..6 and optional notes {agent: text}. Observations contain automatically retrieved notes; evidence is attached by the server.",
            "local_codes": {"-1": "unobserved", "0": "empty", "1": "wall", "2": "food", "3": "prey", "4": "predator", "5": "alert prey"}}


@app.post("/environments", status_code=201)
def create(request: CreateRequest):
    with lock:
        if len(sessions) >= MAX_SESSIONS:
            raise HTTPException(429, "Session limit reached; delete unused environments")
        env = PopulationEnv(reproduction=request.reproduction, social_preference=request.social_preference)
        observations, _ = env.reset(seed=request.seed)
        if request.corpus_interface == "notes" and request.corpus_mode == "off":
            raise HTTPException(422, "Notes require corpus_mode private or shared")
        if request.note_memory == "tactics" and request.corpus_interface != "notes":
            raise HTTPException(422, "Tactics require corpus_interface notes")
        training = TrainingConfig(corpus_mode=request.corpus_mode, controller_architecture="split",
                                  corpus_interface=request.corpus_interface, note_style=request.note_style,
                                  note_memory=request.note_memory, corpus_tool_cost=0, note_write_penalty=0)
        corpus, observations = make_corpus(training, observations)
        i = str(uuid4())
        sessions[i] = env
        corpora[i] = corpus, training
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
            corpus, training = corpora.get(i, (None, None))
            actions = dict(request.actions)
            if set(actions) != set(env.agents):
                raise ValueError("Supply exactly one action for every living agent")
            if request.knowledge_actions is not None:
                if training and training.corpus_interface == "notes":
                    raise ValueError("Notes sessions accept optional text, not knowledge tool actions")
                if corpus is None:
                    raise ValueError("Enable corpus_mode when creating this environment")
                if not set(request.knowledge_actions) <= set(env.agents) or any(not 0 <= a < 7 for a in actions.values()):
                    raise ValueError("Named knowledge tools require living agents and physical actions 0..6")
                actions = {a: v + 7*TOOL_NAMES.index(request.knowledge_actions.get(a, "none")) for a, v in actions.items()}
            notes_mode = training and training.corpus_interface == "notes"
            if any(not 0 <= a < (35 if corpus and not notes_mode else 7) for a in actions.values()):
                raise ValueError("Action outside the session's action space")
            current = {a: env.observe(a) for a in env.agents}
            publication_rewards = {}
            if request.notes is not None:
                if not notes_mode:
                    raise ValueError("Enable corpus_interface=notes before depositing text")
                if not set(request.notes) <= set(env.agents) or any(not text.strip() or len(text) > 480 for text in request.notes.values()):
                    raise ValueError("Notes require living authors and 1..480 characters")
                current = copy.deepcopy(corpus).augment(current)
                if any(not current[a]["note_opportunity"] for a in request.notes):
                    raise ValueError("No fresh evidence or publication cooldown active")
                # Validate the entire batch before any mutation; prose is untrusted.
                publication_rewards = publish_notes(corpus, {a: int(a in request.notes) for a in env.agents},
                    current, None, env, training, texts=request.notes if training.note_style == "prose" else None)
            elif notes_mode:
                corpus.pending.clear()
            physical = execute_tools(corpus, actions, current, env, training)
            obs, rewards, terminated, truncated, infos = env.step(physical)
            record_experience(corpus, physical, current, obs, env)
            obs = next_observations(corpus, obs, env.agents)
            if corpus:
                corpus.events.clear()
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return serializable({"tick": env.tick, "agents": env.agents, "observations": obs,
                             "rewards": rewards, "publication_rewards": publication_rewards,
                             "terminated": terminated, "truncated": truncated, "infos": infos})


@app.get("/environments/{i}")
def inspect(i: str):
    with lock:
        env = get_env(i)
        obs = {a: env.observe(a) for a in env.agents}
        corpus, _ = corpora.get(i, (None, None))
        return serializable({"tick": env.tick, "agents": env.agents,
                             "observations": copy.deepcopy(corpus).augment(obs) if corpus else obs})


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
        corpora.pop(i, None)
