"""Start `millst serve`, then run this observation-only random client."""
import random
import httpx

with httpx.Client(base_url="http://127.0.0.1:8000", timeout=30) as client:
    response = client.post("/environments", json={"seed": 42, "reproduction": True})
    response.raise_for_status()
    state = response.json()
    env_id = state["id"]
    try:
        for _ in range(128):
            if not state["agents"]:
                break
            actions = {i: random.choice([a for a, valid in enumerate(state["observations"][i]["action_mask"]) if valid])
                       for i in state["agents"]}
            response = client.post(f"/environments/{env_id}/step", json={"actions": actions, "expected_tick": state["tick"]})
            response.raise_for_status()
            state = response.json()
        print({"tick": state["tick"], "population": len(state["agents"])})
    finally:
        client.delete(f"/environments/{env_id}")
