from fastapi.testclient import TestClient

from millstlabs.api import app


def test_http_lifecycle_and_stale_tick():
    with TestClient(app) as client:
        response = client.post("/environments", json={"seed": 1})
        assert response.status_code == 201
        state = response.json()
        i = state["id"]
        payload = {"actions": dict.fromkeys(state["agents"], 6), "expected_tick": 0}
        result = client.post(f"/environments/{i}/step", json=payload)
        assert result.status_code == 200 and result.json()["tick"] == 1
        assert client.post(f"/environments/{i}/step", json=payload).status_code == 409
        assert client.get(f"/environments/{i}").status_code == 200
        assert client.post(f"/environments/{i}/step", json={"actions": {}, "expected_tick": 1}).status_code == 422
        assert client.delete(f"/environments/{i}").status_code == 204
        assert client.get(f"/environments/{i}").status_code == 404


def test_tactic_publication_and_retrieval_are_energy_free():
    with TestClient(app) as client:
        states = [client.post("/environments", json={"seed": 11, "reproduction": False,
            "corpus_mode": "shared", "corpus_interface": "notes", "note_memory": "tactics",
            "note_style": "prose"}).json() for _ in range(2)]
        for state in states:
            result = client.post(f"/environments/{state['id']}/step", json={
                "actions": dict.fromkeys(state["agents"], 6), "expected_tick": 0})
            assert result.status_code == 200
            state.update(result.json())
        author = states[0]["agents"][0]
        results = []
        for index, state in enumerate(states):
            payload = {"actions": dict.fromkeys(state["agents"], 6), "expected_tick": 1}
            if index == 0:
                payload["notes"] = {author: "Rest used energy. Its long-term effect is uncertain."}
            result = client.post(f"/environments/{state['id']}/step", json=payload)
            assert result.status_code == 200
            results.append(result.json())
        assert results[0]["observations"][author]["notes"]
        for agent in results[0]["agents"]:
            assert results[0]["observations"][agent]["self"] == results[1]["observations"][agent]["self"]
        for state in states:
            client.delete(f"/environments/{state['id']}")
