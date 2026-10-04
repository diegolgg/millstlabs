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
