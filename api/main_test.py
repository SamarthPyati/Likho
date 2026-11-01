from fastapi.testclient import TestClient

from api.main import app


def client() -> TestClient:
    return TestClient(app)


def test_health_after_startup_warms_models():
    with client() as c:  # entering runs the lifespan startup hook
        body = c.get("/health").json()
    assert body == {
        "status": "ok",
        "mock_models": True,
        "models": {"xlit": True, "llm": True, "translate": True},
    }


def test_assist_round_trip():
    with client() as c:
        r = c.post(
            "/assist",
            json={"text": "kal ke meeting ke liye Rohit ko bata dena", "translate": True},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["native_script"] == "कल के meeting के लिए Rohit को बता देना"
    assert body["english"].startswith("[EN] ")
    assert body["route"]["stages"][-1] == "translate"
    assert body["errors"] == []


def test_assist_rejects_unknown_register():
    with client() as c:
        r = c.post("/assist", json={"text": "kal", "register": "shouty"})
    assert r.status_code == 422


def test_route_endpoint():
    with client() as c:
        r = c.post("/route", json={"text": "मुझे कल जाना है"})
    assert r.status_code == 200
    assert "transliterate" not in r.json()["stages"]
