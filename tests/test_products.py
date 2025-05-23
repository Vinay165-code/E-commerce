def test_user_registration_and_login(client):
    user = {"name": "User1", "email": "user1@test.com", "password": "pass123"}
    resp = client.post("/users", json=user)
    assert resp.status_code == 200
    # Attempt login
    resp = client.post(
        "/token", data={"username": user["email"], "password": user["password"]}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
