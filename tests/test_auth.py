def test_list_products_empty(client):
    response = client.get("/products")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_create_and_get_product(client):
    # Create admin user
    admin = {
        "name": "Admin",
        "email": "admin@test.com",
        "password": "secret",
        "role": "admin",
    }
    client.post("/users", json=admin)
    # Authenticate
    resp = client.post(
        "/token", data={"username": admin["email"], "password": admin["password"]}
    )
    token = resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Create a product
    prod_data = {
        "name": "Test",
        "description": "Desc",
        "price": 9.99,
        "inventory": 100,
        "category_ids": [],
    }
    resp = client.post("/products", json=prod_data, headers=headers)
    assert resp.status_code == 200
    prod = resp.json()
    assert prod["name"] == "Test"

    # Fetch the product
    resp = client.get(f"/products/{prod['id']}")
    assert resp.status_code == 200
    fetched = resp.json()
    assert fetched["id"] == prod["id"]
    assert fetched["name"] == "Test"

    print(resp.json())
