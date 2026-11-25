import pytest
from app.api import create_app
from app.config import Settings
from fastapi.testclient import TestClient


@pytest.fixture
def settings():
    return Settings(_env_file=None)


@pytest.fixture
def client(settings):
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
