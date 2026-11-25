PY ?= python3

.PHONY: test lint frontend docker

test:
	cd backend && $(PY) -m pytest

lint:
	cd backend && $(PY) -m ruff check app tests
	cd backend && $(PY) -m ruff format --check app tests

frontend:
	cd frontend && npm test && npm run build

docker:
	docker compose build
