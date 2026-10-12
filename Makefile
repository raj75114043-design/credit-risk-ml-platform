.PHONY: install data lint test eval api console up down

install:   ## Install dev dependencies
	pip install -r requirements-dev.txt
data:      ## Regenerate synthetic data (manifest must not change)
	python -m creditml.data.generator
lint:
	ruff check .
test:
	pytest -q
eval:      ## Full evaluation with quality gate (~30 s)
	python -m creditml.eval.run --gate
api:
	uvicorn creditml.api:app --reload
console:
	streamlit run streamlit_app.py
up:
	cp -n .env.example .env || true; docker compose up --build -d
down:
	docker compose down
