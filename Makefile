PYTHON_VERSION ?= 3.10
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: setup setup-models dev api test eval lint format

# Uses uv when available (it can fetch Python 3.10 itself); otherwise needs python3.10 on PATH.
setup:
	@if command -v uv >/dev/null 2>&1; then \
		uv venv --python $(PYTHON_VERSION) $(VENV) && \
		uv pip install --python $(BIN)/python -r requirements.txt; \
	else \
		python$(PYTHON_VERSION) -m venv $(VENV) && \
		$(BIN)/pip install -r requirements.txt; \
	fi

# Adds the real model stack (IndicXlit + fairseq + torch). Needs a C compiler for fairseq.
setup-models: setup
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install --python $(BIN)/python -r requirements-models.txt; \
	else \
		$(BIN)/pip install -r requirements-models.txt; \
	fi

dev:
	$(BIN)/python -m app.gradio_app

api:
	$(BIN)/uvicorn api.main:app --port 8000 --reload

test:
	MOCK_MODELS=1 $(BIN)/pytest -q

eval:
	$(BIN)/python -m eval.run_eval

lint:
	$(BIN)/ruff check .
	$(BIN)/black --check .

format:
	$(BIN)/ruff check --fix .
	$(BIN)/black .
