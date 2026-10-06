# Makefile
# ============================================================================
# Globals
# ============================================================================
CONTAINER_NAME:=threat-detekt-model-training

# ============================================================================
# Development Commands
# ============================================================================
.PHONY: help
help: ## List available commands
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

.PHONY: deps
deps: ## Create .venv, activate it, and install requirements.txt
	python3 -m venv .venv
	bash -c 'source .venv/bin/activate && pip install -r requirements.txt'

.PHONY: build
build: ## Build the training container
	docker build \
			--target=release \
			-t $(CONTAINER_NAME):latest .


.PHONY: run
run: ## Run the training container
	docker compose stop
	docker compose up

.PHONY: down
down: ## Stop the training container
	docker compose stop
	docker compose down --volumes

.PHONY: lint-check
lint-check: ## Run linting check
	ruff check

.PHONY: lint
lint: ## Fix source linting issues
	ruff check --fix

.PHONY: format
format: ## Format source code
	ruff check --select I --fix && ruff format