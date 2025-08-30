# Repository maintenance Makefile
SHELL := /bin/bash
.PHONY: help precommit-bootstrap precommit-run secrets-scan repo-clean
.DEFAULT_GOAL := help

# Colors for output
NO_COLOR=\033[0m
OK_COLOR=\033[32;01m
ERROR_COLOR=\033[31;01m
WARN_COLOR=\033[33;01m

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(OK_COLOR)%-20s$(NO_COLOR) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

precommit-bootstrap: ## Install and update pre-commit hooks
	@echo "$(OK_COLOR)→ Installing pre-commit...$(NO_COLOR)"
	@pip install --upgrade pre-commit || { echo "$(ERROR_COLOR)Failed to install pre-commit$(NO_COLOR)"; exit 1; }
	@echo "$(OK_COLOR)→ Updating pre-commit hooks...$(NO_COLOR)"
	@pre-commit autoupdate
	@echo "$(OK_COLOR)→ Installing git hooks...$(NO_COLOR)"
	@pre-commit install --install-hooks
	@echo "$(OK_COLOR)✓ Pre-commit bootstrap complete$(NO_COLOR)"

precommit-run: ## Run all pre-commit hooks on all files
	@echo "$(OK_COLOR)→ Running pre-commit hooks...$(NO_COLOR)"
	@pre-commit run --all-files --show-diff-on-failure || { \
		echo "$(WARN_COLOR)Some hooks failed. Review the output above.$(NO_COLOR)"; \
		exit 1; \
	}
	@echo "$(OK_COLOR)✓ All pre-commit checks passed$(NO_COLOR)"

secrets-scan: ## Scan for secrets and update baseline
	@echo "$(OK_COLOR)→ Scanning for secrets...$(NO_COLOR)"
	@if [ ! -f .secrets.baseline ]; then \
		echo "$(WARN_COLOR)Creating initial secrets baseline...$(NO_COLOR)"; \
		detect-secrets scan --baseline .secrets.baseline; \
	else \
		echo "$(OK_COLOR)Updating secrets baseline...$(NO_COLOR)"; \
		detect-secrets scan --baseline .secrets.baseline --update .secrets.baseline; \
	fi
	@echo "$(OK_COLOR)→ Auditing detected secrets...$(NO_COLOR)"
	@detect-secrets audit .secrets.baseline || true
	@echo "$(OK_COLOR)✓ Secrets scan complete$(NO_COLOR)"

repo-clean: ## Clean repository of backup files and maintain curated snapshot
	@echo "$(OK_COLOR)→ Running repository cleanup...$(NO_COLOR)"
	@bash scripts/cleanup_compose_backups.sh
	@echo "$(OK_COLOR)→ Removing Python cache files...$(NO_COLOR)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(OK_COLOR)→ Removing editor backup files...$(NO_COLOR)"
	@find . -type f \( -name "*~" -o -name "*.swp" -o -name "*.swo" \) -delete 2>/dev/null || true
	@echo "$(OK_COLOR)✓ Repository cleanup complete$(NO_COLOR)"

# Docker compose specific targets (if needed)
compose-validate: ## Validate docker-compose.yml syntax
	@echo "$(OK_COLOR)→ Validating docker-compose.yml...$(NO_COLOR)"
	@docker compose config --quiet || { echo "$(ERROR_COLOR)docker-compose.yml validation failed$(NO_COLOR)"; exit 1; }
	@echo "$(OK_COLOR)✓ docker-compose.yml is valid$(NO_COLOR)"

# Development workflow helpers
dev-setup: precommit-bootstrap ## Complete development environment setup
	@echo "$(OK_COLOR)✓ Development environment ready$(NO_COLOR)"

ci-local: precommit-run compose-validate ## Run CI checks locally
	@echo "$(OK_COLOR)✓ All CI checks passed locally$(NO_COLOR)"
