# Makefile for automatic README generation & Documentation
.PHONY: readme clean help docs docs-pages docs-examples

README_FILE := README.md
SCRIPTS_DIR := src/frogchem/cli
PYTHON ?= python3

# Prefer sphinx-build from current environment if available, otherwise fallback to conda base
SPHINXBUILD := $(shell which sphinx-build 2>/dev/null)
ifeq ($(SPHINXBUILD),)
  SPHINXBUILD := /opt/homebrew/Caskroom/miniconda/base/bin/sphinx-build
endif

readme:
	@echo "Generating $(README_FILE)..."
	$(PYTHON) generate_readme.py

docs-pages:
	$(PYTHON) generate_readme.py --docs-only

docs-examples:
	$(PYTHON) docs/build_examples.py $(CMD)

docs:
	@echo "Building Sphinx documentation with $(SPHINXBUILD)..."
	$(SPHINXBUILD) -b html docs docs/_build/html
	@echo "✅ Documentation built: open docs/_build/html/index.html"

# Target to clean generated README
clean:
	@if [ -f $(README_FILE) ]; then \
		rm $(README_FILE) && echo "🗑️  $(README_FILE) deleted"; \
	else \
		echo "ℹ️  $(README_FILE) not found"; \
	fi
	@if [ -d docs/_build ]; then \
		rm -rf docs/_build && echo "🗑️  docs/_build deleted"; \
	fi

# Target to show Makefile help
help:
	@echo "Available commands:"
	@echo "  make docs    - Build Sphinx HTML documentation"
	@echo "  make readme  - Generate README.md and docs/cli pages"
	@echo "  make docs-pages    - Regenerate only docs/cli/<command>.md"
	@echo "  make docs-examples - Re-run the real examples (CMD=frog-rrho for one)"
	@echo "  make clean   - Delete generated README.md and docs/_build"
	@echo "  make help    - Show this help message"
	@echo ""
	@echo "Note: __init__.py files are automatically excluded from documentation"

# Default target
.DEFAULT_GOAL := help