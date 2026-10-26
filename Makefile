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


docs:
	@echo "Building Sphinx documentation with $(SPHINXBUILD)..."
	$(SPHINXBUILD) -b html docs docs/_build/html
	@echo "✅ Documentation built: open docs/_build/html/index.html"


# Target to show Makefile help
help:
	@echo "Available commands:"
	@echo "  make docs    - Build Sphinx HTML documentation"	
	@echo "  make help    - Show this help message"
	@echo ""
	@echo "Note: __init__.py files are automatically excluded from documentation"

# Default target
.DEFAULT_GOAL := help