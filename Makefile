# Makefile for the Sphinx documentation
.PHONY: help docs

SPHINXBUILD ?= sphinx-build


docs:
	@echo "Building Sphinx documentation with $(SPHINXBUILD)..."
	$(SPHINXBUILD) -b html docs docs/_build/html
	@echo "✅ Documentation built: open docs/_build/html/index.html"


# Target to show Makefile help
help:
	@echo "Available commands:"
	@echo "  make docs    - Build Sphinx HTML documentation"	
	@echo "  make help    - Show this help message"

# Default target
.DEFAULT_GOAL := help