.DEFAULT_GOAL := check

PYTHON ?= python3

.PHONY: check changelog install build check-dist test

check:
	pre-commit run -a

changelog:
	cz bump --changelog

install:
	$(PYTHON) -m pip install -e .

build:
	$(PYTHON) -m build

check-dist: build
	$(PYTHON) -m twine check --strict dist/*

test:
	$(PYTHON) -m unittest discover -s tests -v
