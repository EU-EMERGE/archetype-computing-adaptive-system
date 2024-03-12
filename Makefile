.DEFAULT_GOAL := check

check:
	pre-commit run -a

changelog:
	cz bump --changelog

install:
	python3 -m venv .buildenv
	.buildenv/bin/pip install poetry
	.buildenv/bin/poetry install
	rm -rf .buildenv
