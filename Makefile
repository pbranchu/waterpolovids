.PHONY: build test lint shell run

build:
	docker compose build

test:
	python3 -m pytest tests/ -v

lint:
	python3 -m ruff check src/ tests/

shell:
	docker compose run --rm wpv /bin/bash

run:
	docker compose run --rm wpv $(ARGS)
