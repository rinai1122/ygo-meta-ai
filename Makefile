.PHONY: setup submodule card-db server play simulate test lint clean

setup: submodule
	pip install -e ".[dev]"

submodule:
	git submodule update --init --recursive

card-db:
	python scripts/build_card_db.py

server:
	python -m ygo_meta.engine.runner start

play:
	python -m ygo_meta.cli.play

simulate:
	python -m ygo_meta.cli.simulate \
		--archetypes snake_eye blue_eyes hero \
		--staples-dir data/staples/ \
		--episodes 128 \
		--generations 10

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
