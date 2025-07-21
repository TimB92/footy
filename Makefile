export PYTHONPATH=./src:

init:
	uv sync

run/train:
	python src/apps/main.py --data data/results.csv --output data/predictions.csv

