SHELL := /bin/bash
# My machine keeps caches on an external drive; elsewhere the tools' defaults are used.
ifneq ($(wildcard /Volumes/Crucial_1TB/cache),)
export UV_CACHE_DIR ?= /Volumes/Crucial_1TB/cache/uv
export PLAYWRIGHT_BROWSERS_PATH ?= /Volumes/Crucial_1TB/cache/ms-playwright
endif
PY := uv run python
AS_OF ?= $(shell date +%F)

.PHONY: help setup browsers download ingest audit simulate simulate-quick scenarios memos memos-pdf report test demo screenshots fixtures all clean

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-15s %s\n", $$1, $$2}'

setup:           ## install pinned dependencies into .venv
	uv sync

browsers:        ## headless Chromium for PDF memos and screenshots (optional)
	uv run playwright install chromium

download:        ## fetch the Upworthy archive from OSF and verify SHA-256
	$(PY) -m expengine.ingest download

ingest:          ## raw CSV -> experiments/arms/quarantine parquet + manifest
	$(PY) -m expengine.ingest ingest

audit:           ## integrity rules, fixed-horizon inference, historical readouts for all 32,487 tests
	$(PY) -m expengine.audit

simulate:        ## seeded Monte Carlo benchmarks, 10,000 replications per scenario (~1 min)
	$(PY) -m expengine.benchmarks

simulate-quick:  ## same benchmarks at 1,000 replications
	$(PY) -m expengine.benchmarks --reps 1000

scenarios:       ## synthetic product experiments -> decisions (exits non-zero if any differs from its documented case)
	$(PY) -m expengine.product

memos:           ## one-page HTML readouts (set AS_OF=YYYY-MM-DD for byte-identical output)
	$(PY) -m expengine.memo --as-of $(AS_OF)

memos-pdf:       ## HTML + PDF readouts (needs `make browsers`)
	$(PY) -m expengine.memo --as-of $(AS_OF) --pdf

report:          ## statistical validation report + figures from persisted outputs
	$(PY) -m expengine.report

test:            ## unit + integration tests
	uv run pytest -q

demo:            ## Streamlit app on http://localhost:8501
	uv run streamlit run app/Home.py

screenshots:     ## capture the five screens (app must be running on :8501)
	$(PY) scripts/screenshots.py http://localhost:8501

fixtures:        ## regenerate the tiny synthetic archive fixtures used by tests
	$(PY) scripts/make_fixtures.py

all: setup download ingest audit simulate scenarios memos report  ## one-command rebuild of everything

clean:           ## remove derived data (keeps the downloaded raw files)
	rm -rf data/processed data/runs data/synthetic
