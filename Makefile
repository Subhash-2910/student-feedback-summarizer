# One command to rebuild everything: make all
PY ?= .venv/bin/python
export HF_HOME := $(CURDIR)/.cache/huggingface

.PHONY: all data train evaluate merge app test
all: data train evaluate merge

data:      ## Milestone 1: write dataset/train.jsonl and dataset/test.jsonl
	$(PY) -m app.dataset
train:     ## Milestone 2: train the LoRA adapter -> adapters/default
	$(PY) -m app.train
evaluate:  ## Milestone 4: compare against baselines -> results/
	$(PY) -m app.evaluate
merge:     ## Milestone 5: merge adapter into the base model -> merged/default
	$(PY) -m app.merge
app:       ## Milestone 3: the Streamlit UI
	$(PY) -m streamlit run app/main.py
test:
	$(PY) -m pytest -q
