# Student Feedback Summarizer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/mbgxgavm8n/badge.svg)](https://ask.thili.ai/b/mbgxgavm8n)

This project creates a Student Feedback Summarizer, an EdTech tool for course instructors. It utilizes LoRA fine-tuning on a language model to generate concise summaries of student comments, streamlining curriculum improvement. The tool demonstrates the power of adapting LLMs for specific domain tasks.

## What you are building

**Goal.** You will build a tool that fine-tunes a language model using LoRA to summarize student feedback, enabling course instructors to quickly understand student comments.

**Done when.** The project is done when you have successfully fine-tuned a pre-trained LLM with LoRA on synthetic student feedback, can generate relevant summaries through a Streamlit UI, and have merged the adapter for efficient inference.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a small dataset of synthetic student feedback and summary pairs.**  
      Hint: Focus on creating diverse examples for effective training.
- [ ] **2. Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).**  
      Hint: Apply the `lfllm-lflora` technique to inject and train the adapter.
- [ ] **3. Develop a Streamlit UI to upload feedback and trigger summarization.**  
      Hint: Use Streamlit components to build an interactive interface.
- [ ] **4. Evaluate the quality of generated summaries against original feedback.**  
      Hint: Design metrics or human evaluation criteria to assess summary relevance and conciseness.
- [ ] **5. Implement adapter merging for inference.**  
      Hint: Use the `lfllm-lflora` technique to merge the LoRA adapter with the base model for deployment.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Explore additional quantitative metrics for summary quality beyond basic evaluation.
- Add options for users to adjust summarization parameters (e.g., summary length) in the Streamlit UI.
- Implement functionality to save and load multiple fine-tuned LoRA adapters for different courses or datasets.
