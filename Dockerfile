FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY . .
# The trained adapter is in the repo (adapters/default). Build the merged model into the image so the
# container starts without LoRA code paths; this downloads GPT-2 once at build time.
ENV HF_HOME=/srv/.cache/huggingface
RUN python -m app.merge
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "streamlit run app/main.py --server.address 0.0.0.0 --server.port ${PORT}"]
