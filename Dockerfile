FROM python:3.11-slim

WORKDIR /app
# CPU-only torch keeps the image ~1.5 GB smaller (no GPU needed for inference)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd -m -u 1000 app && chown -R app /app
USER app

ENV MLFLOW_DISABLE_AGENT_HINT=1
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
