FROM python:3.11-slim

# System deps for Tesseract OCR and trimesh
RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (no torch needed at runtime for cascade)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Streamlit config
RUN mkdir -p /root/.streamlit
RUN printf '[server]\nheadless = true\nport = 8501\naddress = "0.0.0.0"\nenableCORS = false\nenableXsrfProtection = false\n\n[browser]\ngatherUsageStats = false\n' > /root/.streamlit/config.toml

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

ENTRYPOINT ["streamlit", "run", "app.py"]
