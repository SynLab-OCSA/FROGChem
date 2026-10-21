FROM python:3.11-slim

# Install essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy project specification and source code
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install paclab-lib in editable mode and Streamlit
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir streamlit

# Create directories for matplotlib cache and user files
RUN mkdir -p /tmp/matplotlib /app/files && chmod -R 777 /tmp/matplotlib

# Environment configuration
ENV MPLCONFIGDIR=/tmp/matplotlib \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/frogchem/reaction_plotter/web_gui.py"]
