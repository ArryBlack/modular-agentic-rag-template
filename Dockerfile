FROM python:3.11-slim

WORKDIR /app

# Install libmagic1 (system dependency for python-magic)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*
    
# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose the default ADK web UI port
EXPOSE 8000

# Run the ADK web interface bound to all interfaces
CMD ["bash", "-c", "adk web --host=0.0.0.0 agents"]