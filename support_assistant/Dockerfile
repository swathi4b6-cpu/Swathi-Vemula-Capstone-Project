# Use official highly compressed light production linux core layer
FROM python:3.11-slim

# Enforce direct tracking output parameters avoiding buffering lag artifacts
ENV PYTHONUNBUFFERED=1
ENV MOCK_LLM=1

WORKDIR /workspace

# Install system dependencies required for compilation tasks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cache optimization sequence layers
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Migrate source file structures
COPY ./docs ./docs
COPY ./app ./app

# Expose standard container interface port mapping
EXPOSE 7860

# Launch server binding standard communication layer ports
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
