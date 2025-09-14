FROM python:3.10-slim

# This is a dummy Dockerfile to force Railway to use Docker builder
# The actual services are defined in railway.toml with their own Dockerfiles

WORKDIR /app

# Copy a minimal requirements.txt to satisfy Railway
COPY action-server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Dummy command - actual services run from railway.toml
CMD ["echo", "Multi-service app - see railway.toml for service definitions"]