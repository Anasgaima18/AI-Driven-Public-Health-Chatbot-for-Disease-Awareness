FROM python:3.10-slim

# This is a dummy Dockerfile to force Railway to use Docker builder
# The actual services are defined in railway.toml with their own Dockerfiles

WORKDIR /app

# Copy source (no pip installs to avoid timeout)
COPY . .

# Dummy command - actual services run from railway.toml
CMD ["echo", "Multi-service app - see railway.toml for service definitions"]