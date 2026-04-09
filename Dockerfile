# ─────────────────────────────
# Base Image
# ─────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL org.opencontainers.image.title="Robot Assembly QA"
LABEL org.opencontainers.image.description="OpenEnv robot arm pick & place environment"
LABEL org.opencontainers.image.tags="openenv, robotics"

# ─────────────────────────────
# Working Directory
# ─────────────────────────────
WORKDIR /app

# ─────────────────────────────
# Install Dependencies
# ─────────────────────────────
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────
# Copy Project Files
# ─────────────────────────────
COPY . .

# 🔥 IMPORTANT FIX
ENV PYTHONPATH=/app

# ─────────────────────────────
# Expose Port (optional)
# ─────────────────────────────
EXPOSE 7860

# ─────────────────────────────
# Default Command
# ─────────────────────────────
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
