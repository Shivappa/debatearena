# ──────────────────────────────────────────────────────────────────────────────
# DebateArenaEnv — HuggingFace Space Dockerfile
#
# Two services, one image:
#   • Port 7860 — Gradio UI  (HF Spaces default)
#   • Port 8000 — FastAPI environment server  (OpenEnv judge endpoint)
#
# HuggingFace Spaces requires the app to listen on port 7860.
# The environment server runs on 8000 as a background process.
#
# Official OpenEnv base image:
#   • Python 3.11 pre-installed
#   • FastAPI + uvicorn pre-installed
#   • WORKDIR /app, PYTHONUNBUFFERED=1, UV_SYSTEM_PYTHON=1
#   • uv as the fast package manager
# ──────────────────────────────────────────────────────────────────────────────
FROM ghcr.io/meta-pytorch/openenv-base:latest

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python path ───────────────────────────────────────────────────────────────
# Ensure both /app and /app/environment are importable
ENV PYTHONPATH=/app

# ── Runtime config ────────────────────────────────────────────────────────────
ENV ENV_PORT=8000
ENV APP_PORT=7860
ENV TOPIC_LEVEL=easy
ENV ENABLE_WEB_INTERFACE=true

# ── Install server-only dependencies (no torch/trl — training runs in Colab) ──
COPY requirements.server.txt ./requirements.server.txt
RUN uv pip install --no-cache -r requirements.server.txt

# ── Copy source code ──────────────────────────────────────────────────────────
COPY . .

# ── Output directories ────────────────────────────────────────────────────────
RUN mkdir -p assets

EXPOSE 7860
EXPOSE 8000

# ── Health check (against the Gradio UI port) ─────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${APP_PORT}/ || exit 1

# ── Entrypoint — start env server in background, then Gradio UI ───────────────
#
#   • server/app.py   — FastAPI RL environment HTTP API  (port 8000)
#   • client/ui.py    — Gradio demo UI                  (port 7860, HF default)
#
CMD ["sh", "-c", "\
    uvicorn server.app:app --host 0.0.0.0 --port ${ENV_PORT} & \
    python client/ui.py \
"]
