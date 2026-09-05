# ─────────────────────────────────────────────────────────────────────────────
# Stage: Runtime image for Student ML API
# Best Practices Applied:
#   - Explicit base image version (no implicit 'latest')
#   - Non-root user for security
#   - requirements.txt copied first to leverage Docker layer caching
#   - Minimal final image size
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# ── Layer Cache Optimization ──────────────────────────────────────────────────
# Copy ONLY requirements first so this layer is cached unless deps change.
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ── Application Code ──────────────────────────────────────────────────────────
# Copied AFTER deps so code changes don't invalidate the pip cache layer.
COPY app.py .

# ── Security: Run as non-root user ────────────────────────────────────────────
RUN adduser --disabled-password --gecos "" appuser
USER appuser

# Expose the application port
EXPOSE 5000

# ── Start the application ─────────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
