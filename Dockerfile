# Stage 1: build dependencies into a venv
FROM python:3.12-slim AS builder
WORKDIR /app

# System deps needed to build/install wheels (some libs may need compilation)
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     curl \
     build-essential \
     libffi-dev \
     libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VERSION=2.1.4
ENV PATH=${POETRY_HOME}/bin:${PATH}
RUN curl -sSL https://install.python-poetry.org | python3 -

# Create venv (keeps runtime clean)
RUN python -m venv /app/venv
ENV VIRTUAL_ENV=/app/venv
ENV PATH="/app/venv/bin:${PATH}"

# CRITICAL: install into /app/venv, not Poetry's own venv
ENV POETRY_VIRTUALENVS_CREATE=false


# Copy only dependency files first (better Docker layer caching)
COPY backend/pyproject.toml backend/poetry.lock ./

# Install runtime deps (no dev deps, no installing the project itself)
# Poetry 2.x: --only main is the clean way
RUN poetry install --only main --no-root --no-interaction

# Stage 2: runtime image
FROM python:3.12-slim AS runner
WORKDIR /app

# Runtime libs (libgomp1 is commonly needed by faiss / scipy stack)
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /app/venv /app/venv
ENV VIRTUAL_ENV=/app/venv
ENV PATH="/app/venv/bin:${PATH}"

# Copy your application code
COPY . .

# If your FastAPI app is at: src/app/main.py with `app = FastAPI()`
# this will work because we tell uvicorn the app dir is "src"
WORKDIR /app/backend
ENV PYTHONPATH=/app/backend
CMD ["python", "-m", "uvicorn", "src.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]

