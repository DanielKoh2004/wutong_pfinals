# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.13.9
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Fix matplotlib config directory
ENV MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Download dependencies as a separate step to take advantage of Docker's caching.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Copy the source code into the container.
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p /app/Datasets/Student/Results \
    /app/Datasets/Fraud/Results \
    /app/models \
    /tmp/matplotlib && \
    chmod -R 777 /app/Datasets /app/models /tmp/matplotlib

# Expose the port that the application listens on.
EXPOSE 7860

# Run the application.
CMD ["python", "src/main.py"]
