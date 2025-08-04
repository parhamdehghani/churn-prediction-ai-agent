# Stage 1: Build stage to install dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# Install dependencies into a wheels directory
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt


# Stage 2: Final stage for the production image
FROM python:3.10-slim

WORKDIR /app

# Create a non-root user for better security
RUN addgroup --system app && adduser --system --group app

# Copy the application code and dependencies from the build stage
COPY --from=builder /app/wheels /wheels
COPY ./src /app/src

# Install dependencies from the wheels
RUN pip install --no-cache /wheels/*

# Switch to the non-root user
USER app

# Expose the port the app will run on
EXPOSE 8080

# Run the app with Gunicorn, a production-grade web server
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "src.churn_prediction.api.app:app", "-b", "0.0.0.0:8080"]
