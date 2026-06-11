# Multi-stage build to minimize production footprint
FROM python:3.10-slim AS builder

WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.10-slim AS runner
WORKDIR /code

# Copy installed dependencies from builder stage
COPY --from=builder /root/.local /root/.local

# Copy your local project files into the container
COPY app.py .
COPY mrnet_architecture.py .
COPY mrnet_3dcnn_artifacts.pth .
COPY gatekeeper_weights.pth .

ENV PATH=/root/.local/bin:$PATH
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]