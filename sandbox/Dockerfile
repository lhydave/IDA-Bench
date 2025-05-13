FROM python:3.11.4-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV PATH="/uv/bin:${PATH}"
ENV PATH="/root/.cargo/bin/:$PATH"

# Install system dependencies
RUN apt update && apt install -y wget build-essential

# Copy dependency files
COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock

# 使用冻结的锁文件将项目同步到新环境
WORKDIR /app
RUN uv sync --frozen

# # Install Python dependencies
# RUN uv pip install pandas psutil open-interpreter backoff funcy dataclasses-json --system
# # Data science packages
# # TODO: specify data science packages
# RUN uv pip install numpy matplotlib seaborn scikit-learn statsmodels scipy plotly jupyterlab ipython --system

# Create app directory where code will run
WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

# Create directories for read-only mounts
RUN mkdir -p /app/interpreter /app/datasets /app/instructions /app/configs /app/data_manager

# Create directory for logs and checkpoints
RUN mkdir -p /app/logs /app/checkpoints
RUN chmod 777 /app/logs /app/checkpoints

# Set default command
CMD ["python", "-u", "/app/runner.py"]