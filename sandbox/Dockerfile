FROM python:3.11.4-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV PATH="/uv/bin:${PATH}"

# Install system dependencies
RUN apt update && apt install -y wget build-essential

# Install Python dependencies
RUN uv pip install pandas psutils open-interpreter --system
# Data science packages
# TODO: specify data science packages
RUN uv pip install numpy matplotlib seaborn scikit-learn statsmodels scipy plotly jupyterlab ipython --system

# Create app directory where code will run
WORKDIR /app

# Create directories for read-only mounts
RUN mkdir -p /app/interpreter /app/datasets /app/instructions /app/configs /app/data_manager

# Create directory for logs and checkpoints
RUN mkdir -p /app/logs /app/checkpoints
RUN chmod 777 /app/logs /app/checkpoints

# Set default command
CMD ["python", "-u", "/app/runner.py"]