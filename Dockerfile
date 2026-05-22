# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface

WORKDIR /opt/custom-sam-peft

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project \
            --extra qlora --extra tensorboard --extra wandb --extra jupyter

COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen \
            --extra qlora --extra tensorboard --extra wandb --extra jupyter

ENV PATH="/opt/custom-sam-peft/.venv/bin:$PATH"

LABEL org.opencontainers.image.source="https://github.com/NguyenJus/custom-sam-peft" \
      org.opencontainers.image.description="Parameter-efficient finetuning of SAM3.1 with LoRA/QLoRA" \
      org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /workspace
EXPOSE 8888

ENTRYPOINT ["custom-sam-peft"]
CMD ["--help"]
