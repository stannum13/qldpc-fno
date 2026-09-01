FROM python:3.14-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY experiments ./experiments
COPY configs ./configs
COPY scripts ./scripts

RUN uv sync --frozen --no-dev \
    && /app/.venv/bin/python experiments/01_build_lp_codes.py --out /app/campaign-code \
    && /app/.venv/bin/python experiments/02_validate_lp_codes.py --code /app/campaign-code

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED="1" \
    CAMPAIGN_CONFIG="/app/configs/accuracy_campaign.json" \
    CAMPAIGN_CODE="/app/campaign-code" \
    CAMPAIGN_WORKDIR="/tmp/qldpc-fno-work"

ENTRYPOINT ["/app/.venv/bin/python", "-m", "qldpc_fno.campaign.runner"]
