# pmcp-python — reference Docker image
#
# Build (from pmcp-org root, so this file's context is pmcp-python/):
#   docker build -t pcp:0.5.0 -f pmcp-python/Dockerfile pmcp-python
#
# Or from inside pmcp-python/:
#   docker build -t pcp:0.5.0 .
#
# Default entrypoint runs the in-process arm demo (no hardware needed).
# Override to run the server or the registry, e.g.:
#   docker run --rm pcp:0.5.0 pcp-server --help
#   docker run --rm -p 8080:8080 pcp:0.5.0 pcp-registry
FROM python:3.12-slim

LABEL org.opencontainers.image.title="pcp" \
      org.opencontainers.image.description="Physical Context Protocol — MCP-compatible protocol for commanding physical robots with a mandatory safety pipeline" \
      org.opencontainers.image.source="https://github.com/physicalcontextprotocol/pmcp-python" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="0.5.0"

WORKDIR /app

# Copy metadata first for better layer caching
COPY pyproject.toml README.md CHANGELOG.md LICENSE ./
COPY v05/ ./v05/
COPY pcp/ ./pcp/
COPY sdk/ ./sdk/

# Numerics extra gives us numpy so the shadow simulator's forward-kinematic
# path is available. Core dep is just cryptography.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[numerics]"

ENTRYPOINT ["pcp-demo"]
CMD ["--demo", "arm"]
