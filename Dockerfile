FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3

ARG REVISION=unknown
LABEL org.opencontainers.image.source="https://github.com/LCGaoZzz/iobrx" \
      org.opencontainers.image.description="iobrx 0.2.0: portable tumor microenvironment analysis" \
      org.opencontainers.image.version="0.2.0" \
      org.opencontainers.image.revision=$REVISION

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    IOBRX_TESTDATA=/opt/iobrx/tutorials/data

COPY requirements-container.lock /tmp/requirements-container.lock
COPY dist/*.whl /tmp/wheels/
RUN python -m pip install --no-cache-dir --only-binary=:all: --require-hashes -r /tmp/requirements-container.lock \
    && python -m pip install --no-cache-dir --no-deps /tmp/wheels/*.whl \
    && python -m pip check \
    && rm -rf /tmp/wheels /tmp/requirements-container.lock \
    && useradd --create-home --uid 1000 iobrx \
    && mkdir /work && chown iobrx:iobrx /work

COPY tutorials /opt/iobrx/tutorials
COPY LICENSE /opt/iobrx/LICENSE
USER iobrx
WORKDIR /work
CMD ["python", "-c", "import iobrx; print(iobrx.backend_info())"]
