FROM python:3.12-slim

ARG JAX_PLATFORM=cpu
ARG PYFCIRT_EXTRAS

ENV JAX_PLATFORMS=$JAX_PLATFORM \
    PYTENSOR_FLAGS=cxx= \
    PYTHONUNBUFFERED=1

COPY dist/*.whl /tmp/wheels/
# CUDA 13.0 avoids a device initialization crash on some WSL 2 drivers.
RUN set -- /tmp/wheels/*.whl; \
    set -- "$1$PYFCIRT_EXTRAS"; \
    if [ "$PYFCIRT_EXTRAS" = "[cuda13]" ]; then \
        set -- "$@" nvidia-cuda-runtime==13.0.96; \
    fi; \
    python -m pip install --no-cache-dir "$@" && rm -rf /tmp/wheels

CMD ["python"]
