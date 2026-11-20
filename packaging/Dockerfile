FROM python:3.12-slim

ARG JAX_PLATFORM=cpu
ARG PYFCIRT_EXTRAS

ENV JAX_PLATFORMS=$JAX_PLATFORM \
    PYTENSOR_FLAGS=cxx= \
    PYTHONUNBUFFERED=1

COPY dist/*.whl /tmp/wheels/
RUN set -- /tmp/wheels/*.whl; python -m pip install --no-cache-dir "$1$PYFCIRT_EXTRAS" \
    && rm -rf /tmp/wheels

CMD ["python"]
