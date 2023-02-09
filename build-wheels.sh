#!/bin/bash
set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    elif echo "$wheel" | grep -q manylinux; then
        echo "Skipping wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /io/dist/
    fi
}


# Install a system package required by our library
yum install -y atlas-devel lapack-devel blas-devel libjpeg-devel

mkdir -p /io/dist/

# Compile wheels (only python >= 3.5)
for PYBIN in /opt/python/cp3*/bin; do
    "${PYBIN}/pip" install -r /io/requirements.txt
    cd /io; "${PYBIN}/python" setup.py build
    "${PYBIN}/pip" wheel /io/ --no-deps -w /io/dist/
done

# Bundle external shared libraries into the wheels
for whl in /io/dist/*.whl; do
    repair_wheel "$whl"
done
