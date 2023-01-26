#!/usr/bin/env sh

docker run --rm \
    -v $(pwd):/io \
    -e PLAT=manylinux1_x86_64 \
    quay.io/pypa/manylinux1_x86_64 \
    /io/build-wheels.sh
