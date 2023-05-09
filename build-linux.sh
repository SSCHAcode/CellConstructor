#!/usr/bin/env sh


docker run --rm \
    -v $(pwd):/io \
    -e PLAT=manylinux2014_x86_64 \
    quay.io/pypa/manylinux2014_x86_64 \
    /io/build-wheels.sh
