#!/bin/bash -e

# Installs system dependencies required to run LightGBM on macOS runners.
# Invoked by .github/workflows/release.yml on macOS jobs.

if [ "$(uname)" = "Darwin" ]; then
    echo "installing necessary dependencies..."
    brew install libomp

    LIBOMP_PREFIX="$(brew --prefix libomp)"

    echo "Verifying libomp installation..."
    ls "$LIBOMP_PREFIX/lib/libomp.dylib"

    {
        echo "DYLD_LIBRARY_PATH=$LIBOMP_PREFIX/lib:\$DYLD_LIBRARY_PATH"
        echo "LDFLAGS=-L$LIBOMP_PREFIX/lib"
        echo "CPPFLAGS=-I$LIBOMP_PREFIX/include"
    } >> "$GITHUB_ENV"
else
    echo "This script is intended to run on macOS (Darwin)."
fi
