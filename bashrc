#!/bin/bash

# export directory to global variable
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export APP_DIR

# add bin to path
APP_BIN="$APP_DIR/bin"
export APP_BIN
export PATH="$APP_BIN:$PATH"
