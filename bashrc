#!/bin/bash

# export directory to global variable
ACT_METH_TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ACT_METH_TOOLS_DIR

# add bin to path
ACT_METH_TOOLS_BIN="$ACT_METH_TOOLS_DIR/bin"
export ACT_METH_TOOLS_BIN
export PATH="$ACT_METH_TOOLS_BIN:$PATH"
