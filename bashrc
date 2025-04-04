#!/bin/bash

# export directory to global variable
ACT_POST_PROC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ACT_POST_PROC_DIR

# add bin to path
ACT_POST_PROC_BIN="$ACT_POST_PROC_DIR/bin"
export ACT_POST_PROC_BIN
export PATH="$ACT_POST_PROC_BIN:$PATH"
