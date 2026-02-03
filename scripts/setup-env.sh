#!/bin/bash

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Fallback to gh auth token if GITHUB_TOKEN is not set
export GITHUB_TOKEN="${GITHUB_TOKEN:-$(gh auth token 2>/dev/null)}"
