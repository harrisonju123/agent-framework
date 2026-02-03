#!/bin/bash
set -e

TARGET_REPO="$1"
if [ -z "$TARGET_REPO" ]; then
  echo "Usage: $0 <owner/repo>"
  echo "Example: $0 justworkshr/some-service"
  exit 1
fi

cd "$(dirname "$0")/.."
source scripts/setup-env.sh

echo "Applying S2S auth pattern to $TARGET_REPO..."
agent apply-pattern \
  --reference justworkshr/sui \
  --files "internal/handler/handler.go,internal/config/config.go" \
  --targets "$TARGET_REPO" \
  --description "Add S2S authentication with JWKS and Okta verifiers" \
  --branch-prefix "feature/s2s-auth"
