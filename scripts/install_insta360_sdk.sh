#!/usr/bin/env bash
set -euo pipefail

# Insta360 Media SDK Installation
#
# The SDK must be downloaded manually from the Insta360 developer portal:
#   https://www.insta360.com/sdk/apply
#
# Steps:
# 1. Create an Insta360 developer account
# 2. Apply for the Media SDK (Linux)
# 3. Download the SDK archive
# 4. Extract to vendor/insta360/ in this repository:
#      mkdir -p vendor/insta360
#      tar xzf MediaSDK_Linux_*.tar.gz -C vendor/insta360/
# 5. Verify:
#      ls vendor/insta360/

VENDOR_DIR="$(cd "$(dirname "$0")/.." && pwd)/vendor/insta360"

if [ -d "$VENDOR_DIR" ] && [ "$(ls -A "$VENDOR_DIR" 2>/dev/null)" ]; then
    echo "Insta360 SDK found at $VENDOR_DIR"
    ls "$VENDOR_DIR"
else
    echo "ERROR: Insta360 SDK not found at $VENDOR_DIR"
    echo "Please download it from https://www.insta360.com/sdk/apply"
    echo "and extract to $VENDOR_DIR"
    exit 1
fi
