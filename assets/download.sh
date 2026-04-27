#!/usr/bin/env bash
# Download external assets that are not stored in this repository.
#
# Usage:
#   bash assets/download.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# San Miguel 2.0 (Morgan McGuire's Computer Graphics Archive)
#   https://casual-effects.com/data
SAN_MIGUEL_URL="https://casual-effects.com/g3d/data10/research/model/San_Miguel/San_Miguel.zip"
SAN_MIGUEL_DIR="san_miguel_2.0"
SAN_MIGUEL_ZIP="${SAN_MIGUEL_DIR}/San_Miguel.zip"

mkdir -p "$SAN_MIGUEL_DIR"

if [ ! -f "$SAN_MIGUEL_ZIP" ]; then
    echo "Downloading San Miguel 2.0 (~523 MB) from casual-effects.com ..."
    curl -L --fail --progress-bar -o "$SAN_MIGUEL_ZIP" "$SAN_MIGUEL_URL"
else
    echo "San Miguel zip already present at $SAN_MIGUEL_ZIP, skipping download."
fi

echo "Extracting San Miguel 2.0 into $SAN_MIGUEL_DIR ..."
unzip -o -q "$SAN_MIGUEL_ZIP" -d "$SAN_MIGUEL_DIR"

echo "Done."
