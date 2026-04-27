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

# Amazon Lumberyard Bistro (Morgan McGuire's Computer Graphics Archive)
#   https://casual-effects.com/data
# Five archives totalling ~1.4 GB. Each archive is unpacked into a
# subdirectory of "bistro/" with the same name as the archive (Exterior,
# Interior, BuildingTextures, OtherTextures, PropTextures).
BISTRO_BASE_URL="https://casual-effects.com/g3d/data10/research/model/bistro"
BISTRO_DIR="bistro"

# Each entry: "<remote path> <local zip filename> <extract subdirectory>"
BISTRO_FILES=(
    "Exterior.zip Exterior.zip Exterior"
    "Interior.zip Interior.zip Interior"
    "BuildingTextures BuildingTextures.zip BuildingTextures"
    "OtherTextures OtherTextures.zip OtherTextures"
    "PropTextures PropTextures.zip PropTextures"
)

mkdir -p "$BISTRO_DIR"

for entry in "${BISTRO_FILES[@]}"; do
    read -r remote local subdir <<< "$entry"
    local_zip="${BISTRO_DIR}/${local}"
    target_dir="${BISTRO_DIR}/${subdir}"

    if [ ! -f "$local_zip" ]; then
        echo "Downloading Bistro / ${subdir} from casual-effects.com ..."
        curl -L --fail --progress-bar -o "$local_zip" "${BISTRO_BASE_URL}/${remote}"
    else
        echo "Bistro / ${subdir} zip already present at $local_zip, skipping download."
    fi

    echo "Extracting Bistro / ${subdir} into $target_dir ..."
    mkdir -p "$target_dir"
    unzip -o -q "$local_zip" -d "$target_dir"
done

echo "Done."
