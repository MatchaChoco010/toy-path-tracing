#!/usr/bin/env bash
# Download external assets that are not stored in this repository.
#
# Usage:
#   bash assets/download.sh
#
# After running this script, "assets/" contains everything required to render
# scene 0..24 from a freshly-cloned checkout. The script is idempotent: it
# skips downloads and conversions whose outputs are already up to date.
#
# Requirements: curl, unzip, git, cargo, a C/C++ toolchain, and cmake. Cargo
# is used to compile the in-repo `convert-bistro` binary, which links against
# assimp via the russimp Rust bindings (statically built from source).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

# San Miguel 2.0 (Morgan McGuire's Computer Graphics Archive)
#   https://casual-effects.com/data
SAN_MIGUEL_URL="https://casual-effects.com/g3d/data10/research/model/San_Miguel/San_Miguel.zip"
SAN_MIGUEL_DIR="san_miguel_2.0"
SAN_MIGUEL_ZIP="${SAN_MIGUEL_DIR}/San_Miguel.zip"
SAN_MIGUEL_MARKER="${SAN_MIGUEL_DIR}/.extracted"

mkdir -p "$SAN_MIGUEL_DIR"

if [ ! -f "$SAN_MIGUEL_ZIP" ]; then
    echo "Downloading San Miguel 2.0 (~523 MB) from casual-effects.com ..."
    curl -L --fail --progress-bar -o "$SAN_MIGUEL_ZIP" "$SAN_MIGUEL_URL"
else
    echo "San Miguel zip already present at $SAN_MIGUEL_ZIP, skipping download."
fi

if [ ! -f "$SAN_MIGUEL_MARKER" ]; then
    echo "Extracting San Miguel 2.0 into $SAN_MIGUEL_DIR ..."
    unzip -o -q "$SAN_MIGUEL_ZIP" -d "$SAN_MIGUEL_DIR"
    touch "$SAN_MIGUEL_MARKER"
else
    echo "San Miguel already extracted, skipping."
fi

# Original Amazon Lumberyard Bistro (NVIDIA ORCA, FBX format)
#   https://developer.nvidia.com/orca/amazon-lumberyard-bistro
# The landing URL "https://developer.nvidia.com/bistro" issues a 302 redirect
# to a tokenized download URL on developer.download.nvidia.com. curl -L follows
# the redirect chain automatically; no token handling is needed in this script.
BISTRO_URL="https://developer.nvidia.com/bistro"
BISTRO_DIR="bistro"
BISTRO_ZIP="${BISTRO_DIR}/Bistro_v5_2.zip"
BISTRO_EXTRACT_MARKER="${BISTRO_DIR}/.extracted"
BISTRO_GLTF_DIR="${BISTRO_DIR}/gltf"

mkdir -p "$BISTRO_DIR"

if [ ! -f "$BISTRO_ZIP" ]; then
    echo "Downloading original Bistro (~894 MB) from developer.nvidia.com ..."
    curl -L --fail --progress-bar -o "$BISTRO_ZIP" "$BISTRO_URL"
else
    echo "Original Bistro zip already present at $BISTRO_ZIP, skipping download."
fi

if [ ! -f "$BISTRO_EXTRACT_MARKER" ]; then
    echo "Extracting original Bistro into $BISTRO_DIR ..."
    unzip -o -q "$BISTRO_ZIP" -d "$BISTRO_DIR"
    touch "$BISTRO_EXTRACT_MARKER"
else
    echo "Original Bistro already extracted, skipping."
fi

# russimp-sys 2.0.2 ships only headers on crates.io (so its `static-link`
# feature cannot build assimp from the registry copy) and on Linux x86_64 the
# build script forgets to add cmake's `lib64/` install path to the rustc link
# search list. Clone the upstream source with submodules and apply the small
# in-repo patch that adds the missing search path. The resulting working tree
# is referenced via `[patch.crates-io]` in the workspace root Cargo.toml.
RUSSIMP_SYS_REPO="https://github.com/jkvargas/russimp-sys.git"
RUSSIMP_SYS_TAG="v2.0.2"
RUSSIMP_SYS_DIR="$REPO_DIR/tools/russimp-sys"
RUSSIMP_SYS_PATCH="$REPO_DIR/tools/russimp-sys-lib64.patch"
if [ ! -d "$RUSSIMP_SYS_DIR" ]; then
    echo "Cloning russimp-sys ${RUSSIMP_SYS_TAG} with assimp submodule ..."
    git clone --quiet --recurse-submodules --depth 1 \
        --branch "$RUSSIMP_SYS_TAG" "$RUSSIMP_SYS_REPO" "$RUSSIMP_SYS_DIR"
fi
if ! grep -q 'cmake_dir.join("lib64")' "$RUSSIMP_SYS_DIR/build.rs"; then
    echo "Applying lib64 link-search patch to russimp-sys/build.rs ..."
    git -C "$RUSSIMP_SYS_DIR" apply "$RUSSIMP_SYS_PATCH"
fi

echo "Building convert-bistro helper via cargo (this builds assimp on first run) ..."
(cd "$REPO_DIR" && cargo build --release -p convert-bistro --quiet)
CONVERT_BISTRO_BIN="$REPO_DIR/target/release/convert-bistro"
if [ ! -x "$CONVERT_BISTRO_BIN" ]; then
    echo "convert-bistro binary was not produced at $CONVERT_BISTRO_BIN" >&2
    exit 1
fi

mkdir -p "$BISTRO_GLTF_DIR"

# Convert each Bistro FBX into a self-contained glTF directory:
#   bistro/gltf/<Name>/<Name>.gltf
#   bistro/gltf/<Name>/<Name>.bin
#   bistro/gltf/<Name>/textures/*.png
shopt -s nullglob globstar
for fbx in "$BISTRO_DIR"/Bistro_v5_2/**/*.fbx "$BISTRO_DIR"/Bistro_v5_2/*.fbx; do
    name="$(basename "$fbx" .fbx)"
    out_dir="${BISTRO_GLTF_DIR}/${name}"
    gltf="${out_dir}/${name}.gltf"
    if [ -f "$gltf" ] && [ "$gltf" -nt "$fbx" ]; then
        echo "glTF already up to date for $(basename "$fbx"), skipping conversion."
        continue
    fi
    echo "Converting $(basename "$fbx") -> ${gltf} ..."
    rm -rf "$out_dir"
    "$CONVERT_BISTRO_BIN" "$fbx" "$out_dir" "$name"
done
shopt -u nullglob globstar

echo "Done."
