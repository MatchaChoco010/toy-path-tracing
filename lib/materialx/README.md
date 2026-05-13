# Vendored MaterialX libraries

A trimmed copy of the MaterialX standard libraries used by this
project's mtlx loader. See `NOTICE` for details on what is and isn't
included, and `LICENSE` for the upstream license.

Source: https://github.com/AcademySoftwareFoundation/MaterialX, tag `v1.39.4`.

## Layout

```
libraries/
  stdlib/        # Standard pattern / math / channel / logical nodes
  pbrlib/        # PBR BSDF/EDF nodedefs and a few NG implementations
  bxdf/          # Standard shader presets (standard_surface, open_pbr_surface,
                 # disney_principled, gltf_pbr, usd_preview_surface)
  nprlib/        # NPR shading nodes (gooch_shade, viewdirection, facingratio)
```

## Re-vendoring

To pull a newer revision of the standard libraries:

```bash
TAG=v1.39.4    # or newer
ROOT=$(git rev-parse --show-toplevel)
LIB="$ROOT/lib/materialx/libraries"

curl -sSL -o "$ROOT/lib/materialx/LICENSE" \
  "https://raw.githubusercontent.com/AcademySoftwareFoundation/MaterialX/$TAG/LICENSE"

for path in \
  stdlib/stdlib_defs.mtlx stdlib/stdlib_ng.mtlx \
  pbrlib/pbrlib_defs.mtlx pbrlib/pbrlib_ng.mtlx \
  bxdf/standard_surface.mtlx bxdf/disney_principled.mtlx \
  bxdf/open_pbr_surface.mtlx bxdf/usd_preview_surface.mtlx \
  bxdf/gltf_pbr.mtlx \
  nprlib/nprlib_defs.mtlx nprlib/nprlib_ng.mtlx
do
  curl -sSL -o "$LIB/$path" \
    "https://raw.githubusercontent.com/AcademySoftwareFoundation/MaterialX/$TAG/libraries/$path"
done
```

Then update `NOTICE` to record the new tag.
