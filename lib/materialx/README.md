# Vendored MaterialX libraries

A trimmed copy of the MaterialX standard libraries used by this
project's mtlx loader. See `NOTICE` for details on what is and isn't
included, and `LICENSE` for the upstream license. The update tool applies
the local compatibility patches needed by this evaluator after copying
the upstream files.

Source: https://github.com/AcademySoftwareFoundation/MaterialX, tag `v1.39.5`.

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
git -C third_party/MaterialX fetch --tags
git -C third_party/MaterialX checkout v1.39.5
cargo run --manifest-path tools/update_materialx_libs/Cargo.toml
```

Then update `NOTICE` to record the new tag.
