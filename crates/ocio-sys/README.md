# ocio-sys

Low-level C ABI bindings for the vendored OpenColorIO C++ API.

This crate owns the C++ shim boundary. It intentionally exposes opaque handles
and plain C functions only; renderer policy belongs in higher-level crates.
