from __future__ import annotations

from pathlib import Path

import PyOpenColorIO as ocio

DISPLAY = "sRGB - Display"
RENDERING_SPACE = "Linear sRGB"
TEXTURE_SPACE = "sRGB - Texture"
RAW_SPACE = "Raw"
ACES_INTERCHANGE_SPACE = "ACES2065-1"
ACES_LOG_SPACE = "ACEScct"
CIE_XYZ_D65_INTERCHANGE_SPACE = "CIE XYZ-D65 - Display-referred"
REINHARD_VIEW = "Reinhard"
REINHARD_LUT = "reinhard_33.cube"
REINHARD_LUT_SIZE = 33
REINHARD_DOMAIN_MAX = 16.0
SRGB_GAMMA = [2.4, 2.4, 2.4, 1.0]
SRGB_OFFSET = [0.055, 0.055, 0.055, 0.0]
XYZ_D65_TO_LINEAR_SRGB = [
    3.240969941905,
    -1.537383177570,
    -0.498610760293,
    0.0,
    -0.969243636281,
    1.875967501508,
    0.041555057407,
    0.0,
    0.055630079697,
    -0.203976958889,
    1.056971514243,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
]


def reinhard(value: float) -> float:
    return value / (1.0 + value)


def create_linear_srgb_space() -> ocio.ColorSpace:
    color_space = ocio.ColorSpace(ocio.REFERENCE_SPACE_SCENE, name=RENDERING_SPACE)
    color_space.addAlias("linear_srgb")
    color_space.addAlias("lin_srgb")
    color_space.addAlias("lin_rec709")
    color_space.setFamily("Rendering")
    color_space.setBitDepth(ocio.BIT_DEPTH_F32)
    color_space.setEncoding("scene-linear")
    color_space.setAllocation(ocio.ALLOCATION_UNIFORM)
    color_space.setAllocationVars([0.0, REINHARD_DOMAIN_MAX])
    return color_space


def create_texture_space() -> ocio.ColorSpace:
    color_space = ocio.ColorSpace(ocio.REFERENCE_SPACE_SCENE, name=TEXTURE_SPACE)
    color_space.addAlias("srgb_texture")
    color_space.addAlias("srgb")
    color_space.setFamily("Texture")
    color_space.setBitDepth(ocio.BIT_DEPTH_F32)
    color_space.setEncoding("sdr-video")
    color_space.setAllocation(ocio.ALLOCATION_UNIFORM)
    color_space.setTransform(
        ocio.ExponentWithLinearTransform(gamma=SRGB_GAMMA, offset=SRGB_OFFSET),
        ocio.COLORSPACE_DIR_TO_REFERENCE,
    )
    return color_space


def create_raw_space() -> ocio.ColorSpace:
    color_space = ocio.ColorSpace(ocio.REFERENCE_SPACE_SCENE, name=RAW_SPACE)
    color_space.setFamily("Utility")
    color_space.setBitDepth(ocio.BIT_DEPTH_F32)
    color_space.setIsData(True)
    color_space.setEncoding("data")
    color_space.setAllocation(ocio.ALLOCATION_UNIFORM)
    return color_space


def create_display_space() -> ocio.ColorSpace:
    transform = ocio.GroupTransform()
    transform.appendTransform(
        ocio.ExponentWithLinearTransform(
            gamma=SRGB_GAMMA,
            offset=SRGB_OFFSET,
            direction=ocio.TRANSFORM_DIR_INVERSE,
        )
    )
    transform.appendTransform(
        ocio.RangeTransform(
            minInValue=0.0,
            minOutValue=0.0,
            maxInValue=1.0,
            maxOutValue=1.0,
        )
    )

    color_space = ocio.ColorSpace(ocio.REFERENCE_SPACE_DISPLAY, name=DISPLAY)
    color_space.setFamily("Display")
    color_space.setBitDepth(ocio.BIT_DEPTH_F32)
    color_space.setEncoding("sdr-video")
    color_space.setAllocation(ocio.ALLOCATION_UNIFORM)
    color_space.setTransform(transform, ocio.COLORSPACE_DIR_FROM_REFERENCE)
    return color_space


def create_aces2065_to_linear_srgb_transform() -> ocio.GroupTransform:
    transform = ocio.GroupTransform()
    transform.appendTransform(
        ocio.BuiltinTransform("ACEScg_to_ACES2065-1", ocio.TRANSFORM_DIR_INVERSE)
    )
    transform.appendTransform(ocio.BuiltinTransform("UTILITY - ACES-AP1_to_LINEAR-REC709_BFD"))
    return transform


def create_aces_interchange_space() -> ocio.ColorSpace:
    color_space = ocio.ColorSpace(ocio.REFERENCE_SPACE_SCENE, name=ACES_INTERCHANGE_SPACE)
    color_space.setFamily("Interchange")
    color_space.setBitDepth(ocio.BIT_DEPTH_F32)
    color_space.setEncoding("scene-linear")
    color_space.setAllocation(ocio.ALLOCATION_LG2)
    color_space.setAllocationVars([-10.0, 6.0, 0.00390625])
    color_space.setTransform(
        create_aces2065_to_linear_srgb_transform(),
        ocio.COLORSPACE_DIR_TO_REFERENCE,
    )
    return color_space


def create_acescct_space() -> ocio.ColorSpace:
    transform = ocio.GroupTransform()
    transform.appendTransform(ocio.BuiltinTransform("ACEScct_to_ACES2065-1"))
    transform.appendTransform(create_aces2065_to_linear_srgb_transform())

    color_space = ocio.ColorSpace(ocio.REFERENCE_SPACE_SCENE, name=ACES_LOG_SPACE)
    color_space.setFamily("Log")
    color_space.setBitDepth(ocio.BIT_DEPTH_F32)
    color_space.setEncoding("log")
    color_space.setAllocation(ocio.ALLOCATION_UNIFORM)
    color_space.setTransform(transform, ocio.COLORSPACE_DIR_TO_REFERENCE)
    return color_space


def create_cie_xyz_d65_interchange_space() -> ocio.ColorSpace:
    color_space = ocio.ColorSpace(ocio.REFERENCE_SPACE_DISPLAY, name=CIE_XYZ_D65_INTERCHANGE_SPACE)
    color_space.addAlias("cie_xyz_d65_display")
    color_space.setFamily("Interchange")
    color_space.setBitDepth(ocio.BIT_DEPTH_F32)
    color_space.setEncoding("display-linear")
    color_space.setAllocation(ocio.ALLOCATION_UNIFORM)
    color_space.setTransform(
        ocio.MatrixTransform(matrix=XYZ_D65_TO_LINEAR_SRGB),
        ocio.COLORSPACE_DIR_TO_REFERENCE,
    )
    return color_space


def create_reinhard_lut_transform() -> ocio.Lut3DTransform:
    lut = ocio.Lut3DTransform(REINHARD_LUT_SIZE)
    lut.setInterpolation(ocio.INTERP_LINEAR)

    last_index = REINHARD_LUT_SIZE - 1
    for index_r in range(REINHARD_LUT_SIZE):
        red = reinhard(index_r / last_index * REINHARD_DOMAIN_MAX)
        for index_g in range(REINHARD_LUT_SIZE):
            green = reinhard(index_g / last_index * REINHARD_DOMAIN_MAX)
            for index_b in range(REINHARD_LUT_SIZE):
                blue = reinhard(index_b / last_index * REINHARD_DOMAIN_MAX)
                lut.setValue(index_r, index_g, index_b, red, green, blue)

    return lut


def create_reinhard_view_transform() -> ocio.GroupTransform:
    transform = ocio.GroupTransform()
    transform.appendTransform(
        ocio.RangeTransform(
            minInValue=0.0,
            minOutValue=0.0,
            maxInValue=REINHARD_DOMAIN_MAX,
            maxOutValue=1.0,
        )
    )
    transform.appendTransform(
        ocio.FileTransform(src=REINHARD_LUT, interpolation=ocio.INTERP_LINEAR)
    )
    return transform


def create_reinhard_bake_target_space() -> ocio.ColorSpace:
    color_space = ocio.ColorSpace(ocio.REFERENCE_SPACE_SCENE, name="Reinhard Linear sRGB")
    color_space.setFamily("Bake")
    color_space.setBitDepth(ocio.BIT_DEPTH_F32)
    color_space.setAllocation(ocio.ALLOCATION_UNIFORM)
    color_space.setAllocationVars([0.0, 1.0])
    color_space.setTransform(create_reinhard_lut_transform(), ocio.COLORSPACE_DIR_FROM_REFERENCE)
    return color_space


def create_file_rules() -> ocio.FileRules:
    rules = ocio.FileRules()
    rules.insertPathSearchRule(0)
    rules.setDefaultRuleColorSpace(TEXTURE_SPACE)
    return rules


def create_lut_bake_config() -> ocio.Config:
    config = ocio.Config()
    config.setVersion(2, 5)
    config.setName("linear-srgb-reinhard-lut-bake")
    config.setFamilySeparator("/")
    config.setRole(ocio.ROLE_SCENE_LINEAR, RENDERING_SPACE)
    config.setRole("rendering", RENDERING_SPACE)
    config.addColorSpace(create_linear_srgb_space())
    config.addColorSpace(create_reinhard_bake_target_space())
    return config


def bake_reinhard_lut(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    baker = ocio.Baker()
    baker.setConfig(create_lut_bake_config())
    baker.setFormat("resolve_cube")
    baker.setInputSpace(RENDERING_SPACE)
    baker.setTargetSpace("Reinhard Linear sRGB")
    baker.setCubeSize(REINHARD_LUT_SIZE)
    baker.bake(str(path))


def create_config() -> ocio.Config:
    config = ocio.Config()
    config.setVersion(2, 5)
    config.setName("linear-srgb-reinhard")
    config.setDescription("Linear sRGB rendering config with a Reinhard display view.")
    config.setSearchPath("luts")
    config.setFamilySeparator("/")
    config.setFileRules(create_file_rules())

    config.setRole(ocio.ROLE_INTERCHANGE_SCENE, ACES_INTERCHANGE_SPACE)
    config.setRole(ocio.ROLE_INTERCHANGE_DISPLAY, CIE_XYZ_D65_INTERCHANGE_SPACE)
    config.setRole("color_picking", TEXTURE_SPACE)
    config.setRole(ocio.ROLE_COLOR_TIMING, ACES_LOG_SPACE)
    config.setRole("compositing_linear", RENDERING_SPACE)
    config.setRole(ocio.ROLE_COMPOSITING_LOG, ACES_LOG_SPACE)
    config.setRole(ocio.ROLE_DATA, RAW_SPACE)
    config.setRole(ocio.ROLE_DEFAULT, TEXTURE_SPACE)
    config.setRole("rendering", RENDERING_SPACE)
    config.setRole(ocio.ROLE_SCENE_LINEAR, RENDERING_SPACE)
    config.setRole("texture_paint", TEXTURE_SPACE)

    config.addColorSpace(create_linear_srgb_space())
    config.addColorSpace(create_texture_space())
    config.addColorSpace(create_raw_space())
    config.addColorSpace(create_display_space())
    config.addColorSpace(create_aces_interchange_space())
    config.addColorSpace(create_acescct_space())
    config.addColorSpace(create_cie_xyz_d65_interchange_space())

    view_transform = ocio.ViewTransform(ocio.REFERENCE_SPACE_SCENE, name=REINHARD_VIEW)
    view_transform.setFamily("Tone Mapping")
    view_transform.setTransform(
        create_reinhard_view_transform(), ocio.VIEWTRANSFORM_DIR_FROM_REFERENCE
    )
    config.addViewTransform(view_transform)

    config.addSharedView(REINHARD_VIEW, REINHARD_VIEW, DISPLAY)
    config.addSharedView("Raw", "", RAW_SPACE)
    config.addDisplaySharedView(DISPLAY, REINHARD_VIEW)
    config.addDisplaySharedView(DISPLAY, "Raw")
    config.setActiveDisplays(DISPLAY)
    config.setActiveViews(f"{REINHARD_VIEW}, Raw")
    config.setDefaultViewTransformName(REINHARD_VIEW)
    config.validate()
    return config


def write_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(create_config().serialize(), encoding="utf-8", newline="\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "assets" / "ocio_configs" / "srgb_reinhard"
    bake_reinhard_lut(config_dir / "luts" / REINHARD_LUT)
    write_config(config_dir / "config.ocio")
    print(config_dir / "config.ocio")


if __name__ == "__main__":
    main()
