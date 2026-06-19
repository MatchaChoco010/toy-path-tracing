# toy-path-tracing renderer

## 使い方

```bash
cargo run --release -- [OPTIONS]
```

## 実行例

```bash
cargo run --release -- --scene 0 --spp 64 -o result/scene-0.png
```

```bash
cargo run --release -- --scene 1 --width 1280 --height 720 --spp 128 -o result/scene-1.png
```

```bash
cargo run --release -- --scene 0 --spp 64 --output-display "sRGB - Display" --output-view "ACES 2.0 - SDR 100 nits (Rec.709)" -o result/scene-0.png
```

```bash
cargo run --release -- --scene 0 --spp 64 -o result/scene-0.exr
```

## CLI 引数

| 引数 | 既定値 | 説明 |
| --- | --- | --- |
| `-o, --output <OUTPUT>` | `result/output.png` | 出力画像の保存先です。親ディレクトリが存在しなければ作成されます。対応拡張子は `.exr`, `.png`, `.webp`, `.avif` です。 |
| `--width <WIDTH>` | `512` | 出力画像の横幅です。`1` 以上を指定できます。 |
| `--height <HEIGHT>` | `512` | 出力画像の高さです。`1` 以上を指定できます。 |
| `--scene <SCENE>` | `0` | 読み込むシーン番号です。 |
| `--spp <SPP>` | `32` | 1 ピクセルあたりのサンプル数です。`1` 以上を指定できます。 |
| `--depth <DEPTH>` | `16` | パストレースの最大バウンス数です。`1` 以上を指定できます。 |
| `-i, --integrator <INTEGRATOR>` | `mis` | 使用する integrator です。`mis`, `pt`, `nee`, `bdpt` を指定できます。 |
| `--render-threads <RENDER_THREADS>` | CPU 論理コア数 - 2（下限 1） | Rayon がレンダリングや BVH / light tree 構築に使うスレッド数です。 |
| `--ocio-config <OCIO_CONFIG>` | `ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5` | 使用する OCIO config です。`.ocio` / `.ocioz` のパス、または OCIO built-in config URI を指定できます。 |
| `--ocio-rendering-space <OCIO_RENDERING_SPACE>` | config の `rendering` role、なければ `scene_linear` role | レンダラー内部の scene-linear 作業色空間です。 |
| `--texture-color-space <TEXTURE_COLOR_SPACE>` | `sRGB - Texture` | 明示 color space がない color texture の入力色空間です。 |
| `--output-display <OUTPUT_DISPLAY>` | 非 EXR は `sRGB - Display`、EXR は未指定 | 出力に適用する OCIO display です。 |
| `--output-view <OUTPUT_VIEW>` | 非 EXR は `ACES 2.0 - SDR 100 nits (Rec.709)`、EXR は未指定 | 出力に適用する OCIO view です。 |
| `--log-filter <LOG_FILTER>` | `RUST_LOG` または `info` | `tracing_subscriber::EnvFilter` 形式のログフィルタです。 |
| `-h, --help` | なし | ヘルプを表示します。 |

## MaterialX

MaterialX の `opacity < 1` は `thin_walled = true` のマテリアルで使用してください。
