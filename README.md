# toy-path-tracing

Rust で学習用のパストレーサーを実装していくためのプロジェクトです。

## 使い方

基本的な実行方法は次のとおりです。

```bash
cargo run --release -- [OPTIONS]
```

現在の CLI では、出力先、出力画像サイズ、シーン番号、1 ピクセルあたりのサンプル数、最大パストレース深度、使用する integrator を指定できます。integrator を省略した場合は、MIS を使う `mis` が既定で選ばれます。

### 実行例

シーン番号と spp、出力先を指定して実行します。

```bash
cargo run --release -- --scene 0 --spp 64 -o result/scene-0.png
```

integrator を明示したい場合は `--integrator` または `-i` を使います。現状は `mis`、`pt`、`nee` を選べます。

```bash
cargo run --release -- --scene 1 --spp 128 --depth 24 -i pt -o result/scene-1-pt.png
```

解像度も含めて指定したい場合は `--width` と `--height` を使います。

```bash
cargo run --release -- --scene 1 --width 1280 --height 720 --spp 128 -o result/scene-1.png
```

## CLI 引数

`cargo run -- --help` の現在の内容は次のオプションです。

| 引数 | 既定値 | 説明 |
| --- | --- | --- |
| `-o, --output <OUTPUT>` | `result/output.png` | 出力画像の保存先です。親ディレクトリが存在しなければ自動で作成されます。 |
| `--width <WIDTH>` | `512` | 出力画像の横幅です。`1` 以上のみ指定できます。 |
| `--height <HEIGHT>` | `512` | 出力画像の高さです。`1` 以上のみ指定できます。 |
| `--scene <SCENE>` | `0` | 読み込むシーン番号です。 |
| `--spp <SPP>` | `32` | Samples Per Pixel。各ピクセルで何本のパスを積分するかを指定します。`1` 以上のみ指定できます。 |
| `--depth <DEPTH>` | `16` | パストレースの最大バウンス数です。`1` 以上のみ指定できます。 |
| `-i, --integrator <INTEGRATOR>` | `mis` | 使用する integrator を指定します。現在は `mis`、`pt`、`nee` を選べます。存在しない名前を指定するとエラーになります。 |
| `-h, --help` | なし | ヘルプを表示します。 |

## 現在のシーン番号

現状のソースコードでは、次のシーンが実装されています。

| シーン番号 | 内容 |
| --- | --- |
| `0` | Cornell box 風の部屋に箱とバニーを配置。 |
| `1` | Cornell box 風の部屋にバニーと 2 つの球を配置。 |
| `2` | Cornell box 風の部屋に完全鏡面の銀色バニーと金色の球を配置。 |
| `3` | Cornell box 風の部屋に透明ガラス球、左右に thin / 通常の水色ガラスバニー、薄青の Lambert バニーを配置。 |
| `4` | Cornell box 風の部屋に、roughness を左から `0.0 / 0.25 / 0.5 / 0.75 / 1.0` にした金色 Conductor GGX 球を 5 つ並べる。 |
| `5` | Cornell box 風の部屋に、roughness `0.3` の銀色 Conductor GGX 球を 3 つ並べ、中央を isotropic、左右を `anisotropy = -1.0 / +1.0` にする。 |
| `6` | Cornell box 風の部屋に、roughness を左から `0.0 / 0.15 / 0.3 / 0.45 / 0.6` にした透明 Dielectric GGX ガラス球を 5 つ、少し宙に浮かせて並べる。 |
| `7` | Cornell box 風の部屋に、roughness `0.3` の薄水色 Dielectric GGX 球を 3 つ並べ、中央を isotropic、左右を `anisotropy = -1.0 / +1.0` にする。 |
| `8` | 広い Lambert 床の上に Conductor GGX 金属球と Dielectric GGX ガラス球を 2 段に並べ、各段とも roughness を左から `0.0 / 0.15 / 0.3 / 0.45 / 0.6 / 0.75` で振る。`assets/sky/` の HDRI を IBL として使う。 |
| `9` | Cornell box の中央に金色 Conductor GGX (`roughness = 0.35`) のバニーを置き、天井のエリアライトと `assets/sky/` の HDRI を併用。カメラはボックスの外。 |
| `10` | 一様な白い環境光 (intensity `1.0`) のもと、上段に Dielectric GGX ガラス球、下段に銀色 Conductor GGX 金属球を、各段とも roughness を `0.0 / 0.15 / 0.3 / 0.45 / 0.6 / 0.75` で 6 つずつ並べる。 |
| `11` | 広い Lambert 床の上に薄青の Lambert バニーを置き、`DirectionalLight` 1 つで照らす。 |
| `12` | Cornell box 風の部屋に薄青 Lambert バニーと金色 / 銅色の Conductor GGX 球を配置し、暖色 / 寒色の `PointLight` 2 灯と、バニーに向けたマゼンタ / ティールの `SpotLight` 2 灯で照らす。 |
| `13` | 広い Lambert 床の上に diffuse バニーを置き、`assets/sky/brown_photostudio_02_4k.hdr` の環境光だけで照らす。 |
| `14` | 広い Lambert 床の上に diffuse バニーを置き、`assets/sky/kloofendal_48d_partly_cloudy_puresky_4k.hdr` の環境光だけで照らす。 |
| `15` | Cornell box 風の部屋に、`assets/models/sphere-color.png` と `assets/models/sphere-roughness.png` を使う Conductor GGX 球と、同じ color texture を使う Lambert 球を斜めに並べる。 |
| `16` | 広い Lambert 床と `assets/sky/kloofendal_48d_partly_cloudy_puresky_4k.hdr` の puresky 環境光のもと、`assets/models/bunny-color.png` を使う Lambert バニーと `assets/models/sphere-color.png` / `assets/models/sphere-roughness.png` を使う Conductor GGX 球を配置。 |
| `17` | `assets/models/floor-brick.png` を貼った Lambert 床に完全鏡面の金属球とガラス球を配置し、`assets/sky/kloofendal_48d_partly_cloudy_puresky_4k.hdr` の IBL で照らす。 |
| `18` | Cornell box 風の部屋に、`assets/models/sphere-normal.png` を normal strength `0.2` で使う Lambert 球と、roughness `0.4` の Conductor GGX 球を斜めに配置。 |
| `19` | 広い Lambert 床と `assets/sky/kloofendal_48d_partly_cloudy_puresky_4k.hdr` の puresky 環境光のもと、`assets/models/sphere-normal.png` を normal strength `0.2` で使う Lambert 球、roughness `0.4` の Conductor GGX 球、完全鏡面の Mirror 球を配置。 |
| `20` | 広い Lambert 床と `assets/sky/kloofendal_48d_partly_cloudy_puresky_4k.hdr` の puresky 環境光のもと、`assets/models/dragon.obj` と dragon 用の base color / metallic / roughness / normal texture を使う SimplePBR ドラゴンを配置。 |
| `21` | 広い Lambert 床と puresky 環境光のもと、SimplePBR ドラゴン、roughness `0.4` の金色 Conductor GGX バニー、`Glass` 球、薄青の NormalizedLambert バニーを並べる。 |
| `22` | `assets/san_miguel_2.0/san-miguel-low-poly.obj` のローポリ版 San Miguel を `assets/sky/kloofendal_48d_partly_cloudy_puresky_4k.hdr` の puresky 環境光で照らす。OBJ マテリアルは SimplePBR / DielectricGgx / Emissive に振り分け、`map_Kd` の alpha は `opacity_texture` として使う。 |
| `23` | `assets/san_miguel_2.0/san-miguel.obj` の通常版 San Miguel を puresky 環境光で照らす。マテリアル割り当ては scene 22 と同じ。 |
| `24` | `assets/bistro/Exterior/exterior.obj` と `assets/bistro/Interior/interior.obj` を結合した Amazon Lumberyard Bistro を、環境光なしでシーン中の emissive ポリゴンだけで照らす。OBJ マテリアルは SimplePBR / DielectricGgx / Emissive に振り分け、`map_Ke` は `EmissiveMaterial` の `color_texture` として使う。 |
| `25` | `assets/sky/studio_small_08_4k.hdr` の環境光のもと、Disney BRDF 球を 11 列 × 10 段で並べる。各段で `subsurface / metallic / specular / specularTint / roughness / anisotropic / sheen / sheenTint / clearcoat / clearcoatGloss` のいずれか 1 つを `0.0` → `1.0` で振る。 |
| `26` | `assets/sky/kloofendal_48d_partly_cloudy_puresky_4k.hdr` の puresky 環境光のもと、同じドラゴンモデルと dragon 用 texture を使い、左に SimplePBR、右に Disney BRDF (clearcoat `0.4`、clearcoatGloss `0.9`) を並べる。 |
| `27` | `assets/sky/studio_small_08_4k.hdr` 環境下に、`baseColor = (0.5, 0.15, 0.05)` の Disney BRDF 球 (sheen `0`、sheenTint `0`、specular `0`) を 1 つ配置。 |
| `28` | `assets/sky/studio_small_08_4k.hdr` 環境下に、`baseColor = (0.5, 0.15, 0.05)` の Disney BRDF 球 (sheen `1`、sheenTint `0`、specular `0`) を 1 つ配置。 |
| `29` | puresky 環境光のもと、同じドラゴンモデルと dragon 用 texture を使い、左に SimplePBR、右に Autodesk Standard Surface を並べる。 |
| `30` | `assets/mori-knob/` の floor / base / knob を 3 × 3 グリッドに並べ、knob だけ Standard Surface のバリエーション (polished gold、iridescent metal、brushed copper、non-dispersive glass、smooth dispersive glass、rough dispersive glass、red velvet sheen、coated plastic、matte ceramic) を割り当てる。`assets/mori-knob/light.obj` を Emissive ライトとして使い、`assets/sky/studio_small_08_4k.hdr` の IBL を併用。 |
| `31` | puresky 環境光のもと、`assets/models/paper-plane.obj` の紙飛行機を thin_walled Standard Surface (`subsurface = 0`) として配置。 |
| `32` | puresky 環境光のもと、`assets/models/paper-plane.obj` の紙飛行機を thin_walled Standard Surface (`subsurface = 0.5`) として配置。 |
| `33` | `assets/mori-knob/floor.obj` の床の上に銀色 (`F0 = 0.92`) Conductor GGX (single-scattering) 球を 9 個並べ、roughness を左から右へ `0.0` → `1.0` で振る。`assets/sky/studio_small_08_4k.hdr` を IBL として使う。 |
| `34` | scene 33 と同じ床 / 配置 / IBL のもと、マテリアルを Cui et al. 2023 multi-scattering Conductor GGX に差し替える。 |
| `35` | 一様な白い環境光 (intensity `1.0`) のもと、`F0 = (1, 1, 1)` の球を 9 列 × 2 段で並べる。上段が single-scattering Conductor GGX、下段が Cui 2023 multi-scattering Conductor GGX で、各段とも roughness を左から右へ `0.0` → `1.0` で振る。 |
| `36` | `assets/sky/brown_photostudio_02_4k.hdr` の IBL のもと、9 列 × 2 段で上段に銀色 (`F0 = 0.92`) Conductor GGX (compensation OFF)、下段に白 Dielectric GGX (`eta = 1.5`、compensation OFF) を並べ、各段とも roughness を左から右へ `0.0` → `1.0` で振る。 |
| `37` | scene 36 と同じ配置 / IBL のもと、上下段とも `with_energy_compensation()` を有効にした Kulla & Conty 2017 multi-scattering 版。 |
| `38` | 一様な白い環境光 (intensity `1.0`) のもと、`F0 = (1, 1, 1)` / `color = (1, 1, 1)` の球を 9 列 × 4 段で並べる。上から SS Conductor、SS Dielectric (`eta = 1.5`)、MS Conductor、MS Dielectric の順、各段とも roughness を左から右へ `0.0` → `1.0` で振る。 |
| `39` | `assets/sky/brown_photostudio_02_4k.hdr` の IBL のもと、sRGB ゴールド (`F0 = (1.00, 0.78, 0.34)`) の Conductor GGX 球を 9 列 × 2 段で並べる。上段が compensation OFF、下段が `with_energy_compensation()` 有効、各段とも roughness を左から右へ `0.0` → `1.0` で振る。 |

未定義のシーン番号を指定した場合は `scene 0` が読み込まれます。

## 現在の実装上の注意

- `--width` と `--height` を省略した場合は、既定値として `512 x 512` の画像を出力します。
- `--integrator` を省略した場合は `mis` が選択されます。
- `-i mis` を指定すると、BSDF サンプリングとエリアライトの明示サンプリングを MIS で合成する integrator が選択されます。
- `-i nee` を指定すると、エリアライトの明示サンプリングを使う next event estimation integrator が選択されます。
- 生成画像は `result/` 以下に保存する運用を想定しています。
- 初回の `cargo run` では依存クレートのビルドに時間がかかることがあります。
