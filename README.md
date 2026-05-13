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
| `8` | Lambert 床に Conductor GGX 金属球と Dielectric GGX ガラス球を 2 段に並べ、各段とも roughness をスイープ。HDRI 環境光。 |
| `9` | Cornell box の中央に金色 Conductor GGX バニー。エリアライトと HDRI 環境光を併用、カメラはボックスの外。 |
| `10` | 一様な白い環境光のもと、Dielectric GGX ガラス球と銀色 Conductor GGX 金属球を 2 段に並べ、各段とも roughness をスイープ。 |
| `11` | Lambert 床に薄青 Lambert バニーを置き、DirectionalLight 1 つで照らす。 |
| `12` | Cornell box 風の部屋に薄青 Lambert バニーと金色 / 銅色の Conductor GGX 球を配置、PointLight と SpotLight で照らす。 |
| `13` | Lambert 床に diffuse バニー、HDRI 環境光のみ。 |
| `14` | Lambert 床に diffuse バニー、puresky HDRI 環境光のみ。 |
| `15` | Cornell box 風の部屋にテクスチャ付き Conductor GGX 球と Lambert 球を斜めに並べる。 |
| `16` | Lambert 床と puresky 環境光のもと、テクスチャ付き Lambert バニーと Conductor GGX 球を配置。 |
| `17` | テクスチャ付き Lambert 床に完全鏡面の金属球とガラス球、puresky HDRI で照らす。 |
| `18` | Cornell box 風の部屋にノーマルマップ付き Lambert 球と Conductor GGX 球を斜めに配置。 |
| `19` | Lambert 床と puresky 環境光のもと、ノーマルマップ付き Lambert 球、Conductor GGX 球、Mirror 球を配置。 |
| `20` | Lambert 床と puresky 環境光のもと、テクスチャ付き SimplePBR ドラゴンを配置。 |
| `21` | Lambert 床と puresky 環境光のもと、SimplePBR ドラゴン、金色 Conductor GGX バニー、Glass 球、NormalizedLambert バニーを並べる。 |
| `22` | ローポリ版 San Miguel を puresky HDRI で照らす。OBJ マテリアルは SimplePBR / DielectricGgx / Emissive に振り分け。 |
| `23` | 通常版 San Miguel、配置とマテリアル割り当ては scene 22 と同じ。 |
| `24` | Amazon Lumberyard Bistro (Exterior + Interior)、シーン中の emissive ポリゴンだけで照らす。 |
| `25` | HDRI 環境光のもと、Disney BRDF 球を 11 列 × 10 段で並べ、各段で各パラメータをスイープ。 |
| `26` | puresky 環境光のもと、同じドラゴンモデルを左 SimplePBR / 右 Disney BRDF で並べる。 |
| `27` | HDRI 環境下に sheen=0 の Disney BRDF 球を配置。 |
| `28` | HDRI 環境下に sheen=1 の Disney BRDF 球を配置。 |
| `29` | puresky 環境光のもと、同じドラゴンモデルを左 SimplePBR / 右 Standard Surface で並べる。 |
| `30` | mori-knob を 3 × 3 グリッドに配置し、knob だけ Standard Surface のバリエーション (polished gold、iridescent metal、brushed copper、non-dispersive glass、smooth/rough dispersive glass、red velvet sheen、coated plastic、matte ceramic) を割り当てる。 |
| `31` | puresky 環境光のもと、紙飛行機を subsurface=0 の thin_walled Standard Surface として配置。 |
| `32` | puresky 環境光のもと、紙飛行機を subsurface=0.5 の thin_walled Standard Surface として配置。 |
| `33` | 床の上に銀色 single-scattering Conductor GGX 球を 9 個並べ、roughness をスイープ。 |
| `34` | scene 33 と同じ配置、マテリアルを Cui 2023 multi-scattering Conductor GGX に差し替え。 |
| `35` | 一様な白い環境光のもと、SS Conductor / MS Conductor を 9 列 × 2 段で並べ、roughness をスイープ。 |
| `36` | HDRI のもと、Conductor GGX (compensation OFF) と Dielectric GGX (compensation OFF) を 9 列 × 2 段で並べ、roughness をスイープ。 |
| `37` | scene 36 と同じ配置、Kulla & Conty 2017 energy compensation を有効にした版。 |
| `38` | 一様な白い環境光のもと、SS Conductor / SS Dielectric / MS Conductor / MS Dielectric を 9 列 × 4 段で並べ、roughness をスイープ。 |
| `39` | HDRI のもと、ゴールドの Conductor GGX 球を 9 列 × 2 段で並べ、上段 compensation OFF / 下段 ON、roughness をスイープ。 |
| `40` | mori-knob 風の白い床に Glavenus STL モデルの 9 パーツを配置、 NormalizedLambert を割り当て。 |
| `41` | scene 40 と同じ配置、 マテリアルを EON (energy-preserving Oren-Nayar) に差し替え。 |
| `45` | mori-knob を 4×4 グリッドに配置し、各ノブに別々の MaterialX マテリアルを割り当てる。 |

未定義のシーン番号を指定した場合は `scene 0` が読み込まれます。

## MaterialX サポート

MaterialX 1.39.4 仕様の Volume を除くサブセットを `src/scene_loader/mtlx_loader/` 以下にロード機構として実装しています。`.mtlx` ファイルを読み込み、`<surfacematerial>` を `Material::Mtlx` バリアントとして本リポジトリの BSDF / EDF / Light tree 経路に統合できます。

### 取り込み済みのライブラリ

`lib/materialx/libraries/` 以下に以下を vendor しています (Apache License 2.0、`lib/materialx/LICENSE` および `NOTICE` 参照)。

- `stdlib/stdlib_defs.mtlx`, `stdlib/stdlib_ng.mtlx`
- `pbrlib/pbrlib_defs.mtlx`, `pbrlib/pbrlib_ng.mtlx`
- `bxdf/standard_surface.mtlx`, `disney_principled.mtlx`, `open_pbr_surface.mtlx`, `usd_preview_surface.mtlx`, `gltf_pbr.mtlx`
- `nprlib/nprlib_defs.mtlx`, `nprlib/nprlib_ng.mtlx`

### 対応している主なノード

- **BSDF**: `oren_nayar_diffuse_bsdf`, `burley_diffuse_bsdf`, `translucent_bsdf`, `dielectric_bsdf`, `conductor_bsdf`, `generalized_schlick_bsdf`, `sheen_bsdf` (`conty_kulla` / `zeltner` モード), `chiang_hair_bsdf`。
- **薄膜干渉**: `dielectric_bsdf` / `conductor_bsdf` / `generalized_schlick_bsdf` の `thinfilm_thickness` / `thinfilm_ior` 入力に対応 (Belcour & Barla 2017 ベースで既存 `bsdf::thin_film` を再利用)。`layer(thin_film_bsdf, base)` 形式で書かれた場合も、薄膜パラメータが下層 BSDF の Fresnel に正しく伝搬する。
- **未対応**: `subsurface_bsdf` は warning を出して `burley_diffuse_bsdf` に近似フォールバック。
- **EDF**: `uniform_edf`, `conical_edf`, `generalized_schlick_edf`。
- **Combinators**: `mix`, `layer`, `add`, `multiply`。
- **Pattern**: 数学・logical・channel・color (luminance / rgbtohsv / hsvtorgb / hsvadjust)・geometric (`position` / `normal` / `texcoord` / 等)・procedural (`noise2d/3d`, `fractal2d/3d` (fBm), `cellnoise2d/3d`, `worleynoise2d/3d`, `randomfloat`, `randomcolor`)・`blackbody`・`artistic_ior`・`roughness_anisotropy`・`image`。NG 実装のあるノード (`tiledimage`, `latlongimage`, `unifiednoise2d/3d`, `checkerboard`, `place2d`, `colorcorrect`, `range`, `contrast`, `saturate` ほか) は flatten で自動展開されます。
- **NPR**: `viewdirection`, `facingratio`, `gooch_shade` (`Material::le()` 経由で view 依存の outgoing radiance を返し、間接光は受けない扱い)。
- **標準シェーダ**: `standard_surface`, `disney_principled`, `open_pbr_surface`, `usd_preview_surface`, `gltf_pbr`。これらは vendor された mtlx の nodegraph 実装を flatten して評価します。

### 制限事項

- Volume / VDF 系 (`absorption_vdf`, `anisotropic_vdf`, `volume`)、`subsurface_bsdf` の正確な MFP 評価、`measured_edf` (IES プロファイル) は未対応。
- mtlx `<light>` / `lightshader` 型による明示的な光源エンティティは未対応 (本プロジェクトの既存 light system を使ってください)。
- `<look>`, `<materialassign>`, `<collection>`, `<visibility>`, `<propertyset>`, `<variantset>` による高度な割り当ては未対応。
- UDIM / UVTILE のフル展開は未対応。
- ShaderGen 互換 (`genglsl/genosl/genmdl/genmsl`) は対象外 (本実装は独自評価器)。
- 第一弾の color management は `srgb_texture`, `lin_rec709`, `g22_rec709`, `none` をサポート。document 既定 `colorspace` は `lin_rec709` を仮定。

### opacity (cutout) 使用時の規約

`opacity < 1` を含む MaterialX マテリアル (alpha cutout) は **必ず `thin_walled = true` で使用してください**。 closed mesh + cutout + 非 thin-walled の組み合わせは光学的にトポロジ整合性が取れず (光線が cutout で通り抜けた先で同じメッシュの裏面に当たり、 「現在媒質は air か glass か」 が path-history dependent になる)、 dielectric の back-face Fresnel が誤って TIR を起こすなど非物理的な結果を生みます。

これは OpenPBR Surface Specification ([academysoftwarefoundation.github.io/OpenPBR](https://academysoftwarefoundation.github.io/OpenPBR/)) が「non-thin-walled で α < 1 は厳密な物理的意味を持たない」 「fractional opacity は thin-walled モードでのみ明確な意味を持つ ('alpha blend' として)」 と spec レベルで明記している方針に揃えています。

`standard_surface` ノードに `<input name="thin_walled" type="boolean" value="true" />` を追加してください。 vendor された `lib/materialx/libraries/bxdf/standard_surface.mtlx` は、 上層の `<surface>` ノードへ `thin_walled` 入力を伝搬するように修正済みです (`assets/mtlx/shader_ops.mtlx` の運用例を参照)。

## 現在の実装上の注意

- `--width` と `--height` を省略した場合は、既定値として `512 x 512` の画像を出力します。
- `--integrator` を省略した場合は `mis` が選択されます。
- `-i mis` を指定すると、BSDF サンプリングとエリアライトの明示サンプリングを MIS で合成する integrator が選択されます。
- `-i nee` を指定すると、エリアライトの明示サンプリングを使う next event estimation integrator が選択されます。
- 生成画像は `result/` 以下に保存する運用を想定しています。
- 初回の `cargo run` では依存クレートのビルドに時間がかかることがあります。
