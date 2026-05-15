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
