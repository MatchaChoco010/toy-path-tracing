# toy-path-tracing

Rust で学習用のパストレーサーを実装していくためのプロジェクトです。

## 使い方

基本的な実行方法は次のとおりです。

```bash
cargo run --release -- [OPTIONS]
```

現在の CLI では、出力先、出力画像サイズ、シーン番号、1 ピクセルあたりのサンプル数、最大パストレース深度、使用する integrator を指定できます。integrator を省略した場合は、MIS を使う `mis` が既定で選ばれます。

### 実行例

`scene 0` を `64 spp` でレンダリングして `result/scene-0.png` に保存する例です。

```bash
cargo run --release -- --spp 64 --scene 0 -o result/scene-0.png
```

integrator を明示して実行したい場合は `--integrator` または `-i` を使います。現状は `mis`、`pt`、`nee` を選べます。

```bash
cargo run --release -- --scene 1 --spp 128 --depth 24 -i mis -o result/scene-1-mis.png
```

```bash
cargo run --release -- --scene 1 --spp 128 --depth 24 -i pt -o result/scene-1-pt.png
```

```bash
cargo run --release -- --scene 1 --spp 128 --depth 24 -i nee -o result/scene-1-nee.png
```

完全鏡面反射の確認用シーンは `scene 2` で実行できます。

```bash
cargo run --release -- --scene 2 --spp 256 --depth 24 -i mis -o result/scene-2-mirror.png
```

ガラス材質の確認用シーンは `scene 3` で実行できます。

```bash
cargo run --release -- --scene 3 --spp 512 --depth 32 -i mis -o result/scene-3-glass.png
```

Conductor GGX の roughness 差確認用シーンは `scene 4` で実行できます。

```bash
cargo run --release -- --scene 4 --spp 512 --depth 32 -i mis -o result/scene-4-conductor-ggx.png
```

Conductor GGX の anisotropy 差確認用シーンは `scene 5` で実行できます。

```bash
cargo run --release -- --scene 5 --spp 512 --depth 32 -i mis -o result/scene-5-conductor-ggx-anisotropy.png
```

Dielectric GGX の roughness 差確認用シーンは `scene 6` で実行できます。

```bash
cargo run --release -- --scene 6 --spp 1024 --depth 32 -i mis -o result/scene-6-dielectric-ggx.png
```

Dielectric GGX の anisotropy 差確認用シーンは `scene 7` で実行できます。

```bash
cargo run --release -- --scene 7 --spp 1024 --depth 32 -i mis -o result/scene-7-dielectric-ggx-anisotropy.png
```

解像度も含めて指定したい場合は `--width` と `--height` を使います。

```bash
cargo run --release -- --scene 1 --width 1280 --height 720 --spp 128 --depth 24 -o result/scene-1.png
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
| `--env-scale <ENV_SCALE>` | `1.0` | 環境光 (HDRI) の強度倍率です。シーンが環境光を持っている場合、ロード時の scale にこの値を乗算します。環境光が無いシーンでは効果はありません。 |
| `-h, --help` | なし | ヘルプを表示します。 |

## 現在のシーン番号

現状のソースコードでは、次のシーンが実装されています。

| シーン番号 | 内容 |
| --- | --- |
| `0` | Cornell box 風の部屋に、箱とバニーを配置したシーンです。 |
| `1` | Cornell box 風の部屋に、バニーと 2 つの球を配置したシーンです。 |
| `2` | Cornell box 風の部屋に、完全鏡面の銀色バニーと金色の球を配置したシーンです。 |
| `3` | Cornell box 風の部屋に、手前の透明ガラス球、左右の thin / 通常の水色ガラスバニー、球越しの歪み確認用の薄い青の Lambert バニーを配置した確認用シーンです。 |
| `4` | Cornell box 風の部屋に、roughness を左から `0.0 / 0.25 / 0.5 / 0.75 / 1.0` にした金色の Conductor GGX 球を 5 つ並べた確認用シーンです。 |
| `5` | Cornell box 風の部屋に、roughness `0.3` の銀色 Conductor GGX 球を 3 つ並べ、中央を isotropic、左右を `anisotropy = -1.0 / +1.0` の異方性違いにした確認用シーンです。 |
| `6` | Cornell box 風の部屋に、roughness を左から `0.0 / 0.15 / 0.3 / 0.45 / 0.6` にした透明な Dielectric GGX ガラス球を 5 つ、少しだけ宙に浮かせて並べ、地面への影と集光模様が見えるようにした確認用シーンです。 |
| `7` | Cornell box 風の部屋に、roughness `0.3` の薄水色 Dielectric GGX 球を 3 つ並べ、中央を isotropic、左右を `anisotropy = -1.0 / +1.0` の異方性違いにした確認用シーンです。 |
| `8` | 広い Lambert 床の上に、roughness を左から `0.0 / 0.15 / 0.3 / 0.45 / 0.6 / 0.75` にした Conductor GGX 金属球を 6 つ並べ、その上段に同じ roughness 列の Dielectric GGX ガラス球を 6 つ並べ、`assets/sky/` の HDRI を IBL として読み込む屋外風シーンです。 |
| `9` | Cornell box の中央に大きめのラフな金色 Conductor GGX (`roughness = 0.35`) のバニーを置き、天井のエリアライトと `assets/sky/` の HDRI を IBL として併用する確認用シーンです。カメラはボックスの外からやや引いた位置にあり、ボックス外周の Sky も写り込みます。 |

補足:
`load_scene()` の現在の実装では、`1` を指定すると `scene_1`、`2` を指定すると `scene_2`、`3` を指定すると `scene_3`、`4` を指定すると `scene_4`、`5` を指定すると `scene_5`、`6` を指定すると `scene_6`、`7` を指定すると `scene_7` を読み込み、それ以外はすべて `scene_0` を読み込みます。

## 現在の実装上の注意

- `--width` と `--height` を省略した場合は、既定値として `512 x 512` の画像を出力します。
- `--integrator` を省略した場合は `mis` が選択されます。
- `-i mis` を指定すると、BSDF サンプリングとエリアライトの明示サンプリングを MIS で合成する integrator が選択されます。
- `-i nee` を指定すると、エリアライトの明示サンプリングを使う next event estimation integrator が選択されます。
- 生成画像は `result/` 以下に保存する運用を想定しています。
- 初回の `cargo run` では依存クレートのビルドに時間がかかることがあります。
