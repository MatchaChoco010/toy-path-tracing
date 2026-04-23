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
| `10` | White furnace test 用シーン。`1.0` の一様な白い環境光のもと、上段に roughness を `0.0 / 0.15 / 0.3 / 0.45 / 0.6 / 0.75` と並べた Dielectric GGX ガラス球、下段に同じ roughness 列の銀色 Conductor GGX 金属球を 6 個ずつ配置し、エネルギー保存をチェックします。 |
| `11` | 広い Lambert 床の上に薄青の Lambert バニーを置き、`DirectionalLight` (太陽相当の平行光) 1 つだけで照らす delta 光源の基本確認シーンです。 |
| `12` | Cornell box 風の部屋からエリアライトを外し、薄青 Lambert バニーとラフな金色 / 銅色の Conductor GGX 球を配置。暖色と寒色の `PointLight` を 2 灯、バニーに向けたマゼンタ / ティールの `SpotLight` を 2 灯で照らす delta 光源ミックスの確認シーンです。 |
| `13` | 広い Lambert 床の上に少し大きめの diffuse バニーを置き、`assets/sky/brown_photostudio_02_4k.hdr` の環境光だけで照らす SkyLight 比較用シーンです。 |
| `14` | scene 13 と同じ床 / バニー構成で、`assets/sky/kloofendal_48d_partly_cloudy_puresky_4k.hdr` の環境光だけで照らす SkyLight 比較用シーンです。 |

未定義のシーン番号を指定した場合は `scene 0` が読み込まれます。

## 現在の実装上の注意

- `--width` と `--height` を省略した場合は、既定値として `512 x 512` の画像を出力します。
- `--integrator` を省略した場合は `mis` が選択されます。
- `-i mis` を指定すると、BSDF サンプリングとエリアライトの明示サンプリングを MIS で合成する integrator が選択されます。
- `-i nee` を指定すると、エリアライトの明示サンプリングを使う next event estimation integrator が選択されます。
- 生成画像は `result/` 以下に保存する運用を想定しています。
- 初回の `cargo run` では依存クレートのビルドに時間がかかることがあります。
