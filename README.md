# toy-path-tracing

Rust で学習用のパストレーサーを実装していくためのプロジェクトです。

## 使い方

基本的な実行方法は次のとおりです。

```bash
cargo run --release -- [OPTIONS]
```

現在の CLI では、出力先、出力画像サイズ、シーン番号、1 ピクセルあたりのサンプル数、最大パストレース深度、使用する integrator を指定できます。

### 実行例

`scene 0` を `64 spp` でレンダリングして `result/scene-0.png` に保存する例です。

```bash
cargo run --release -- --spp 64 --scene 0 -o result/scene-0.png
```

integrator を明示して実行したい場合は `--integrator` または `-i` を使います。現状は `pt` と `nee` を選べます。

```bash
cargo run --release -- --scene 1 --spp 128 --depth 24 -i pt -o result/scene-1-pt.png
```

```bash
cargo run --release -- --scene 1 --spp 128 --depth 24 -i nee -o result/scene-1-nee.png
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
| `-i, --integrator <INTEGRATOR>` | `pt` | 使用する integrator を指定します。現在は `pt` と `nee` を選べます。存在しない名前を指定するとエラーになります。 |
| `-h, --help` | なし | ヘルプを表示します。 |

## 現在のシーン番号

現状のソースコードでは、次のシーンが実装されています。

| シーン番号 | 内容 |
| --- | --- |
| `0` | Cornell box 風の部屋に、箱とバニーを配置したシーンです。 |
| `1` | Cornell box 風の部屋に、バニーと 2 つの球を配置したシーンです。 |

補足:
`load_scene()` の現在の実装では、`1` を指定したときだけ `scene_1` を読み込み、それ以外はすべて `scene_0` を読み込みます。

## 現在の実装上の注意

- `--width` と `--height` を省略した場合は、既定値として `512 x 512` の画像を出力します。
- `--integrator` を省略した場合は `pt` が選択されます。
- `-i nee` を指定すると、エリアライトの明示サンプリングを使う next event estimation integrator が選択されます。
- 生成画像は `result/` 以下に保存する運用を想定しています。
- 初回の `cargo run` では依存クレートのビルドに時間がかかることがあります。
