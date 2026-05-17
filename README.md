# toy-path-tracing

Rust で学習用のパストレーサーを実装していくための workspace です。

現在の実行対象は `crates/renderer` です。workspace root からこれまで通り次のコマンドで renderer を起動できます。

```bash
cargo run --release -- [OPTIONS]
```

renderer の CLI 引数、実行例、MaterialX サポートの詳細は `crates/renderer/README.md` を参照してください。
