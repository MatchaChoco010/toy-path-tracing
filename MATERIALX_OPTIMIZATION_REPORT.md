# MaterialX Rendering Optimization Summary

## 概要

scene 45 は MaterialX を含むシーンで、初期状態では scene 41 に比べて約4.5倍遅かった。

| 時点 | scene 45 | scene 41 | 比率 |
| --- | ---: | ---: | ---: |
| baseline | 30.742s | 6.855s | 4.48x |
| 現在 | 11.327s | 6.649s | 1.70x |

合計では scene 45 の 512x512 / 128spp render 時間が `30.742s -> 11.327s` まで短縮された。

以下は、ログ上の連続測定差分をもとに、インパクトの大きかった最適化を大きい順に並べたもの。

## 1. MaterialX Sheen directional albedo を LUT 化

- 該当: opt7
- 変化: `26.633s -> 22.261s`
- 改善: `4.372s`, 約16.4%

最も効いた最適化。

MaterialX の Sheen directional albedo が light-tree precompute 中で毎回積分されており、ここが非常に重かった。既存の scene-owned `SheenDirectionalAlbedoLut` を MaterialX からも使うようにして、shading vertex ごとの多サンプル積分を lookup に置き換えた。

これは速度改善としても大きいが、directional albedo を適切に扱う設計への移行でもある。

## 2. closure parameter の compile-time constant 化

- 該当: opt50
- 変化: `14.926s -> 11.932s` 程度
- 改善: 約`2.99s`, 約20.1%

2番目に大きく効いた最適化。

closure node の parameter が compile-time constant なのに `ParamRef::Local` のまま残っており、BSDF traversal のたびに bytecode register を読んでいた。`local_constants()` で `LoadConst` / `Convert` / `Extract` / `Luminance` / 非matrix `Arith` などを追跡し、closure parameter を literal `ParamRef` に置き換えた。

bytecode実行器自体を速くするよりも、closure traversal から bytecode register 参照を消す方がこのシーンでは大きく効いた。

## 3. closure simplification と bytecode DCE

- 該当: opt9
- 変化: `22.261s -> 19.552s`
- 改善: `2.709s`, 約12.2%

MaterialX graph flatten 後に、closure tree に不要な `Zero`、無意味な `Add` / `Mix` / `Layer` などが残っていた。これを compile-time に簡約し、不要になった bytecode を DCE した。

profile でも bytecode instruction 数が大きく減っており、MaterialX evaluator の実行回数そのものを減らしたのが効いた。

## 4. opacity-only bytecode for alpha test

- 該当: opt29 / opt36
- 変化: `17.637s -> 15.684s`
- 改善: `1.953s`, 約11.1%

alpha test のために full material bytecode を評価していたのが重かった。

opacity 判定だけに必要な命令列を別 bytecode として作り、透明判定では full material evaluation を避けるようにした。opt36 後の profile では bytecode instruction 数がほぼ半減しており、実行回数削減として非常に効いた。

## 5. light-tree Layer precompute の重複削減

- 該当: opt6
- 変化: `28.338s -> 26.633s`
- 改善: `1.705s`, 約6.0%

light-tree precompute で Layer closure の directional albedo / energy 評価が重複していた。

Layer 上下の lobe energy をまとめて扱い、light-tree 用の material summary を作る過程の重複評価を減らした。scene 45 は LightTree が支配的なので、この経路の削減がそのまま効いた。

## 6. sample 後の eval/pdf 共有

- 該当: opt2, opt4, opt10
- 合計改善: 約`3.1s`

内訳:

| opt | 内容 | 変化 | 改善 |
| --- | --- | ---: | ---: |
| opt2 | MaterialX sampling 後の eval/pdf 統合 | `30.742s -> 29.681s` | `1.061s` |
| opt4 | leaf lobe の eval/pdf をまとめて評価 | `29.681s -> 28.650s` | `1.031s` |
| opt10 | direct-light MIS で combined eval/pdf を使う | `19.552s -> 18.505s` | `1.047s` |

MaterialX closure tree を `eval` と `pdf` で別々に辿っていた箇所を、`eval_pdf` として一回の traversal にまとめた。

1つずつは約1秒級だが、同じ構造問題を複数経路で潰したため合計インパクトは大きい。

## 7. MaterialX directional albedo cache per shading vertex

- 該当: opt18
- 変化: `18.620s -> 17.977s`
- 改善: `0.643s`, 約3.5%

Directional Albedo は LUT 化後も呼び出し回数が多かった。

同じ shading vertex 内で `le` / `pdf` / `eval` / `eval_pdf` / light-tree precompute などから同じ closure の directional albedo が繰り返し必要になるため、scratch に cache を持たせて共有した。

このシーンでは劇的ではないが、論理的に重複評価を削る構造改善なので採用した。

## 8. pure uniform bytecode constant folding

- 該当: opt65
- 変化: `11.997s -> 11.327s`
- 改善: `0.670s`, 約5.6%

`local_constants()` による補助解析だけでなく、compile pipeline上の明示的なconstant folding passとしてpure命令を `LoadConst` に置き換えるようにした。

profileではbytecode命令数が約 `1.164B -> 0.939B` に減った。scene 45では `Image` はほぼそのままだが、`Arith` / `MixValue` / `RoughnessAnisotropy` / `Clamp` などのuniform計算が削減され、通常renderでも約0.67秒改善した。

速度改善だけでなく、uniform式の簡約が「closure param inline用の副作用」ではなく、独立したcompile passとして見えるようになった点も採用理由。

## 9. constant-one opacity fold と opaque alpha test skip

- 該当: opt33 / opt44
- 変化: `15.684s -> 15.230s`
- 改善: `0.454s`, 約2.9%

MaterialX material の opacity が local constant 経由で 1.0 になる場合でも、alpha test 用 bytecode が残っていた。

constant opacity を compile-time に認識し、完全opaqueな material では alpha test 自体を省略するようにした。opacity-only bytecode の後に残った無駄をさらに削った形。

## 10. MtlxScratch lifetime の修正

- 該当: opt5
- 変化: `28.650s -> 28.338s`
- 改善: `0.312s`, 約1.1%

`MtlxScratch` の lifetime を camera trace 単位で再利用する形に戻し、余計な確保や巻き戻しを減らした。

改善幅は大きくないが、scratch の lifetime として自然な構造に戻す修正だったため採用。

## 11. Arith evaluator dispatch の整理

- 該当: opt45
- 変化: `15.230s -> 14.926s`
- 改善: `0.304s`, 約2.0%

`Arith` 命令が profile 上で大きなhotspotだった。

closure-based scalar evaluator をやめ、`arith_scalar` / `arith_vec2` / `arith_vec3` / `arith_vec4` に分け、vector演算は glam の vector op に寄せた。bytecode VM の命令数を減らす変更ではないため効果は限定的だが、1%超の改善があった。

## 12. texture power-of-two wrapping

- 該当: opt21
- 変化: `17.977s -> 17.720s`
- 改善: `0.257s`, 約1.4%

MaterialX texture sampling は明確なhotspotだったが、sampler側の大きな改善は難しかった。

その中で、power-of-two texture の wrap を `rem_euclid` ではなく bitmask にする局所最適化だけは小幅に効いた。実装が低複雑性なので保持。

## 13. geometric load CSE

- 該当: opt25
- 変化: `17.720s -> 17.637s`
- 改善: `0.083s`, 約0.47%

通常render時間だけを見ると新基準では微小。

ただし profile 上では `LoadGeom` calls が `396M -> 219M` に減っており、bytecode命令数削減としては明確。runtime hot path に分岐を増やさず、compile時に同じ geometric input を同じ virtual register に束ねるだけなので保持した。

## 効果が小さく、revert または不採用にしたもの

## single-lobe sample fast path

- 該当: opt13
- ログ上の改善: 約0.55%
- 監査後判断: revert

`sample_needs_eval_pdf` field、compile walker、runtime branch を追加していたが、複数回測定では差が1%未満で分散内だった。

新基準では、0.1s程度の差のために compile/runtime 構造を複雑にする価値がないためrevertした。

## bytecode実行器への細かい分岐追加

- 該当: opt49
- 判断: 不採用

`ArithOp::Multiply` / `Add` / `Max` / `Power` を `execute_instruction()` の match arm で専用化したが、逆に悪化した。

巨大matchに細かい分岐を増やすことで i-cache や分岐予測が悪化した可能性が高い。

## Image sampler fast path 系

- 該当: opt27, opt45付近の periodic image fast path
- 判断: 不採用

非UDIM periodic image の default処理を省略するなどの sampler fast path を試したが悪化した。

MaterialX image命令はhotspotだが、分岐追加やhelper分割でhot pathが重くなりやすく、単純なfast path追加では改善しなかった。

## LoadConst operand inline

- 該当: opt47
- 判断: 不採用

`LoadConst` register を使う代わりに hot opcode の operand を `Operand::Const` に差し替える実験。

命令数は減らせても、value_pool 読みが増えて register slot 読みより重くなった可能性が高く、通常renderでは悪化した。

## まとめ

大きく効いたのは、細かいマイクロ最適化ではなく以下の3系統だった。

1. 重い数値積分を LUT / cache に置き換える。
   - Sheen directional albedo LUT
   - directional albedo cache

2. MaterialX evaluator の呼び出し回数・bytecode命令数を減らす。
   - closure simplification
   - bytecode DCE
   - opacity-only bytecode
   - constant opacity fold
   - closure constant parameter 化

3. 同じ closure traversal を別目的で何度も走らせない。
   - sample後の eval/pdf 共有
   - leaf lobe eval/pdf共有
   - direct-light MIS の combined eval/pdf
   - light-tree Layer precompute の重複削減

逆に、bytecode VM のごく細かい分岐最適化や sampler fast path は、効果が小さいか悪化しやすかった。

今回の最終的な改善は、MaterialX の評価器を少しずつ速くしたというより、MaterialX 評価を「そもそも何度も走らせない」「不要なbytecodeを作らない」「重い積分をruntimeに残さない」方向の構造改善が支配的だった。
