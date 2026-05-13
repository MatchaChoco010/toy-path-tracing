# MaterialX material evaluation optimization log

作業開始日: 2026-05-13

## 方針

- レンダリング時間は `time` コマンドではなく、実行ログの `render:` のみを比較する。
- 画像と計測用出力は `result/perf/` 以下に保存する。
- まずコード変更前の scene 41 / scene 45 のレンダリング時間と、scene 45 の参照画像を残す。
- 調査用の一時変更は、目的、測定結果、採否、戻したかどうかを同じセクション内に書く。
- profile の thread累計時間は構造把握用。採否は通常実行の `render:` を優先する。

## 1. Baseline

コード変更前の初期計測。

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/baseline_scene41_512_spp128.png` | 00m:06s:855ms |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/baseline_scene45_512_spp128.png` | 00m:30s:742ms |
| 45 | 1024x1024, 512spp, integrator=mis, depth=16 | `result/perf/reference_scene45_1024_spp512.png` | 08m:16s:917ms |

scene 45 / scene 41 = 4.49x。ユーザー報告の 4-5 倍差と一致。

参照画像:

- `result/perf/baseline_scene45_512_spp128.png`
- `result/perf/reference_scene45_1024_spp512.png`

## 2. Baseline profile

実行:

`cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --profile-mtlx -o result/perf/profile_baseline_scene45_512_spp128.png`

結果:

- render: 00m:39s:287ms
- bytecode: 132468ms / 103,664,989 calls / 6,941,110,489 instrs
- sample: 40311ms / 29,118,822 calls
- eval: 145041ms / 41,580,388 calls
- pdf: 130320ms / 33,232,278 calls
- directional albedo profile counter: 0 calls

slot別の累計上位:

| slot | calls | total | per_call | avg_instrs |
| --- | ---: | ---: | ---: | ---: |
| 2 | 8,106,620 | 19611ms | 2419ns | 96 |
| 1 | 6,264,538 | 13240ms | 2114ns | 128 |
| 10 | 5,621,749 | 11927ms | 2122ns | 95 |
| 3 | 6,728,196 | 9769ms | 1452ns | 69 |
| 11 | 7,369,128 | 8130ms | 1103ns | 95 |

closure walk の visit 数が非常に大きい:

- `Mix`: eval 178,491,117 / pdf 143,209,862 / sample 65,166,890
- `Layer`: eval 132,042,903 / pdf 105,865,284 / sample 65,625,476
- `Dielectric`: eval 132,042,903 / pdf 105,865,284 / sample 6,684,039 / dalbedo 272,212,265
- `Surface` / `Multiply` / `OrenNayar` / `Conductor` も eval 44,014,301 / pdf 35,288,428

観察:

- wall time に対して thread累計の bytecode / eval / pdf が大きく、MaterialX 側が十分に支配的。
- bytecode calls が約1.04億回あり、MaterialX locals の評価が交差点ごとに大量に走っている。
- closure walker visit が呼び出し数に対してさらに膨らんでいる。標準surface由来の mix/layer ツリーを eval/pdf/sample で毎回再帰走査していることが効いている。
- directional albedo は top-level counter では 0 だが、closure visit の dalbedo は非常に多い。profileの取り方か呼び出し経路を確認する必要がある。

## 3. Optimization 1: directional albedo cache in shading scratch

調査:

- `src/material/mtlx/runtime.rs` の `Layer` は sample / eval / pdf / light_tree_precompute のたびに `directional_albedo_idx` を呼ぶ。
- top-level の `directional_albedo_closure()` は使われていないため `PROF_DALBEDO_CALLS` は 0 のままだが、内部再帰の `record_dalbedo_visit()` は走っている。
- scene 45 profile では dalbedo visit が 2.72億で、`Dielectric` / `Sheen` などの directional albedo が eval/pdf/sample の内側で繰り返されている。
- `wo` と MaterialX bytecode locals は shading vertex ごとに固定なので、Layer の top closure や non-BSDF Add の branch closure の directional albedo は同じ shading vertex 内で共有できる。

実装:

- `MtlxScratch` に directional albedo 用の `Vec<Vec3>` 相当のプール、valid flag、regs handle から cache handle への対応を追加した。
- 通常の `MtlxMaterial::precompute_shading()` では bytecode locals を作った後、Layer / non-BSDF Add を含む material だけ directional albedo cache を作る。
- alpha test の `any_hit()` は opacity だけに必要なので、directional albedo cache を作らない `precompute_shading_values()` を使う。
- sample / eval / pdf / light_tree_precompute は cache があれば `directional_albedo_idx` の代わりに cached value を読む。

確認:

- `cargo check`: OK

測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt1_scene45_512_spp128.png` | 01m:30s:036ms |

結果:

- baseline 00m:30s:742ms から大きく悪化。
- Layer / Add の target closure を precompute したが、shading vertex ごとに先回り計算する量が多く、実際に eval/pdf/sample で必要にならない branch や any-hit 以外の経路までコストを増やした。
- この最適化は不採用。`src/material/mtlx/mod.rs`, `src/material/mtlx/runtime.rs`, `src/material/mtlx_material.rs` の変更を戻した。

戻した後の確認:

- `cargo check`: OK

## 4. Optimization 2: combine eval and pdf after MaterialX sampling

調査:

- `MtlxMaterial::sample()` は `sample_closure()` で wi を選んだ後、同じ `wo/wi` に対して `eval_closure()` と `pdf_closure()` を独立に呼んでいる。
- scene 45 baseline profile では sample 29,118,822 calls に対して eval 41,580,388 calls / pdf 33,232,278 calls。sample 後の eval/pdf だけでも数千万回の closure traversal になる。
- Layer は eval と pdf の両方で同じ top directional albedo を計算するため、ここを1 traversalにまとめると少なくとも sample 後の重複分は削れる。

実装:

- `runtime::eval_pdf_closure()` を追加。
- Mix / Layer / Add / Multiply / branch / Surface の closure tree を1回だけ再帰し、leaf では既存の eval/pdf を呼ぶ。
- `MtlxMaterial::sample()` の非 thin-walled transmission 経路で、従来の `eval_closure()` + `pdf_closure()` を `eval_pdf_closure()` に置き換えた。

確認:

- `cargo check`: OK

測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt2_scene45_512_spp128.png` | 00m:29s:681ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt2_scene41_512_spp128.png` | 00m:06s:878ms |

結果:

- scene 45 は baseline 00m:30s:742ms から 1.061s 改善。約3.5%短縮。
- 同時点の scene 45 / scene 41 = 4.32x。
- 目標の 1.4x には全く届かないが、sample後の eval/pdf 重複を構造的に減らせているので採用候補として残す。

profile測定:

`cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --profile-mtlx -o result/perf/profile_opt2_scene45_512_spp128.png`

結果:

- render: 00m:35s:665ms
- bytecode: 133584ms / 103,673,910 calls / 6,941,176,405 instrs
- sample: 38219ms / 29,121,838 calls
- eval: 44482ms / 14,015,097 calls
- pdf: 20113ms / 5,663,544 calls

profiler確認で分かった問題:

- `eval_pdf_closure()` を追加したが、既存 profiler は `eval_closure()` / `pdf_closure()` の public entry だけを計測している。
- そのため opt2 の profile は、sample後に `eval_pdf_closure()` へ移った分を eval/pdf time として直接数えていない。
- `eval_pdf_closure_idx()` 内の leaf fallback は既存 `eval_closure_idx()` / `pdf_closure_idx()` を呼ぶため leaf visit は数えられるが、Mix / Layer / Surface / Multiply など combined traversal 側の visit は eval/pdf visit としては数えられない。
- 従って opt2 の profile 比較で eval/pdf calls や Mix/Layer visits が減ったことは「呼び出し構造が変わった」ことを示すが、「その分の実コストが消えた」とそのまま解釈してはいけない。
- render 時間 00m:29s:681ms は `println` の wall-clock なので採否判断には有効。

採否:

- 実レンダリング時間は改善しているため採用候補として残す。
- profiler の `eval_pdf_closure()` 経路を測れるように修正してから、次の分析に進む。

## 5. Profiler audit and fix

確認した構造:

- `--profile-mtlx` は `main.rs` で `runtime::set_profile_enabled(true)` を呼び、scene load / MaterialX compile 前から有効になる。
- bytecode profile は `run_instructions()` を `Instant::now()` で囲み、global atomic に `ns/calls/instrs` を加算する。per-material slot も同じ bytecode 実行だけを集計する。
- sample/eval/pdf profile は `sample_closure()` / `eval_closure()` / `pdf_closure()` の public entry を計測する。
- closure visit histogram は `eval_closure_idx()` / `pdf_closure_idx()` / `sample_closure_idx()` / `directional_albedo_idx()` の各再帰関数に入った回数を variant 別に数える。

正しく読めること:

- `render:` は通常実行の wall-clock で、最適化の採否に使える。
- profile の `bytecode` は MaterialX locals 評価の thread累計時間、call 数、命令dispatch総数として読める。
- per-material slot はどの MaterialX material の bytecode が重いかを見る用途では使える。
- closure visit histogram は再帰走査の構造的な増幅を見る用途では使える。

注意点 / 不備:

- profile の ns は rayon worker 全体の thread累計なので、`render:` wall-clock より大きくなる。これは異常ではない。
- `Instant::now()` と atomic 加算が非常に多く入るため、profile有効時の `render:` は通常実行より遅く、絶対時間比較には向かない。
- `PROF_DALBEDO_CALLS` は `directional_albedo_closure()` public entry だけを数えるが、実際の hot path は `Layer` 内の `directional_albedo_idx()` 直接呼び出しなので 0 のままになる。dalbedo の有無は `DALBEDO_VISITS` を見る必要がある。
- `PROF_PRE_DALBEDO_*` は現在実質使われていない。
- opt2 で追加した `eval_pdf_closure()` が profile対象外だったため、opt2 profile の eval/pdf time と visit histogram は測りたい内容を完全には測れていない。

修正:

- `PROF_EVAL_PDF_NS` / `PROF_EVAL_PDF_CALLS` を追加。
- `eval_pdf_closure()` を `Instant::now()` で計測し、profile出力の summary に `eval_pdf` として追加。
- `EVAL_PDF_VISITS` と `record_eval_pdf_visit()` を追加。
- closure visit histogram に `eval_pdf` 列を追加。

確認:

- `cargo check`: OK

修正版 profile 測定:

`cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --profile-mtlx -o result/perf/profile_opt2_profiler_fix_scene45_512_spp128.png`

注: この測定時点では後述の texture bilinear fast path の WIP 変更も入っていた。profile構造の確認には使えるが、純粋な opt2 profile としては扱わない。

結果:

- render: 00m:36s:693ms
- bytecode: 129224ms / 103,679,672 calls / 6,942,177,220 instrs
- sample: 43508ms / 29,123,572 calls
- eval: 54380ms / 14,013,136 calls
- pdf: 19590ms / 5,662,623 calls
- eval_pdf: 180962ms / 27,574,843 calls
- dalbedo top-level counter: 0ms / 0

clean profile 測定:

`cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --profile-mtlx -o result/perf/profile_opt2_profiler_fix_clean_scene45_512_spp128.png`

結果:

- render: 00m:37s:774ms
- bytecode: 136508ms / 103,670,274 calls / 6,941,032,601 instrs
- sample: 44857ms / 29,120,659 calls
- eval: 53978ms / 14,010,523 calls
- pdf: 19994ms / 5,661,663 calls
- eval_pdf: 179713ms / 27,570,320 calls
- dalbedo top-level counter: 0ms / 0

per-material bytecode 上位:

| slot | calls | total | per_call | avg_instrs |
| --- | ---: | ---: | ---: | ---: |
| 2 | 8,105,527 | 20193ms | 2491ns | 96 |
| 1 | 6,258,076 | 13777ms | 2201ns | 128 |
| 10 | 5,620,984 | 12450ms | 2215ns | 95 |
| 3 | 6,733,945 | 10052ms | 1493ns | 69 |
| 11 | 7,373,361 | 8358ms | 1133ns | 95 |

closure visit 上位:

- `Dielectric`: eval 132,043,458 / pdf 105,863,823 / sample 6,688,174 / dalbedo 213,650,529 / eval_pdf 87,859,908
- `Mix`: eval 59,628,727 / pdf 24,344,862 / sample 65,172,857 / eval_pdf 118,862,860
- `Layer`: eval 44,183,550 / pdf 18,003,915 / sample 65,633,436 / eval_pdf 87,859,908
- `Sheen`, `OrenNayar`, `Conductor`, `BurleyDiffuse`, `Translucent` は eval/pdf/eval_pdf がほぼ同数で、標準surface系の closure tree が全lobeを広く歩いている。

観察:

- opt2 で sample後の `eval + pdf` は消えたのではなく `eval_pdf` に移った。以降の profile は eval/pdf/eval_pdf を合わせて読む。
- slot 2/1/10/3 など画像テクスチャを含む材質の bytecode が上位。bytecode locals 作成は依然として大きい。
- closure 側では、標準surface由来の多層 lobe を毎回ほぼ全走査していることも大きい。

## 6. Optimization 3: texture bilinear fast path

調査:

- 修正版 profile でも bytecode calls は約1.04億で変わらず、per-material 上位は画像テクスチャを持つ MaterialX material に偏っている。
- `Texture::bilerp_level()` は通常の 0..1 UV の内側でも4 texelそれぞれに `pixel_level_wrapped()` を呼び、4回の wrap index 計算を行っている。
- MaterialX の image sampling は bytecode中で非常に頻繁に走るため、UVが完全に内側にある通常ケースでは直接 index する fast path が効く可能性がある。

実装:

- `src/material/texture.rs` の `bilerp_level()` に、`x0/y0` と隣接 texel が texture level 内に収まる場合だけ直接 `pixels[row + x]` を読む fast path を追加。
- 境界や wrap が必要な場合は従来どおり `pixel_level_wrapped()` を使う。

確認:

- `cargo check`: OK

測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt3_scene45_512_spp128.png` | 00m:29s:867ms |

結果:

- opt2 の 00m:29s:681ms より 0.186s 遅い。
- 差は小さいが、改善が確認できないため不採用。
- `src/material/texture.rs` の fast path 変更は戻した。

戻した後の確認:

- `cargo check`: OK

## 7. Texture cost experiments

方針:

- テクスチャが本当に支配的か確認するため、一時変更で MaterialX image sampling の寄与を切り分ける。
- まず MaterialX の image sampling を `default` 返しにして、画像サンプリング自体の上限改善幅を見る。
- 次に 4k texture が問題かを見るため、サンプリングする mip を +3 して 512 相当へ寄せた場合を測る。
- これらは画質を壊す調査用変更なので、採用前提ではなく、測定後に戻す。

### 7.1 Experiment A: disable MaterialX image sampling

一時変更:

- `runtime::sample_image_texture()` を `default` 返しにした。
- `runtime::hextiled_color_sample()` を `default` 返しにした。
- hextiled normal map の直接 texture sample はまだ残っているが、通常 image / hextiled color の主要経路は潰している。

確認:

- `cargo check`: OK

測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/exp_no_mtlx_textures_scene45_512_spp128.png` | 00m:26s:238ms |

結果:

- texture 有効の clean 測定 00m:29s:331ms から 3.093s 改善。約10.5%短縮。
- 画像サンプリングは無視できないが、全体差 4.3x の主因を単独で説明するほどではない。
- texture sampling 改善の上限はこの条件では数秒程度。先に texture だけを深掘りするより、closure/lobe 評価の構造改善も並行して見る必要がある。

profile測定:

`cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --profile-mtlx -o result/perf/profile_exp_no_mtlx_textures_scene45_512_spp128.png`

結果:

- render: 00m:33s:876ms
- bytecode: 74029ms / 105,473,152 calls / 7,058,866,694 instrs
- sample: 38705ms / 29,317,241 calls
- eval: 37314ms / 11,641,464 calls
- pdf: 14572ms / 4,006,025 calls
- eval_pdf: 168752ms / 28,145,793 calls

観察:

- bytecode thread累計は clean profile の 136508ms から 74029ms へ大きく減る。image sampling は bytecode実行時間のかなりの部分を占める。
- 一方で通常renderの改善は 3.1s / 10.5%。closure側の sample/eval/eval_pdf がまだ大きく、textureだけを潰しても scene 41 との差は大きく残る。
- per-material の per_call は slot 2: 2491ns -> 842ns、slot 1: 2201ns -> 1131ns、slot 10: 2215ns -> 848ns と大きく下がる。画像テクスチャ付き material の bytecode cost は確かに高い。
- この実験変更は画質を壊すため戻した。

戻した後の確認:

- `cargo check`: OK

### 7.2 Experiment B: force lower mip level for MaterialX texture sampling

一時変更:

- `Texture::sample_mip_bilinear()` の選択 mip level を `+3` した。
- 4k texture ならおおむね 512 相当の mip を読むことになる。
- texture node と sampling 処理は残るため、Experiment A より「テクスチャサイズ/cache locality」寄りの影響を見る。

確認:

- `cargo check`: OK

測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/exp_mtlx_texture_mip_plus3_scene45_512_spp128.png` | 00m:28s:948ms |

結果:

- clean 測定 00m:29s:331ms から 0.383s 改善。約1.3%短縮。
- no-texture の 3.093s 改善に比べるとかなり小さい。
- 4k texture のメモリ局所性/サイズだけが支配しているわけではなく、image node の処理、bilinear filtering、複数texture sample、closure側の評価が複合している。

profile測定:

`cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --profile-mtlx -o result/perf/profile_exp_mtlx_texture_mip_plus3_scene45_512_spp128.png`

結果:

- render: 00m:36s:596ms
- bytecode: 123807ms / 104,308,028 calls / 6,984,136,000 instrs
- sample: 37216ms / 29,304,290 calls
- eval: 44178ms / 14,110,897 calls
- pdf: 20702ms / 5,793,677 calls
- eval_pdf: 162390ms / 27,979,138 calls

観察:

- clean profile の bytecode 136508ms に対して 123807ms。profile上も texture size/cache locality の改善は見える。
- ただし no-texture の 74029ms までは下がらない。実サンプリング処理自体、アドレス処理、複数texture sample、グラフ実行が残っている。
- 512相当mip化は画質も変えるので採用しない。一時変更を戻した。

戻した後の確認:

- `cargo check`: OK

## 8. Optimization 4: evaluate leaf lobe eval/pdf together

調査:

- opt2 の `eval_pdf_closure()` は compositor tree の traversal は1回にしたが、leaf lobe では fallback として `eval_closure_idx()` と `pdf_closure_idx()` を別々に呼んでいた。
- そのため leaf lobe ごとに `override_frame_for_wo()`、`rebase_wi_into_frame()`、BSDF object 構築、MaterialX param read が重複していた。
- clean profile では `eval_pdf` が 27,570,320 calls / 179713ms thread累計で、ここを直接削る余地がある。

実装:

- `eval_pdf_closure_idx()` に OrenNayar / Burley / Translucent / Dielectric / Conductor / GeneralizedSchlick / Sheen / ChiangHair の leaf branch を追加。
- leaf branch では frame override と `wo/wi` rebase を1回だけ行い、eval と pdf を同じ BSDF object から返す。
- Conductor の pdf は既存 `pdf_closure_idx()` と同じ挙動を保つため、従来通り `ConductorBsdf::new(1.0, Vec3::ONE, Vec3::ZERO, rough)` を使う。
- EDF / NPR / ThinFilm 単体は従来通り `(Vec3::ZERO, 0.0)`。

確認:

- `cargo check`: OK

測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt4_scene45_512_spp128.png` | 00m:28s:650ms |

結果:

- clean 測定 00m:29s:331ms から 0.681s 改善。約2.3%短縮。
- opt2 初回測定 00m:29s:681ms と比べると 1.031s 改善。測定ばらつきはあるが、leaf eval/pdf 共有は有効そうなので採用候補として残す。

profile測定:

`cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --profile-mtlx -o result/perf/profile_opt4_scene45_512_spp128.png`

結果:

- render: 00m:32s:007ms
- bytecode: 134206ms / 103,685,463 calls / 6,941,993,396 instrs
- sample: 38964ms / 29,127,372 calls
- eval: 42251ms / 14,016,905 calls
- pdf: 15556ms / 5,666,106 calls
- eval_pdf: 98479ms / 27,577,181 calls
- dalbedo: 0ms / 0

観察:

- clean profile の eval_pdf 179713ms から 98479ms へ大きく低下。leaf lobe で eval と pdf を個別に走らせていた重複を消せている。
- eval と pdf も少し低下しているが、bytecode は 136508ms から 134206ms と小さい低下に留まる。これはこの最適化が MaterialX graph VM より closure walker / BSDF object 構築 / frame rebase 側に効いているため。
- profile付き render は 00m:37s:774ms から 00m:32s:007ms に改善。profile overhead込みでも傾向は明確。
- 通常renderは 00m:28s:650ms で、採用して先に進む。

## 9. Optimization 5: restore MtlxScratch per camera trace

調査:

- `MtlxScratch` は stack allocator として `checkpoint()` / `restore()` を持っているが、通常レンダリングの camera sample ごとには戻されていなかった。
- `IntegratorKind::trace_radiance()` に渡される thread-local scratch は `pixels.par_chunks_mut(...).for_each_init()` で thread ごとに作られ、その後レンダー全体で使い回される。
- そのため `precompute_shading()` の `alloc_regs()` が各hitごとに `regs_pool` を伸ばし続け、過去の camera sample の register 領域もレンダー終了まで保持していた。
- このコストは `run_instructions()` の bytecode profiler の外側なので、既存profileの `bytecode` 時間には含まれない。メモリ使用量、capacity拡張、cache locality の問題として効く可能性がある。

実装:

- `IntegratorKind::trace_radiance()` の入口で `MtlxScratch::checkpoint()` を取り、各 integrator の `trace_radiance()` 呼び出し後に `restore()` する。
- path 内では現在の shading vertex、next vertex、light hit vertex の register lifetime が必要なので、path の途中では戻さない。camera sample 1本が終わった時点でまとめて戻す。

確認:

- `cargo check`: OK

測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt5_scratch_restore_scene45_512_spp128.png` | 00m:28s:338ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt5_scratch_restore_scene41_512_spp128.png` | 00m:06s:558ms |

結果:

- scene45 は opt4 の 00m:28s:650ms から 0.312s 改善。約1.1%短縮。
- scene41 も 00m:06s:878ms 付近から 00m:06s:558ms へ改善しており、MaterialX専用というより thread-local scratch の lifetime 修正と測定ばらつきの両方が出ている。
- scene45 / scene41 比は 28.338 / 6.558 = 4.32x。目標の 1.4x には遠い。
- 構造的には正しい lifetime に戻す修正なので採用して先に進む。

## 10. Profiler expansion and Optimization 6: reduce light-tree Layer duplicate work

profiler確認:

- opt5 時点で `light_tree_precompute_closure()` と `opacity()` が profile 対象外だった。
- `direct_light_mis_contribution()` は shading point ごとに `light_tree::build_query()` を呼び、その中で MaterialX の `light_tree_precompute()` が走る。
- ここが未計測だと、MaterialX closure walker の大きなコストを見落とすため、`light_tree_precompute` と `opacity` の profiler counter を追加した。
- compile profile log には `slot` だけでなく MaterialX material name も出すようにした。

profile測定:

`cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --profile-mtlx -o result/perf/profile_opt5_profiler_expanded_scene45_512_spp128.png`

結果:

- render: 00m:32s:201ms
- bytecode: 130558ms / 103,696,379 calls / 6,942,781,779 instrs
- sample: 38564ms / 29,130,951 calls
- eval: 41435ms / 14,022,146 calls
- pdf: 15299ms / 5,663,661 calls
- eval_pdf: 97495ms / 27,581,331 calls
- light_tree_precompute: 103680ms / 29,745,265 calls
- opacity: 1654ms / 55,710,446 calls

観察:

- `light_tree_precompute` が thread累計 103680ms で、MaterialX closure walker の主要ホットスポットだった。
- closure histogram の `dalbedo` visit が public `directional_albedo_closure()` の 0 calls と矛盾して見えていたが、実際には `light_tree_precompute` 内部の `directional_albedo_idx()` が記録していた。profilerの構造として、公開API別の時間と内部closure visitが一致しない点に注意が必要。
- opacity は call数は多いが 1654ms で、現時点の主因ではない。
- per-material hotspot は slot 2 `Copper_Satin`, slot 1 `Car_Paint`, slot 10 `Bronze_Oxydized`, slot 3 `Emerald_Peaks_Wallpaper`。

実装:

- `light_tree_summary_idx()` の `Layer` は、top branch の `directional_albedo_idx_scalar()` を別 traversal で計算した後、同じ top branch を summary 用に再 traversal していた。
- light-tree sampling 用の近似重みとしては、top summary が既に持つ `diffuse_rho + glossy_rho + btdf_rho` を使えばよいので、`LightTreeClosureSummary::energy_scalar()` を追加してこれを再利用した。
- これにより `Layer` の top branch に対する directional albedo 重複計算を削る。

確認:

- `cargo check`: OK

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt6_light_tree_layer_energy_scene45_512_spp128.png` | 00m:26s:633ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt6_light_tree_layer_energy_scene41_512_spp128.png` | 00m:06s:821ms |

結果:

- scene45 は opt5 の 00m:28s:338ms から 1.705s 改善。約6.0%短縮。
- scene45 / scene41 比は 26.633 / 6.821 = 3.90x。まだ目標の 1.4x には遠い。
- light-tree sampling の近似重み変更なので、画像の大きな破綻がないか継続確認する。

profile測定:

`cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --profile-mtlx -o result/perf/profile_opt6_light_tree_layer_energy_scene45_512_spp128.png`

結果:

- render: 00m:33s:626ms
- bytecode: 132399ms / 103,671,133 calls / 6,940,786,105 instrs
- sample: 45792ms / 29,121,558 calls
- eval: 50764ms / 14,016,234 calls
- pdf: 18637ms / 5,664,541 calls
- eval_pdf: 121979ms / 27,571,308 calls
- light_tree_precompute: 60497ms / 29,736,338 calls
- opacity: 1701ms / 55,695,837 calls

観察:

- `light_tree_precompute` は 103680ms から 60497ms に低下し、狙った箇所に効いている。
- `dalbedo` visit も Dielectric 213,733,389 -> 150,665,649、Sheen 96,574,885 -> 65,046,239 に低下。
- profile付きでは sample/eval/eval_pdf が上振れしているが、通常renderでは明確に改善している。profile overhead と乱数経路の差が大きいため、採否は通常render時間を優先する。
- 採用して先に進む。

## 11. Optimization 7: use scene-owned Sheen directional albedo LUT for MaterialX

調査:

- opt6 後も `dalbedo` visit では Sheen が大きい。
- `SheenBsdfMtlx::directional_albedo()` の Conty/Kulla mode は `sheen_directional_albedo_estimate(self.roughness, wo.z, 32)` を毎回実行していた。
- 一時実験として 32 samples を 8 samples に落としたところ、scene45 は 00m:23s:772ms まで改善した。
- ただし sample数を落とすのは精度を直接落とす変更なので、そのまま採用せず、既存の `SheenDirectionalAlbedoLut` を MaterialX から使う方針にした。

設計:

- 既存の non-MaterialX 側では `Scene` が `DirectionalAlbedoCache` を持ち、`Scene::add_material()` 時に LUT を material に `Arc` で注入している。
- MaterialX も同じ設計に揃えた。lazy static や暗黙globalは使わない。
- `CompiledMaterial` に scene-installed の `sheen_lut` を持たせ、`Scene::add_material(Material::Mtlx)` で `DirectionalAlbedoCache::get_or_build_sheen()` を注入する。
- runtime の Sheen directional albedo は LUT 必須にし、未注入なら panic する。テストだけ別の推定経路に落ちる形にはしない。

実装:

- `MtlxMaterial::install_sheen_lut()` を追加し、front/back の `CompiledMaterial` に LUT を設定。
- `Scene::add_material()` で MaterialX material に Sheen LUT を注入。
- `SheenBsdfMtlx::directional_albedo_with_lut()` を追加し、MaterialX runtime の `light_tree_summary_idx()` と `directional_albedo_idx()` の Sheen branch から使う。
- テストで直接 MaterialX material をロードする `load_shader_ops_thin_walled()` にも `DirectionalAlbedoCache` から実 LUT を注入。
- 手書き `CompiledMaterial` のうち Sheen branch を通らないものは `sheen_lut: None` のままにし、Sheen branch を通る runtime では LUT 必須にした。

確認:

- `cargo check`: OK
- `cargo test --no-run`: OK
- `cargo test`: OK。714 tests passed。

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt7_mtlx_sheen_lut_scene45_512_spp128.png` | 00m:22s:261ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt7_mtlx_sheen_lut_scene41_512_spp128.png` | 00m:07s:090ms |

結果:

- scene45 は opt6 の 00m:26s:633ms から 4.372s 改善。約16.4%短縮。
- 8 samples 一時実験の 00m:23s:772ms よりも速く、既存 LUT の 256 samples 生成結果を使うので画質面でもこちらを採用する。
- scene45 / scene41 比は 22.261 / 7.090 = 3.14x。まだ目標の 1.4x には遠い。

profile測定:

`cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --profile-mtlx -o result/perf/profile_opt7_mtlx_sheen_lut_scene45_512_spp128.png`

結果:

- render: 00m:30s:331ms
- bytecode: 130084ms / 103,679,599 calls / 6,941,123,967 instrs
- sample: 36103ms / 29,122,042 calls
- eval: 38655ms / 14,017,466 calls
- pdf: 14385ms / 5,665,909 calls
- eval_pdf: 98255ms / 27,572,036 calls
- light_tree_precompute: 27013ms / 29,736,480 calls
- opacity: 1683ms / 55,705,409 calls

観察:

- `light_tree_precompute` は opt6 の 60497ms から 27013ms に低下。Sheen directional albedo の毎回積分が主要因だった。
- `dalbedo` visit 数自体は残るが、Sheen branch の中身が積分から LUT lookup に変わったため thread累計が大きく低下した。
- 採用して先に進む。

画像確認:

- 512x512 / 128spp の `baseline_scene45_512_spp128.png` と `opt7_mtlx_sheen_lut_scene45_512_spp128.png` を目視比較。
- ノイズパターンの差はあるが、材質配置、全体の色味、強い発光材質に大きな破綻は見えない。
- `magick compare -metric RMSE` は `3225.95 (0.0492249)`。
- diff画像: `result/perf/diff_opt7_vs_baseline_scene45_512_spp128.png`

## 12. Optimization attempt 8: combine sampling traversal with eval/pdf traversal

調査:

- opt7 profile では `sample` と `eval_pdf` の累計がまだ大きい。
- `MtlxMaterial::sample()` は `sample_closure()` で closure tree から lobe を選んだ直後に、同じ root から `eval_pdf_closure()` を走らせて full closure の `f/pdf` を求めている。
- 選択された branch については sampling 時に compositor path を辿っているため、`sample_eval_pdf_closure()` で選択 branch の sample と eval/pdf を同時に返し、未選択 branch だけ `eval_pdf` する実験を行った。

実装内容:

- `sample_eval_pdf_closure()` と profiler counter `sample_eval_pdf` を一時追加。
- `Mix` / `Layer` / `Add` / `Multiply` / conditional / `Surface` で、選択 branch の sample と eval/pdf を共有し、未選択 branch の eval/pdf だけを追加計算するようにした。
- leaf lobe では既存 `sample_closure_idx()` と `eval_pdf_closure_idx()` を使うため、BSDF内部の eval/pdf 重複までは削っていない。

測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt8_sample_eval_pdf_scene45_512_spp128.png` | 00m:22s:443ms |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt8_sample_eval_pdf_scene45_512_spp128_run2.png` | 00m:23s:428ms |

profile測定:

`cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --profile-mtlx -o result/perf/profile_opt8_sample_eval_pdf_scene45_512_spp128.png`

結果:

- render: 00m:29s:437ms
- bytecode: 132918ms / 103,701,523 calls / 6,942,328,879 instrs
- sample_eval_pdf: 86567ms / 29,130,718 calls
- eval: 35865ms / 14,018,367 calls
- pdf: 14573ms / 5,665,299 calls
- light_tree_precompute: 27938ms / 29,746,174 calls

判断:

- profile上は opt7 の `sample + eval_pdf` 合算より低下して見える。
- しかし通常renderは opt7 の 00m:22s:261ms に対して 00m:22s:443ms / 00m:23s:428ms と悪化した。
- 追加の再帰分岐、未計測部分、leaf側で結局 eval/pdf を再計算していることが効いていると判断。
- この実験実装は戻した。

戻した後の確認:

- `cargo check`: OK

## 13. Optimization 9: compile-time closure simplification and bytecode DCE

調査:

- opt7 profile では bytecode 自体が 130084ms / 6,941,123,967 instrs とまだ大きい。
- scene45 の MaterialX は standard_surface 系の closure tree が多く、weight=0 の lobe や static な `mix=0/1`、`multiply scale=1` が runtime の closure walker と bytecode local 参照に残っていた。
- closure tree を簡略化しても、元の未到達 closure node に残った `ParamRef::Local` を DCE や register allocation が live と見なすと bytecode は消えない。そのため root から到達する closure だけを live-out として扱う必要がある。

実装:

- `compile()` の closure 構築後に `simplify_closure_nodes()` を追加。
- static zero weight の BSDF lobe を `Zero` にし、`Mix` / `Layer` / `Add` / `Multiply` の静的に消せる node を child または `Zero` に畳み込む。
- SSA bytecode に `eliminate_dead_instructions()` を追加し、root closure から到達する `ParamRef::Local` と、それらを生成する命令だけを残す。
- register allocation の live-out 判定も全 closure node ではなく root 到達 closure に揃えた。これにより、簡略化で切った古い branch の local が slot lifetime を伸ばさない。

確認:

- `cargo check`: OK
- `cargo test`: OK。lib 712 tests + main 2 tests passed。

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt9_closure_simplify_dce_scene45_512_spp128.png` | 00m:19s:552ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt9_closure_simplify_dce_scene41_512_spp128.png` | 00m:06s:833ms |

結果:

- scene45 は opt7 の 00m:22s:261ms から 2.709s 改善。約12.2%短縮。
- scene45 / scene41 比は 19.552 / 6.833 = 2.86x。まだ目標の 1.4x には遠い。
- scene41 は MaterialX を含まないためほぼ影響なし。

profile測定:

`cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --profile-mtlx -o result/perf/profile_opt9_closure_simplify_dce_scene45_512_spp128.png`

結果:

- render: 00m:23s:661ms
- bytecode: 108227ms / 103,942,744 calls / 4,553,747,937 instrs
- sample: 18980ms / 29,119,456 calls
- eval: 11626ms / 13,687,031 calls
- pdf: 4277ms / 5,578,234 calls
- eval_pdf: 31177ms / 27,568,663 calls
- light_tree_precompute: 9888ms / 29,732,075 calls
- opacity: 1685ms / 55,399,900 calls

観察:

- bytecode instruction 数は opt7 の 6.94B から 4.55B へ低下。
- `eval` は 38655ms から 11626ms、`eval_pdf` は 98255ms から 31177ms、`light_tree_precompute` は 27013ms から 9888ms に低下。
- closure visit histogram でも Layer / Mix / OrenNayar / Conductor / Sheen などの visit が大きく減っており、静的に無効な lobe を runtime で辿らない形になっている。
- 採用して先に進む。

画像確認:

- `baseline_scene45_512_spp128.png` と `opt9_closure_simplify_dce_scene45_512_spp128.png` を目視比較。
- 材質配置、色味、発光材質に大きな破綻は見えない。
- `magick compare -metric RMSE` は `3253.42 (0.049644)`。
- diff画像: `result/perf/diff_opt9_vs_baseline_scene45_512_spp128.png`

## 14. Optimization 10: use combined eval/pdf for direct-light MIS

調査:

- opt9 profile では direct-light 側で `material.eval()` の後に、非デルタ light のときだけ同じ `wi` で `material.pdf()` を呼んでいた。
- MaterialX runtime にはすでに `eval_pdf_closure()` があり、sample後だけでなく direct-light MIS でも同じ closure traversal の重複を消せる。
- delta light では BSDF pdf が不要なので、そこでは従来どおり eval のみを呼ぶ。

実装:

- `MtlxMaterial::eval_pdf()` を追加し、`eval()` / `pdf()` と同じ geometric / thin-walled guard の後に `runtime::eval_pdf_closure()` を呼ぶようにした。
- `Material::eval_pdf()` を追加。MaterialX 以外は既存の `eval()` と `pdf()` を呼ぶ fallback にした。
- `direct_light_mis_contribution()` で、非デルタ light の direct-light 評価を `material.eval_pdf()` に切り替えた。

確認:

- `cargo check`: OK
- `cargo test`: OK。lib 712 tests + main 2 tests passed。

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt10_direct_light_eval_pdf_scene45_512_spp128.png` | 00m:18s:505ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt10_direct_light_eval_pdf_scene41_512_spp128.png` | 00m:06s:813ms |

結果:

- scene45 は opt9 の 00m:19s:552ms から 1.047s 改善。約5.4%短縮。
- scene45 / scene41 比は 18.505 / 6.813 = 2.72x。まだ目標の 1.4x には遠い。

profile測定:

`cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --profile-mtlx -o result/perf/profile_opt10_direct_light_eval_pdf_scene45_512_spp128.png`

結果:

- render: 00m:24s:323ms
- bytecode: 111954ms / 103,957,135 calls / 4,554,747,253 instrs
- sample: 19527ms / 29,125,898 calls
- eval: 0ms / 0 calls
- pdf: 0ms / 0 calls
- eval_pdf: 48505ms / 41,267,159 calls
- light_tree_precompute: 10183ms / 29,740,783 calls
- opacity: 1795ms / 55,407,304 calls

観察:

- direct-light 側の `eval` / `pdf` は `eval_pdf` に統合され、profile上も `eval` / `pdf` calls は 0 になった。
- profile付きでは opt9 の `eval + pdf + eval_pdf` 合算 47080ms に対して `eval_pdf` 48505ms と少し上振れしたが、通常renderでは改善している。
- 採用して先に進む。

画像確認:

- `baseline_scene45_512_spp128.png` と `opt10_direct_light_eval_pdf_scene45_512_spp128.png` を目視比較。
- 材質配置、色味、発光材質に大きな破綻は見えない。
- `magick compare -metric RMSE` は `3240.95 (0.0494538)`。
- diff画像: `result/perf/diff_opt10_vs_baseline_scene45_512_spp128.png`

## 15. Optimization 11: lazily build light-tree query only for Tree light samples

調査:

- `direct_light_mis_contribution()` は light category を選ぶ前に常に `light_tree::build_query()` を呼んでいた。
- `LightCategory::Environment` や `LightCategory::Directional` が選ばれた場合、この query は使われない。
- scene45 は Tree に格納される emissive mesh / point / spot 側が支配的なので改善幅は出にくいが、構造としては不要な MaterialX `light_tree_precompute` を削れる。

実装:

- 既存の `sample_light_mis_compensated()` は残し、direct-light MIS 用に `sample_light_mis_compensated_lazy()` を追加。
- lazy API は最初に top-level light category を選び、`LightCategory::Tree` の場合だけ `light_tree::build_query()` を呼ぶ。
- `direct_light_mis_contribution()` を lazy API に切り替えた。

確認:

- `cargo check`: OK
- `cargo test`: OK。lib 712 tests + main 2 tests passed。

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt11_lazy_light_tree_query_scene45_512_spp128.png` | 00m:18s:550ms |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt11_lazy_light_tree_query_scene45_512_spp128_run2.png` | 00m:18s:697ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt11_lazy_light_tree_query_scene41_512_spp128.png` | 00m:06s:820ms |

結果:

- scene45 は opt10 の 00m:18s:505ms に対して 00m:18s:550ms / 00m:18s:697ms と僅差で悪化。
- scene45 / scene41 比は 18.550 / 6.820 = 2.72x。
- ただし scene45 は Tree light が支配的で、lazy 化してもほとんどの direct-light sample で query が必要になる。環境光・directional light が一定以上ある scene では不要 precompute 削減として効くため、構造的に正しい最適化として採用する。

profile測定:

`cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --profile-mtlx -o result/perf/profile_opt11_lazy_light_tree_query_scene45_512_spp128.png`

結果:

- render: 00m:23s:507ms
- bytecode: 105854ms / 103,967,288 calls / 4,555,534,780 instrs
- sample: 18947ms / 29,128,248 calls
- eval: 0ms / 0 calls
- pdf: 0ms / 0 calls
- eval_pdf: 40863ms / 41,268,954 calls
- light_tree_precompute: 9392ms / 28,128,177 calls
- opacity: 1677ms / 55,412,521 calls

観察:

- `light_tree_precompute` calls は opt10 の 29,740,783 から 28,128,177 に減った。
- 通常renderで改善が見えないのは、削減対象が scene45 の主要経路ではないためと判断。
- 採用して先に進む。

画像確認:

- `baseline_scene45_512_spp128.png` と `opt11_lazy_light_tree_query_scene45_512_spp128.png` を目視比較。
- 材質配置、色味、発光材質に大きな破綻は見えない。
- `magick compare -metric RMSE` は `3254.61 (0.0496621)`。
- diff画像: `result/perf/diff_opt11_vs_baseline_scene45_512_spp128.png`

## 16. Optimization 12: skip MaterialX emission walk when may_emit is false

調査:

- `MtlxMaterial::le()` は `CompiledMaterial::may_emit == false` でも `evaluate_le()` を呼べる構造だった。
- `may_emit` は compile 時に closure の最大 emission から計算済みなので、false の material は emission closure を runtime で歩く必要がない。
- scene45 では direct hit / BSDF hit 後に非発光 MaterialX material へ `le()` が呼ばれる経路があるため、小さいが確実な guard として入れた。

実装:

- `MtlxMaterial::le()` の先頭で active compiled material の `may_emit` を確認し、false なら `None` を返す。

確認:

- `cargo check`: OK
- `cargo test`: OK。lib 712 tests + main 2 tests passed。

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt12_mtlx_le_may_emit_guard_scene45_512_spp128.png` | 00m:18s:633ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt12_mtlx_le_may_emit_guard_scene41_512_spp128.png` | 00m:07s:261ms |

結果:

- scene45 は opt11 の 00m:18s:550ms / 00m:18s:697ms と同程度で、明確な改善は見えない。
- scene41 は MaterialX を含まないため測定揺れのみ。
- guard 自体は compile 済み情報に基づいて不要な emission walk を消すだけなので、採用して先に進む。

画像確認:

- `baseline_scene45_512_spp128.png` と `opt12_mtlx_le_may_emit_guard_scene45_512_spp128.png` を目視比較。
- 材質配置、色味、発光材質に大きな破綻は見えない。
- `magick compare -metric RMSE` は `3246.88 (0.0495443)`。
- diff画像: `result/perf/diff_opt12_vs_baseline_scene45_512_spp128.png`

## 17. Optimization 13: skip eval_pdf recomputation for single-lobe MaterialX sample paths

調査:

- MaterialX の `sample()` は BSDF sample 後に `eval_pdf_closure()` を呼び、sample済み方向の `eval` / `pdf` / MIS 用 weight を再計算していた。
- compile時に closure graph を見れば、単一ローブ相当で layer / mix / add / switch / conditional を通らない material では sample時の `candidate.weight` と `candidate.pdf` をそのまま使える。
- scene45 ではすべての MaterialX material が該当するわけではないが、`eval_pdf` call 数を削れる可能性がある。

実装:

- `CompiledMaterial` に `sample_needs_eval_pdf` を追加。
- compile時に closure graph を走査し、sample後に eval_pdf 再評価が必要な構造だけ `true` にする。
- `MtlxMaterial::sample()` で `sample_needs_eval_pdf == false` の場合は `eval_pdf_closure()` を呼ばず、sample result の `pdf` / `weight` を使う。
- テスト用の `CompiledMaterial` literal も実行時と同じ field を持つように更新した。

確認:

- `cargo check`: OK
- `cargo test`: OK。lib 712 tests + main 2 tests passed。

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt13_single_lobe_sample_fast_path_scene45_512_spp128.png` | 00m:18s:531ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt13_single_lobe_sample_fast_path_scene41_512_spp128.png` | 00m:07s:015ms |

profile測定:

`cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --profile-mtlx -o result/perf/profile_opt13_single_lobe_sample_fast_path_scene45_512_spp128.png`

結果:

- render: 00m:24s:126ms
- sample: 21980ms
- eval_pdf calls: 39,281,723

観察:

- `eval_pdf` calls は opt11/12 付近の約 41.27M から 39.28M へ減った。
- 一方で profile付きの `sample` thread time は増えており、scene45 通常renderの改善も opt11/12 からほぼ横ばい。
- 単一ローブで sample済みの値を再利用すること自体は論理的に不要計算の削減なので、採用して先に進む。ただし scene45 の支配的な重さはまだ別経路にある。

画像確認:

- `baseline_scene45_512_spp128.png` と `opt13_single_lobe_sample_fast_path_scene45_512_spp128.png` を目視比較。
- 材質配置、色味、発光材質に大きな破綻は見えない。
- `magick compare -metric RMSE` は `3229.07 (0.0492725)`。
- diff画像: `result/perf/diff_opt13_vs_baseline_scene45_512_spp128.png`

## 18. Directional albedo LUT investigation for MaterialX

調査:

- 既存実装では `DirectionalAlbedoCache` を scene が所有し、material には `Arc` で immutable LUT を渡す設計になっている。
- 既存LUTは以下:
  - `DielectricGgxDirectionalAlbedoLut`: etaごとの reflection GGX directional albedo。sqrt(cos), phi, roughness, anisotropy の4D。
  - `SheenDirectionalAlbedoLut`: cos, roughness の2D。MaterialX sheen には導入済み。
  - `ConductorGgxEnergyCompensationLut`: F=1 の GGX reflection directional albedo `E(cos, roughness)` と `Eavg`。energy compensation用だが、白色Fresnelのdirectional albedoとして再利用できる。
  - `DielectricGgxEnergyCompensationLut`: cos, roughness, eta の3D full-sphere energy compensation用。
- `Scene::add_material()` では SimplePBR / StandardSurface / Sheen / energy compensation 用LUTは scene cache から注入しているが、MaterialX の dielectric / conductor / generalized_schlick には sheen 以外のLUTを渡していない。
- MaterialX runtime の `directional_albedo_idx()` と `light_tree_summary_idx()` は dielectric / conductor / generalized_schlick で一度BSDF structを組み立ててから `directional_albedo()` を呼ぶ。ただし現状の各BSDFの `directional_albedo()` は厳密積分ではなく、ほぼ `F(cos) * weight * tint` の近似で、roughness/maskingを含まない。

文献メモ:

- Turquin の "Practical multiple scattering compensation for microfacet models" は directional albedo `E(wo)` を積分として定義し、事前計算または小さなLUTで扱う方向を示している。
- Kulla/Conty の Imageworks SIGGRAPH 2017 course も GGX系の directional albedo / average albedo を使った energy compensation と layering の文脈で近い。
- Heitz et al. 2016 は Smith microfacet の多重散乱を扱うが、runtime random walk は重いので、このコードベースでは Turquin/Kulla-Conty 系のLUT/近似の方が現実的。
- MaterialX は Standard Surface などのPBR shading nodesを標準化しているが、任意のMaterialX closure graph全体をそのままLUT化する規定はない。

設計判断:

- 任意の MaterialX closure graph 全体をLUT化するのは、texture、color、roughness、ior、thin film、layer/mix/addの組み合わせで次元が爆発するため不適切。
- 既存設計に合わせ、scene所有の `DirectionalAlbedoCache` にローブ単位LUTを持たせ、MaterialX compiled material に `Arc` を注入する方針がよい。
- Sheen はすでにこの形で実装済み。
- Dielectric の exact reflection directional albedo は既存4D LUTを使えるが、MaterialX の `ior` がtexture/graph入力になる場合は eta per material key だけでは不十分。static ior なら既存 `DielectricGgxDirectionalAlbedoLut` を使える。
- Conductor / generalized_schlick は eta/extinction/color/exponent まで含めた exact LUT は高次元すぎる。まずは `E_white(cos, roughness)` と runtime Fresnel/color を分離する近似が現実的。
- ただし `directional_albedo_idx()` は layer の実効透過率や pdf mixture に使われるため、既存近似から roughness-aware 近似へ変えるとレンダリング結果も変わる。最初は light-tree summary のような sampling importance 側から適用し、画像差を確認するのが安全。

次に試す実装案:

- MaterialX compiled material に GGX reflection white directional albedo LUT を注入する。
- 既存 `ConductorGgxEnergyCompensationLut::lookup_e(cos, roughness)` を、名前上は energy compensation だが F=1 GGX directional albedoとしてMaterialX light-tree summaryで再利用する。
- Dielectric / conductor / generalized_schlick の `light_tree_summary_idx()` でBSDF struct構築を避け、Fresnel/colorを直接計算し、必要なら `E_white(cos, roughness_proxy)` を掛ける。
- 結果が壊れない場合に、`directional_albedo_idx()` 側へ拡張するかどうかを追加測定する。

## 19. Optimization 14: use scene-owned GGX white directional albedo LUT in MaterialX light-tree summary

調査:

- MaterialX に sheen 以外の directional albedo LUT が渡されていなかった。
- `ConductorGgxEnergyCompensationLut::lookup_e(cos, roughness)` は F=1 の GGX reflection directional albedo として使える。
- まず rendering のBSDF値や layer/pdf mixture は変えず、light-tree summary の glossy importance にだけ `E_white * runtime Fresnel/color` を入れる形で試した。

実装:

- `CompiledMaterial` に `ggx_reflection_lut` を追加。
- `Scene::add_material()` で MaterialX material に scene-owned `ConductorGgxEnergyCompensationLut` を注入。
- `light_tree_summary_idx()` の MaterialX dielectric / conductor / generalized_schlick で `E_white(cos, roughness)` を反映。
- テストfixtureには runtime と同じ field を持たせ、sceneを通さない light-tree テストでは constant LUT を使うようにした。

確認:

- `cargo check`: OK
- `cargo test`: OK。lib 712 tests + main 2 tests passed。

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt14_mtlx_ggx_reflection_lut_light_tree_scene45_512_spp128.png` | 00m:18s:344ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt14_mtlx_ggx_reflection_lut_light_tree_scene41_512_spp128.png` | 00m:06s:784ms |

profile測定:

`cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --profile-mtlx -o result/perf/profile_opt14_mtlx_ggx_reflection_lut_light_tree_scene45_512_spp128.png`

結果:

- render: 00m:21s:313ms
- bytecode: 104450ms / 103,977,408 calls / 4,556,050,249 instrs
- sample: 14460ms / 29,120,157 calls
- eval_pdf: 34315ms / 39,306,923 calls
- light_tree_precompute: 10170ms / 28,120,848 calls
- opacity: 1645ms / 55,402,817 calls
- closure-walker dalbedo visits: Dielectric 61,584,882、Sheen 3,070,781

観察:

- scene45 通常renderは opt13 の 00m:18s:531ms から 00m:18s:344ms へ僅かに改善。
- `light_tree_precompute` のprofile timeは opt13/opt11付近から大きく改善していない。
- 画像比較 RMSE は `3249.31 (0.0495813)`。
- diff画像: `result/perf/diff_opt14_vs_baseline_scene45_512_spp128.png`
- light-tree summary限定なら結果破綻は見えず、わずかに速いので一旦採用して先へ進む。

## 20. Experiment 15: extend separable GGX directional albedo LUT to MaterialX layer/pdf directional_albedo_idx

調査:

- ユーザー指摘どおり directional albedo 自体が厳密なランダムウォーク layering の近似であり、`E_white * runtime Fresnel/color` の分離近似も許容できる可能性がある。
- `directional_albedo_idx()` は layer の透過率や Add の sampling weight、eval_pdf の mixture pdf に多く使われるため、ここにも `E_white` を入れる実験を行った。

実装:

- MaterialX dielectric / conductor / generalized_schlick の directional albedo を、BSDF struct 構築ではなく direct helper に置き換えた。
- direct helper は runtime Fresnel/color に `ggx_reflection_lut.lookup_e(cos, roughness)` を掛ける。
- LUT lookup cost を測るため `ggx_dalbedo_lut` profile counter を追加。
- LUT indexing/補間確認として、cell center lookup が元テーブル値を返すテストと、値域が 0..1 に収まるテストを追加。

確認:

- `cargo check`: OK
- `cargo test`: OK。lib 712 tests + main 2 tests passed。
- `cargo test bsdf::directional_albedo::tests::`: OK。2 tests passed。

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt15_mtlx_separable_directional_albedo_scene45_512_spp128.png` | 00m:18s:790ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt15_mtlx_separable_directional_albedo_scene41_512_spp128.png` | 00m:07s:346ms |

profile測定:

`cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --profile-mtlx -o result/perf/profile_opt15_lut_counter_scene45_512_spp128.png`

結果:

- render: 00m:22s:163ms
- bytecode: 104580ms / 104,039,575 calls / 4,559,326,936 instrs
- sample: 18312ms / 29,138,979 calls
- eval_pdf: 42567ms / 39,368,037 calls
- light_tree_precompute: 15912ms / 28,140,522 calls
- opacity: 1615ms / 55,434,572 calls
- ggx_dalbedo_lut: 4448ms / 97,920,868 calls
- closure-walker dalbedo visits: Dielectric 61,658,737、Sheen 3,072,345

観察:

- `directional_albedo_idx()` のホットローブは Dielectric で、LUTを当てた対象自体はホットだった。
- ただし現在の MaterialX Dielectric directional albedo は元々ほぼ Fresnel 評価だけなので、LUTは「重い積分の置換」ではなく「軽い近似への追加コスト」になっていた。
- `ggx_dalbedo_lut` は約 98M calls / 4448ms と無視できない。
- scene45 通常renderも opt14 の 00m:18s:344ms から 00m:18s:790ms へ悪化。
- 画像比較 RMSE は `3251.04 (0.0496077)`。
- diff画像: `result/perf/diff_opt15_vs_baseline_scene45_512_spp128.png`

判断:

- LUT自体の補間/indexingの基本テストは通っており、実装破綻ではなさそう。
- 悪化原因は、現状の軽い Fresnel 近似に LUT lookup を追加したことによるコスト増と判断。
- `directional_albedo_idx()` 側への全面適用は採用しない。
- 次は opt15 の `directional_albedo_idx()` 側変更を戻し、opt14 の light-tree summary 限定適用に戻して再測定する。

## 21. Optimization 16: replace Fresnel-only MaterialX Dielectric directional albedo with scene-owned GGX integration LUT

調査:

- opt14/15 の `E_white * runtime Fresnel/color` 分離近似は、GGX分布やSmith maskingを含む MaterialX Dielectric の directional albedo そのものではなく、外部文献や既存式に基づく近似としては弱い。
- MaterialX Dielectric の hot path は `directional_albedo_idx()` 内の Dielectric で、profile上も約60M calls規模で支配的だった。
- 既存の `DielectricGgxDirectionalAlbedoLut` は anisotropy と azimuthal orientation を含む4D LUTだが、MaterialX runtime の per-call LUT としては補間コストと次元が大きい。
- ユーザー指摘どおり、まず anisotropy を `sqrt(alpha_x * alpha_y)` に潰した別LUTを作る方針にした。
- thin film は thickness / film ior まで次元に入れるとLUTが大きくなりすぎるため、今回のLUT対象から外し、既存の `DielectricBsdf::directional_albedo()` fallback を維持した。

実装:

- opt14/15 の Fresnel 分離近似用 `ggx_reflection_lut` 実装は採用しない方針に戻した。
- `DirectionalAlbedoCache` に scene-owned `MtlxDielectricGgxDirectionalAlbedoLut` を追加。
- `Scene::add_material()` で MaterialX material に `sheen_lut` と `mtlx_dielectric_lut` を注入。
- `MtlxDielectricGgxDirectionalAlbedoLut` は `sqrt(cos_o) x alpha_eq x eta_rel` の3D LUT。
- LUT build 時は MaterialX `DielectricBsdf::eval()` を uniform hemisphere 64 samples で積分し、GGX分布/Smith masking/Fresnel を含む反射 directional albedo を事前計算する。
- `runtime.rs` の MaterialX Dielectric `directional_albedo_idx()` と `light_tree_summary_idx()` は、thin filmなしの場合にこのLUTを使う。
- `profile` には closure variant別の `dalbedo_ms` を追加済みで、Dielectric directional albedo の時間を直接確認できる。

確認:

- `cargo test`: OK。lib 714 tests + main 2 tests passed。
- 512x512, 128spp の目視では破綻なし。
- baseline scene45 512/128 との RMSE は `3974.99 (0.0606544)`。
- diff画像: `result/perf/diff_opt16_vs_baseline_scene45_512_spp128.png`

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt16_mtlx_dielectric_directional_albedo_lut_scene45_512_spp128.png` | 00m:18s:149ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt16_mtlx_dielectric_directional_albedo_lut_scene41_512_spp128.png` | 00m:06s:800ms |

profile測定:

`cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --profile-mtlx -o result/perf/profile_opt16_mtlx_dielectric_directional_albedo_lut_scene45_512_spp128.png`

結果:

- render: 00m:23s:692ms
- bytecode: 102069ms / 101,995,291 calls / 4,507,885,384 instrs
- sample: 26508ms / 28,487,420 calls
- eval_pdf: 54125ms / 38,119,099 calls
- light_tree_precompute: 9488ms / 27,531,165 calls
- opacity: 1624ms / 54,359,885 calls
- closure-walker dalbedo: Dielectric 59,745,546 calls / 4645ms、Sheen 3,068,429 calls / 182ms

観察:

- scene45/scene41 比率は `18.149 / 6.800 = 2.67x` で、目標の1.4x以内にはまだ届いていない。
- Fresnel単体近似より Dielectric directional albedo 単体の `dalbedo_ms` は増えたが、通常renderは opt13 の 18.531s、opt15 の 18.790s より少し速い。
- LUT化対象は実際に hot な Dielectric directional albedo であり、対象選択は妥当。
- ただし `dalbedo_ms` はまだ 4.6s 規模で、`eval_pdf` も 54.1s と大きい。次の構造的改善は、同一 shading vertex / 同一 wo で sample、eval_pdf、light-tree summary が繰り返し読む `directional_albedo_idx()` の結果共有を検討する。

判断:

- Fresnel単体近似より物理的に筋のよい MaterialX Dielectric directional albedo へ戻せており、速度も悪化していないため採用して先へ進む。
- 次は `directional_albedo_idx()` の per-shading-vertex cache / precompute で、LUT lookupを含む再帰 traversal の重複を削る。

## 22. Investigation: MaterialX directional albedo LUT coverage

ユーザー確認事項:

- 閉形式で解けない MaterialX ローブで、Layer などに必要な directional albedo がすべてLUT化されているか。

コード棚卸し:

| MaterialX closure | 現状 | 判断 |
| --- | --- | --- |
| OrenNayarDiffuse | `color * weight` | roughness / wo 依存を無視しており、厳密ではない。diffuseなので影響は比較的小さいが、Oren-Nayar directional albedoとしては未LUT。 |
| BurleyDiffuse | `color * weight` | diffuse lobeの近似。Burleyのretro-reflectionを含むdirectional albedoではない。 |
| Translucent | `color * weight` | Lambert transmission相当なら閉形式扱いでよい可能性がある。 |
| Dielectric | thin filmなしは `MtlxDielectricGgxDirectionalAlbedoLut` | 今回LUT化済み。thin filmありはfallbackで未LUT。 |
| Conductor | `fresnel(cos_o) * weight` | GGX分布/Smith masking/roughness/anisotropyを含まない。未LUT。 |
| GeneralizedSchlick | `fresnel(cos_o) * weight` | GGX分布/Smith masking/roughness/anisotropyを含まない。未LUT。 |
| Sheen | `SheenDirectionalAlbedoLut` | LUT化済み。 |
| ChiangHair | `(tint_r + tint_tt + tint_trt) / 3` | hair scatteringのdirectional albedoではなく、かなり粗い近似。未LUT。 |

結論:

- 「Layerなどで directional albedo が必要な MaterialX ローブすべて」はまだLUT化できていない。
- 現在LUT化済みなのは hot path として確認済みの Sheen と Dielectric。
- 次に優先すべきは `directional_albedo_idx()` profileで出現している Dielectric 以外のローブの頻度/時間を追加で測り、Conductor / GeneralizedSchlick / OrenNayar / Burley / ChiangHair の順に、LUT化が必要なものを決めること。
- `Conductor` と `GeneralizedSchlick` は今の `F(cos_o)` 近似が特に雑なので、Dielectricと同じ方針で `eval()` 積分LUTを作る候補。
- `OrenNayarDiffuse` / `BurleyDiffuse` は閉形式または低次元LUTで置き換え可能かを確認する。
- `ChiangHair` はパラメータ次元が高いため、まず scene45 で本当にLayer/pdfに効いているかprofile確認してから設計する。

## 23. Optimization 17: align MaterialX layer throughput with layerable BSDFs and add GeneralizedSchlick LUT

調査:

- MaterialX 仕様書 `documents/Specification/MaterialX.PBRSpec.md` の `Layering` / `layer` では、vertical layering の対象として `dielectric_bsdf`、`generalized_schlick_bsdf`、`sheen_bsdf` が挙げられている。
- `conductor_bsdf` は layerable BSDF として挙げられていない。
- 公式 GLSL ShaderGen では `mx_layer_bsdf.glsl` が `top.response + base.response * top.throughput` を使う。
- `mx_conductor_bsdf.glsl` は冒頭で `bsdf.throughput = vec3(0.0)` とし、その後 throughput を更新していない。従って conductor top は base を透過しない。
- `mx_burley_diffuse_bsdf.glsl`、`mx_oren_nayar_diffuse_bsdf.glsl`、`mx_translucent_bsdf.glsl`、`mx_chiang_hair_bsdf.glsl` も throughput を `0` のままにしている。`chiang_hair` は仕様書でも vertical layering を support しないと書かれている。
- `dielectric_bsdf`、`generalized_schlick_bsdf`、`sheen_bsdf` は ShaderGen 側で directional albedo から throughput を計算している。

結論:

- `directional_albedo_idx()` は単なる反射色ではなく、MaterialX layer composition 用の `1 - throughput` として扱う必要がある。
- layerable でない BSDF を top に置いた場合は、雑な `F(cos_o)` や `color * weight` を返して base へ光を漏らすのではなく、`Vec3::ONE` を返して base を遮断するのが MaterialX ShaderGen の throughput と整合する。
- LUT が必要なのは現時点では `Dielectric`、`GeneralizedSchlick`、`Sheen`。`Dielectric` と `Sheen` は済みなので、`GeneralizedSchlick` を追加対象にした。

実装:

- `DirectionalAlbedoCache` に scene-owned `MtlxGeneralizedSchlickGgxDirectionalAlbedoLut` を追加。
- `Scene::add_material()` で MaterialX material に `sheen_lut`、`mtlx_dielectric_lut`、`mtlx_generalized_schlick_lut` を注入。
- `MtlxGeneralizedSchlickGgxDirectionalAlbedoLut` は `sqrt(cos_o) x alpha_eq` の2D LUT。MaterialX GLSL の `mx_ggx_dir_albedo_monte_carlo()` と同じ AB 係数形式を64 samplesで事前計算し、runtime の `color0/color90` と合成する。
- `directional_albedo_idx()` の `GeneralizedSchlick` branch を LUT 参照へ変更。thin film ありは既存 fallback を維持。
- `light_tree_summary_idx()` の `GeneralizedSchlick` branch も同じ LUT helper を使う。
- `OrenNayarDiffuse`、`BurleyDiffuse`、`Translucent`、`Conductor`、`ChiangHair`、`GoochShade` は layer composition 用 directional albedo として `Vec3::ONE` を返すようにした。
- `Layer` の eval/sample/pdf/eval_pdf/light_tree はすべて `directional_albedo_idx()` 経由なので、LightTreeだけでなく全Layer経路が同じ判定を使う。

確認:

- `cargo check`: OK。
- `cargo test`: OK。lib 715 tests + main 2 tests passed。
- 追加テスト: `layer_with_non_layerable_conductor_top_blocks_base`。Conductor top の Layer が base diffuse を漏らさず、direct conductor と同じ eval になることを確認。
- MaterialX spec test の `Add` / `Mix` の directional albedo 期待値は、diffuse closure を throughput=0 と扱う仕様寄せに合わせて `Vec3::ONE` へ更新。
- 512x512, 128spp の目視では破綻なし。
- baseline scene45 512/128 との RMSE は `3958.89 (0.0604088)`。
- diff画像: `result/perf/diff_opt17_vs_baseline_scene45_512_spp128.png`

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt17_mtlx_layerable_throughput_scene45_512_spp128.png` | 00m:18s:620ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt17_mtlx_layerable_throughput_scene41_512_spp128.png` | 00m:06s:720ms |

profile測定:

`cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --profile-mtlx -o result/perf/profile_opt17_mtlx_layerable_throughput_scene45_512_spp128.png`

結果:

- render: 00m:22s:630ms
- bytecode: 106226ms / 102,005,799 calls / 4,508,752,666 instrs
- sample: 20479ms / 28,491,722 calls
- eval_pdf: 47688ms / 38,125,389 calls
- light_tree_precompute: 19678ms / 27,534,946 calls
- opacity: 1694ms / 54,358,016 calls
- closure-walker dalbedo: Dielectric 85,869,322 calls / 6838ms、Sheen 4,241,919 calls / 267ms

観察:

- scene45/scene41 比率は `18.620 / 6.720 = 2.77x`。
- opt16 の light-tree summary shortcut は速かったが、Layer composition と同じ `directional_albedo_idx()` を使っておらず、今回仕様整合を優先して戻したため `light_tree_precompute` と dalbedo visits が増えた。
- Directional Albedo が必要な layerable ローブは LUT 化済みになった。
- ここからは directional albedo そのものではなく、同一 shading vertex / 同一 `wo` で Layer の eval/sample/pdf/light_tree が同じ closure tree を何度も再帰走査している構造が次の大きなホットパス。

判断:

- 正しさ修正として採用。
- この時点をチェックポイントとしてコミットする。
- 次の最適化候補は `directional_albedo_idx()` 結果の per-shading-vertex cache / precompute、または LightTree summary と Layer throughput の共有。

## 24. Checkpoint commit

チェックポイント:

- commit: `06c0b76 Optimize MaterialX material evaluation`
- 対象: MaterialX bytecode/profile改善、eval_pdf統合、lazy light sampling、Sheen / Dielectric / GeneralizedSchlick directional albedo LUT、MaterialX layerable throughput修正。
- `result/` は `.gitignore` 対象のため、この作業ログとレンダリング画像はコミット対象外。

## 25. Optimization 18: cache MaterialX directional albedo per shading vertex

調査:

- opt17 profile では `directional_albedo_idx()` の hot lobe は LUT化済みだったが、呼び出し回数がまだ多かった。
- `sample_closure_idx()`、`eval_closure_idx()`、`pdf_closure_idx()`、`eval_pdf_closure_idx()`、`light_tree_summary_idx()` の Layer / Add は同じ shading vertex と同じ `wo` で同じ closure subtree の directional albedo を何度も再帰計算していた。
- opt17 profile:
  - `eval_pdf`: 47688ms
  - `light_tree_precompute`: 19678ms
  - dalbedo: Dielectric 85,869,322 calls / 6838ms、Sheen 4,241,919 calls / 267ms

実装:

- `MtlxScratch` に `dalbedo_pool: Vec<Cell<Option<Vec3>>>` を追加。
- `ScratchCheckpoint` / `restore()` に dalbedo pool の top を追加し、regs / matrix pool と同じ lifetime で巻き戻す。
- `ShadingVertex` に `mtlx_dalbedo: Option<DalbedoHandle>` を追加。
- `MtlxMaterial::precompute_shading()` で closure node数分の dalbedo cache を確保。
- `sample` / `eval` / `pdf` / `eval_pdf` / `light_tree_precompute` の MaterialX runtime 呼び出しに cache slice を渡す。
- `directional_albedo_idx()` は cache hit なら即 return、missなら計算後に node index へ保存する。
- 既存の runtime public wrapper は cacheなしでも動くよう残し、テストや直接呼び出しは従来通り動く。

確認:

- `cargo check`: OK。
- `cargo test material::mtlx_material::tests::`: OK。8 tests passed。
- `cargo test`: OK。lib 715 tests + main 2 tests passed。
- 512x512, 128spp の目視では破綻なし。
- baseline scene45 512/128 との RMSE は `3981.48 (0.0607534)`。
- diff画像: `result/perf/diff_opt18_vs_baseline_scene45_512_spp128.png`

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt18_mtlx_dalbedo_cache_scene45_512_spp128.png` | 00m:17s:977ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt18_mtlx_dalbedo_cache_scene41_512_spp128.png` | 00m:06s:598ms |

profile測定:

`cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --profile-mtlx -o result/perf/profile_opt18_mtlx_dalbedo_cache_scene45_512_spp128.png`

結果:

- render: 00m:20s:685ms
- bytecode: 103631ms / 101,986,405 calls / 4,507,580,525 instrs
- sample: 12248ms / 28,483,524 calls
- eval_pdf: 29424ms / 38,110,059 calls
- light_tree_precompute: 17052ms / 27,526,792 calls
- opacity: 1637ms / 54,350,063 calls
- closure-walker dalbedo: Dielectric 26,990,158 calls / 2205ms、Sheen 1,211,192 calls / 81ms

観察:

- scene45 render は opt17 の 18.620s から 17.977s へ改善。
- scene45/scene41 比率は `17.977 / 6.598 = 2.72x`。
- dalbedo call は大幅に減ったが、まだ `bytecode` が約103s、`eval_pdf` が約29s、`light_tree_precompute` が約17s と大きい。
- 次の主なホットパスは directional albedo そのものではなく、MaterialX bytecode実行と closure eval_pdf traversal。

判断:

- 改善があり、仕様上の挙動も変えない cache なので採用して先へ進む。
- 次は bytecode 内の opcode / texture / param read の粒度で、どの命令が支配的かを測る。

## 26. Profile 19: MaterialX bytecode instruction breakdown

目的:

- opt18 時点で `bytecode` が最大級の集計 hot spot として残っている。
- 既存 profile は bytecode 全体時間しか出ないため、MaterialX runtime のどの命令が効いているかを見られない。

実装:

- `--profile-mtlx` 有効時だけ、`run_instructions()` 内で各 `Instruction` の実行時間と呼び出し回数を集計する profile を追加。
- 通常 render では命令別 `Instant::now()` を走らせない分岐にしたため、通常の速度計測には影響させない。
- `main.rs` の profile 出力に `[profile] bytecode instructions:` を追加。

確認:

- `cargo check`: OK。

profile測定:

`cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --output result/perf/profile_opt19_opcode_scene45_512_spp128.png --profile-mtlx`

結果:

- render: 01m:05s:983ms
- bytecode: 1,231,500ms / 101,987,215 calls / 4,507,710,471 instrs
- sample: 10,107ms / 28,486,612 calls
- eval_pdf: 22,346ms / 38,117,672 calls
- light_tree_precompute: 15,339ms / 27,528,603 calls
- opacity: 1,449ms / 54,350,520 calls

命令別上位:

| instruction | calls | total | per call |
| --- | ---: | ---: | ---: |
| Image | 306,834,047 | 55,753ms | 182ns |
| Arith | 1,526,815,619 | 47,515ms | 31ns |
| MixValue | 421,949,371 | 13,464ms | 32ns |
| LoadConst | 558,946,370 | 12,198ms | 22ns |
| LoadGeom | 396,172,183 | 11,054ms | 28ns |
| Extract | 298,724,128 | 9,170ms | 31ns |
| Unary | 165,530,406 | 5,415ms | 33ns |
| Clamp | 146,256,650 | 4,587ms | 31ns |
| HextiledImage | 8,771,990 | 4,584ms | 523ns |
| Rotate3d | 112,859,246 | 3,956ms | 35ns |

観察:

- 命令単位で `Instant::now()` と atomic 加算をしているため、この profile の render 時間は通常 render の比較には使わない。
- ただし相対内訳としては `Image` が最大で、テクスチャサンプリングまたはテクスチャ関連の address / default / type変換が次の有力 hot path。
- `Arith` は単価は低いが呼び出しが 15億回を超えており、bytecode の命令数そのものも構造的に効いている。
- `HextiledImage` は回数は `Image` より少ないが単価が高く、専用に見る価値がある。

判断:

- 次は一時的に MaterialX のテクスチャ評価を無効化し、scene45 の通常 render がどの程度改善するかを測る。
- 改善幅が十分なら、テクスチャサンプリング、addressing、texcoord計算、同一 shading vertex 内の texture sample 共有、または低解像度 texture 実験へ進む。

## 27. Experiment 20: bypass MaterialX texture sampling

目的:

- 命令別 profile で `Image` と `HextiledImage` が上位に出たため、MaterialX テクスチャ評価を一時的に無効化した場合の上限改善幅を測る。

一時変更:

- `Image` は texture sample を呼ばず、入力の `default` を返す。
- `HextiledImage` は `default_color` を返す。
- `HextiledNormalMap` は default normal を返す。
- この変更は正しいレンダリングを目的にしたものではなく、切り分け専用。

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/experiment_opt20_bypass_mtlx_textures_scene45_512_spp128.png` | 00m:15s:057ms |

profile測定:

`cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --output result/perf/profile_opt20_bypass_mtlx_textures_scene45_512_spp128.png --profile-mtlx`

結果:

- render: 01m:02s:533ms
- bytecode: 1,149,742ms / 103,199,890 calls / 4,554,607,881 instrs
- sample: 10,465ms / 28,543,510 calls
- eval_pdf: 21,938ms / 36,059,552 calls
- light_tree_precompute: 16,083ms / 27,615,223 calls
- opacity: 1,379ms / 53,099,024 calls

命令別変化:

| instruction | opt19 total | bypass total | 観察 |
| --- | ---: | ---: | --- |
| Image | 55,753ms | 6,916ms | sample本体を消すと大きく低下 |
| HextiledImage | 4,584ms | 218ms | sample本体を消すとほぼ消える |
| Arith | 47,515ms | 47,069ms | テクスチャとは独立に残る |

観察:

- 通常 render では opt18 の 17.977s から 15.057s へ改善。テクスチャ評価の上限改善幅は約 2.9s / 16%。
- `Image` / `HextiledImage` は確かに hot path だが、scene45 と scene41 の 2.7x 差を単独で説明するほどではない。
- テクスチャを消しても `Arith`、`MixValue`、`LoadConst`、`LoadGeom` など bytecode 命令数そのものが残る。

判断:

- テクスチャ最適化は採用候補だが、これだけでは不十分。
- 一時バイパス変更は戻し、まず安全な sampler 内部改善を試す。

## 28. Optimization 21: fast power-of-two texture wrapping

調査:

- MaterialX scene45 の texture は 4k や 512 など power-of-two 寸法が多い可能性が高い。
- `Texture::bilerp_level()` は 1 sample あたり 4 texel を読み、各 texel 読みで `wrap_index()` が `rem_euclid()` を実行していた。
- power-of-two サイズでは modulo は bitmask で置き換えられる。

実装:

- `wrap_index(index, size)` で `size.is_power_of_two()` の場合は `(index as usize) & (size - 1)` を使う。
- 非 power-of-two サイズは従来通り `rem_euclid()` を使う。
- 一時的な texture bypass は戻した。

確認:

- `cargo check`: OK。
- `cargo test material::texture::tests::`: OK。11 tests passed。
- baseline scene45 512/128 との RMSE は `3969.62 (0.0605726)`。
- diff画像: `result/perf/diff_opt21_vs_baseline_scene45_512_spp128.png`

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt21_texture_pow2_wrap_scene45_512_spp128.png` | 00m:17s:720ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt21_texture_pow2_wrap_scene41_512_spp128.png` | 00m:06s:855ms |

観察:

- scene45 は opt18 の 17.977s から 17.720s へ小幅改善。
- scene45/scene41 比率は `17.720 / 6.855 = 2.58x`。
- 改善幅は小さいが、texture sampling の hot path にあり、power-of-two 以外の挙動も保持するため採用する。

判断:

- 採用して先へ進む。
- 次は bytecode の `Arith` / `MixValue` / `LoadConst` / `LoadGeom` の大量実行を減らす方向、または closure eval_pdf/light_tree traversal の構造改善を見る。

## 29. Rejected optimization 22: vector fast path in runtime arith

目的:

- profile で `Arith` が 15億回規模で実行されていたため、`arith()` の単価を下げる。

実装:

- `ArithOp::Add` / `Subtract` / `Multiply` / `Divide` / `Min` / `Max` を `Vec2` / `Vec3` / `Vec4` のベクトル演算へ分岐。
- `Modulo` / `Power` / `SafePower` / `Atan2` は従来通り component ごとに計算。

確認:

- `cargo check`: OK。
- `cargo test material::mtlx::spec_tests::`: OK。211 tests passed。

測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt22_arith_vector_fastpath_scene45_512_spp128.png` | 00m:18s:022ms |

判断:

- opt21 の 17.720s より悪化。
- 追加分岐や関数分割のコストが勝っていないため revert。

## 30. Rejected optimization 23: runtime zero-weight early return in eval_pdf

目的:

- static zero weight は compile時に closure simplification で消えるが、texture / pattern 由来で runtime zero になる場合は各 BSDF の構築と eval/pdf が残る。
- `eval_pdf` の weight付きローブで weight が 0 の場合に早期 return して、closure traversal の無駄を減らせるか確認する。

実装:

- `OrenNayarDiffuse` / `BurleyDiffuse` / `Translucent` / `Dielectric` / `Conductor` / `GeneralizedSchlick` / `Sheen` の `eval_pdf` で `weight.abs() <= 1e-8` なら `(Vec3::ZERO, 0.0)` を返す。

確認:

- `cargo check`: OK。
- `cargo test material::mtlx_material::tests::`: OK。8 tests passed。

測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt23_eval_pdf_zero_weight_scene45_512_spp128.png` | 00m:19s:780ms |

判断:

- 大きく悪化したため revert。
- scene45 では runtime zero weight が支配的ではなく、追加分岐と weight read のコストだけが増えた可能性が高い。

## 31. Rejected optimization 24: compile-time arithmetic folding

目的:

- `Arith` 命令数そのものを減らすため、compile時に両辺定数の算術と `x + 0` / `x * 1` / `x / 1` / `0 * x` を畳み込む。

実装:

- `Builder::emit_arith()` で `Operand::Const` を見て、両辺定数なら `super::runtime::arith()` で結果を `value_pool` へ入れる。
- 恒等演算は既存 operand を返すことで `Arith` 命令を発行しない。

確認:

- `cargo check`: OK。
- `cargo test material::mtlx::spec_tests::`: OK。211 tests passed。

測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt24_compile_arith_folding_scene45_512_spp128.png` | 00m:17s:875ms |

compile probe:

- `result/perf/profile_opt24_compile_arith_folding_compile_probe.png` を 1x1, 1spp, `--profile-mtlx` で出力。
- 命令数の減少は多くの material で 1 命令程度、`Car_Paint` は 115 命令のまま。
- `value_pool` は増えていた。

判断:

- 効果が小さく、通常renderも opt21 より悪化気味なので revert。
- `Arith` の総量は単純な定数畳み込みでは減らず、より上流の nodegraph 構造や頻出 material の bytecode 全体を見直す必要がある。

## 32. Optimization 25: share geometric loads inside MaterialX bytecode

調査:

- `Car_Paint.mtlx` などの頻出 material では、同じ `texcoord`、world normal、world tangent が別ノードとして複数回出ていた。
- 既存 compiler は `FlatInput::GeomProp` 経由では geometric local を共有していたが、`FlatNodeKind::Geometric` ノードは node id ごとに `LoadGeom` を発行していた。
- opt19 profile では `LoadGeom` が 396,172,183 calls / 11,054ms と上位だった。

実装:

- `ensure_geometric_kind_local(GeometricKind)` を追加。
- geometric kind と space を合成した synthetic key を `register_for` に入れ、同一 material 内の同じ geometric input を共有する。
- `FlatNodeKind::Geometric`、`geompropvalue`、`frame`、`time`、default normal/tangent/bitangent 補完などの `emit_load_geom()` 呼び出しを共有版に置き換えた。

確認:

- `cargo check`: OK。
- `cargo test material::mtlx::spec_tests::`: OK。211 tests passed。
- compile probe: `result/perf/profile_opt25_geometric_cse_compile_probe.png`
- baseline scene45 512/128 との RMSE は `3966.59 (0.0605263)`。
- diff画像: `result/perf/diff_opt25_vs_baseline_scene45_512_spp128.png`

compile probe 観察:

| material | before | after |
| --- | ---: | ---: |
| Car_Paint | 115 instrs | 109 instrs |
| Bronze_Oxydized | 75 instrs | 72 instrs |
| Emerald_Peaks_Wallpaper | 49 instrs | 46 instrs |
| Argentinian_Layered_Onyx | 35 instrs | 33 instrs |
| common 35-instr standard materials | 35 instrs | 33 instrs |

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt25_geometric_cse_scene45_512_spp128.png` | 00m:17s:637ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt25_geometric_cse_scene41_512_spp128.png` | 00m:06s:784ms |

観察:

- scene45 は opt21 の 17.720s から 17.637s へ小幅改善。
- scene45/scene41 比率は `17.637 / 6.784 = 2.60x`。
- 命令数削減は小さいが、同一 geometric 入力の重複計算削減として構造的に正しい。

判断:

- 採用して先へ進む。
- 次は `Car_Paint` のような頻出 material で texcoord 変換や image sample の重複共有ができるかを調べる。

## 33. Experiment 26: force MaterialX texture sampling to 512px mip

目的:

- ユーザー指摘の通り、4k texture サイズそのものが速度差の主因かを確認する。
- 実ファイルは変更せず、`Texture::sample_mip_bilinear()` で一時的に最小 mip level を 512px 相当へ押し上げた。

一時変更:

- `DEBUG_FORCE_MAX_MTLX_SAMPLE_DIM = 512` を入れ、sample level の width / height が 512 を超える場合は次の mip へ進める。
- `sample_mip_bilinear()` を使う MaterialX image / hextiled image sampling のみが主対象。

確認:

- `cargo check`: OK。
- baseline scene45 512/128 との RMSE は `4016.43 (0.0612868)`。
- diff画像: `result/perf/diff_opt26_force_512_vs_baseline_scene45_512_spp128.png`

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/experiment_opt26_force_512_mtlx_textures_scene45_512_spp128.png` | 00m:17s:492ms |

観察:

- opt25 の 17.637s から 17.492s へ約0.15s改善。
- texture size / cache locality の影響はあるが、scene45 と scene41 の大きな差を説明するほどではない。
- RMSE は opt25 より少し増えており、通常の品質設定としては雑に採用するほどの速度幅ではない。

判断:

- 一時変更は revert。
- テクスチャの主因はサイズそのものより、Image命令の呼び出し回数と bytecode 全体の繰り返し評価。

## 34. Rejected optimization 27: unit-UV image sampler path

目的:

- `Image` 命令では `apply_address_modes()` で UV を unit 範囲へ畳んだ後、`Texture::sample_mip_bilinear()` 側でも `wrap_unit()` を再実行している。
- `Image` 専用に unit-UV 前提の sampler を作れば、`wrap_unit()` の `rem_euclid()` を避けられるか確認する。

実装:

- `Texture::sample_mip_bilinear_unit()` と `bilerp_level_unit()` を追加。
- `sample_image_texture()` の通常 Image / UDIM path から unit sampler を呼ぶように変更。
- Hextiled は address 適用済みではないため対象外。

確認:

- `cargo test material::texture::tests::`: OK。11 tests passed。

測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt27_image_unit_uv_sampler_scene45_512_spp128.png` | 00m:18s:120ms |

判断:

- opt25 の 17.637s から悪化したため revert。
- 関数分割や追加 path による最適化阻害が、削った `wrap_unit()` より大きい可能性が高い。

## 35. Checkpoint before next hotspot search

目的:

- Directional Albedo の必要箇所の LUT 化と、Directional Albedo 以外で確認できた軽量化をいったんチェックポイント化する。
- ここから先は Directional Albedo 以外のより重たいホットパス探索へ戻る。

採用済みの内容:

- per shading vertex の MaterialX directional albedo cache。
- `--profile-mtlx` の bytecode 命令別プロファイル出力。
- power-of-two texture wrap の bitmask 化。
- MaterialX compile 時の geometric load CSE。

検証:

- `cargo check`: OK。
- `cargo test material::mtlx::spec_tests::`: OK。211 tests passed。
- `cargo test material::texture::tests::`: OK。11 tests passed。
- `cargo test`: OK。715 lib tests, 2 bin tests, 0 doctests passed。

測定:

| scene | 条件 | render |
| --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | 00m:17s:637ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | 00m:06s:784ms |

判断:

- scene45/scene41 比率は `17.637 / 6.784 = 2.60x` で 1.4x には届いていない。
- ただし、Directional Albedo cache と geometric load CSE は構造的に無駄な重複評価を消す内容なので採用する。
- 次は Directional Albedo 以外、特に bytecode 実行回数、texture/image sampling、opacity/any-hit 周辺、頻出 material の graph 評価を再プロファイルする。

commit:

- `b46a99c Optimize MaterialX bytecode evaluation`

## 36. Profile 28: post-checkpoint MaterialX bytecode hotspot

目的:

- Directional Albedo の必要箇所が LUT / cache 化された後に、残っている MaterialX の重い箇所を取り直す。
- scene41 とは同時実行せず、scene45 の profile のみ単独で実行した。

実行:

- `cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --output result/perf/profile_opt35_scene45_512_spp128.png --profile-mtlx`

結果:

- `render: 01m:04s:863ms`
- profile mode は per-instruction `Instant` を含むため通常 render より大幅に遅い。通常速度の比較には使わず、相対的なホットスポット把握に使う。

MaterialX closure / phase:

| phase | total | calls |
| --- | ---: | ---: |
| bytecode | 1166445ms | 101976534 |
| sample | 13774ms | 28479374 |
| eval_pdf | 30471ms | 38105405 |
| light_tree_precompute | 18582ms | 27522381 |
| opacity | 1488ms | 54349383 |

Material 別上位:

| slot | name | calls | total | per_call | avg_instrs |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Car_Paint | 7584545 | 229653ms | 30279ns | 109.0 |
| 2 | Copper_Satin | 7997944 | 166001ms | 20755ns | 74.0 |
| 10 | Bronze_Oxydized | 5573133 | 107570ms | 19301ns | 72.0 |
| 11 | material_checker_opacity | 7316431 | 84024ms | 11484ns | 45.0 |
| 3 | Emerald_Peaks_Wallpaper | 6641182 | 82109ms | 12364ns | 46.0 |

Bytecode opcode 上位:

| opcode | calls | total | per_call |
| --- | ---: | ---: | ---: |
| Image | 306801401 | 55166ms | 180ns |
| Arith | 1526636292 | 47492ms | 31ns |
| MixValue | 421930825 | 13598ms | 32ns |
| LoadConst | 558918275 | 12209ms | 22ns |
| Extract | 298701281 | 8925ms | 30ns |
| LoadGeom | 219219155 | 6701ms | 31ns |
| HextiledImage | 8779028 | 4259ms | 485ns |

観察:

- `LoadGeom` は opt25 前の 396M calls から 219M calls へ減った。geometric CSE は命令数削減としては効いている。
- それでも scene45 の主因は Directional Albedo ではなく、bytecode graph 自体の再実行回数と、その中の `Image` / `Arith` 呼び出し回数。
- texture bypass 実験で Image の理論上限は約3秒程度と見えているため、texture sampling だけでは 1.4x には届かない。
- `opacity` phase 自体は 1.5s 程度だが、opacity 判定のために full material graph を precompute している可能性があるため、any-hit / opacity 周辺は構造確認が必要。

次の調査:

- `any_hit()` / `precompute_shading()` / `opacity()` の呼び出し関係を確認し、opacity だけを見る場面で full BSDF graph を評価していないか調べる。
- 頻出 material の bytecode を見て、`Image` / `Arith` / `MixValue` のうち compile-time または shading-vertex-time に共有できる値が残っていないか確認する。

## 37. Optimization 29: opacity-only bytecode for alpha test

調査結果:

- `Scene::closest_hit()` は alpha test 対象 material で候補 hit ごとに `material.any_hit()` を呼ぶ。
- `MtlxMaterial::any_hit()` は opacity だけを知りたい場合でも `precompute_shading()` を呼び、full material graph の bytecode を実行してから `runtime::opacity()` を呼んでいた。
- profile の `opacity` phase 自体は 1.5s 程度だったが、その前段の full bytecode 実行は material 別 total / bytecode total に含まれていた。
- `material_checker_opacity` は profile material 上位に入っており、any-hit alpha test で full graph を回す構造は無駄が大きい。

実装:

- compile 時に通常 bytecode とは別に opacity 専用 bytecode を生成。
- opacity 専用 dead-code elimination は `Surface.opacity`、surface `Mix.mix`、`Layer` の子だけを live-out として扱う。
- `MtlxMaterial::any_hit()` は `precompute_shading()` を呼ばず、opacity 専用 register region に `run_opacity_instructions()` を実行して `opacity_for_alpha_test()` を評価する。
- 通常 shading path の `sample` / `eval` / `eval_pdf` / `light_tree_precompute` は従来通り full precompute 済み register を使う。

検証:

- `cargo check`: OK。
- `cargo test material::mtlx::spec_tests:: --lib`: OK。211 tests passed。
- `cargo test material::mtlx_material::tests:: --lib`: OK。8 tests passed。
- baseline scene45 512/128 との RMSE は `3960.68 (0.0604362)`。
- diff画像: `result/perf/diff_opt36_opacity_bytecode_vs_baseline_scene45_512_spp128.png`

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt36_opacity_bytecode_scene45_512_spp128.png` | 00m:15s:684ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt36_opacity_bytecode_scene41_512_spp128.png` | 00m:06s:797ms |

効果:

- opt25 の scene45 17.637s から 15.684s へ、約1.95s改善。
- scene45/scene41 比率は `15.684 / 6.797 = 2.31x`。

判断:

- 採用して先へ進む。
- Directional Albedo 以外の構造的ホットパスとして、alpha test / any-hit の full graph precompute は大きく効いていた。
- 次は opt36 後の profile を取り直し、残った bytecode call の内訳を見る。

## 38. Profile 30: after opacity-only bytecode

目的:

- opt36 後に bytecode hot spot がどこへ移ったかを確認する。
- scene41 とは同時実行せず、scene45 profile のみ単独実行。

実行:

- `cargo run --release -- --scene 45 --width 512 --height 512 --spp 128 --output result/perf/profile_opt36_scene45_512_spp128.png --profile-mtlx`

結果:

- `render: 00m:37s:315ms`
- profile mode のため通常 render とは比較しない。

compile 時の確認:

- 多くの material は full bytecode が 10-109 命令なのに対し、opacity bytecode は 2 命令。
- `material_checker_opacity` は full 45 命令、opacity 13 命令。

MaterialX closure / phase:

| phase | total | calls |
| --- | ---: | ---: |
| bytecode | 521391ms | 101980280 |
| sample | 11010ms | 28482799 |
| eval_pdf | 26222ms | 38112185 |
| light_tree_precompute | 15916ms | 27525401 |
| opacity | 1479ms | 54348145 |

Bytecode opcode 上位:

| opcode | calls | total | per_call |
| --- | ---: | ---: | ---: |
| Image | 150201067 | 32057ms | 213ns |
| Arith | 728362147 | 23604ms | 32ns |
| MixValue | 203686451 | 6699ms | 33ns |
| Extract | 208986613 | 6329ms | 30ns |
| LoadConst | 275109759 | 6208ms | 23ns |
| LoadGeom | 106388275 | 3493ms | 33ns |
| HextiledImage | 4336650 | 2634ms | 607ns |

観察:

- opt35 profile の `4330367602 instrs` から opt36 profile の `2202480483 instrs` へ、実行命令数がほぼ半減。
- bytecode calls 数はほぼ同じ。any-hit はまだ候補 hit ごとに呼ばれるが、実行している命令列だけが短くなった。
- `Image` / `Arith` / `MixValue` はまだ上位だが、残りは主に通常 shading hit 側の full graph precompute で発生している。

次の調査:

- integrator 側で同じ hit の `precompute_shading()` を二重に実行していないかを確認する。
- full graph precompute が本当に必要な query と、le/opacity など部分 graph で済む query を分けられるか調べる。

## 39. Rejected optimization 31: shadow visibility before BSDF eval

目的:

- direct light contribution では現在 `eval_pdf` / `eval` の後に shadow ray の `unoccluded()` を呼んでいる。
- 遮蔽される light sample では BSDF 評価が無駄になるため、cos チェックと `unoccluded()` を先に行う実験をした。

実装:

- `integrator::mis::direct_light_mis_contribution()` で `eval_pdf` / `eval` の前に cos チェックと `unoccluded()` を移動。
- `integrator::nee::direct_light_nee_contribution()` も同様に `eval` の前へ移動。

検証:

- `cargo test integrator:: --lib`: OK。32 tests passed。
- baseline scene45 512/128 との RMSE は `3980.92 (0.060745)`。
- diff画像: `result/perf/diff_opt37_shadow_before_eval_vs_baseline_scene45_512_spp128.png`

測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt37_shadow_before_eval_scene45_512_spp128.png` | 00m:15s:805ms |

判断:

- opt36 の 15.684s から小幅悪化したため revert。
- scene45 では遮蔽サンプルで省ける BSDF 評価より、shadow traversal の前倒しや実行順変更の影響が勝っている可能性がある。
- この最適化は採用しない。

## 40. Checkpoint commit after opacity-only bytecode

検証:

- `cargo test`: OK。715 lib tests, 2 bin tests, 0 doctests passed。

commit:

- `4b2e886 Add opacity-only MaterialX bytecode`

次:

- 1.4x にはまだ届いていないため調査継続。
- 残りは full material graph の `Image` / `Arith` / `MixValue` と closure walker の `eval_pdf` / `light_tree_precompute` が中心。

## 41. Rejected optimization 32: per-bytecode texture LOD context

目的:

- MaterialX `Image` hot path では各 texture sample ごとに `sv.uv_dx()` / `sv.uv_dy()` から mip level 用の `log2` を計算している。
- 現在の MaterialX sampler は uv transform を differential に反映していないため、同一 shading vertex の bytecode 実行内では LOD 幅を共有できる。

実装:

- `Texture::sample_mip_bilinear_lod_log2()` を追加。
- `runtime::run_instruction_stream()` で `TextureFilterContext` を作り、`Image` / `HextiledImage` の texture sample に渡した。
- `sample_image_texture()` と `hextiled_color_sample()` で per-sample `log2` を避けるようにした。

検証:

- `cargo check`: OK。
- `cargo test material::texture::tests:: --lib`: OK。11 tests passed。
- `cargo test material::mtlx::spec_tests:: --lib`: OK。211 tests passed。
- baseline scene45 512/128 との RMSE は `3947.64 (0.0602372)`。
- diff画像: `result/perf/diff_opt41_texture_lod_context_vs_baseline_scene45_512_spp128.png`

測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt41_texture_lod_context_scene45_512_spp128.png` | 00m:15s:700ms |

判断:

- opt36 の 15.684s から改善せず、むしろ小幅悪化。
- per-sample `log2` 削減より、追加 context 引数や sampler 分岐の変化による最適化阻害が勝っている可能性がある。
- この最適化は revert する。

## 42. Optimization 33: fold constant-one opacity and skip opaque alpha tests

調査:

- opt36 profile で全 MaterialX material に `opacity_instrs=2` が出ていた。
- `MtlxMaterial::has_alpha_test()` は `compiled.has_opacity_test` を見て scene traversal の candidate hit ごとに `any_hit()` を呼ぶ。
- つまり、実質 opaque な material でも `has_opacity_test=true` になっている場合、candidate hit ごとに opacity-only bytecode を実行してしまう。
- 最初に `LoadConst(1.0)` / `Convert(1.0)` のみを local constant として判定する実装を試したが、profile compile 出力は変わらず `opacity_instrs=2` のままだった。
- 一時 debug 出力で `Argentinian_Layered_Onyx` の opacity bytecode を確認したところ、`LuminanceWithCoeffs(Color3(1))` の後に `Extract` する形だった。

実装:

- `closure_has_opacity_test()` に local constant propagation を追加。
- 対象は `LoadConst`、`Convert`、`LuminanceWithCoeffs`、`Extract`。
- `Surface.opacity` が `ParamRef::Local` でも最終的に 1.0 定数だと分かる場合は opaque とみなし、`has_opacity_test=false` にする。
- profile 用の一時 debug 出力は削除済み。

確認:

- 低解像度 profile compile で、`material_checker_opacity` 以外の scene45 MaterialX material は `opacity_instrs=0` になった。
- `material_checker_opacity` は `opacity_instrs=13` のまま残り、必要な alpha test は維持されている。

検証:

- `cargo test material::mtlx::compile::tests::local_constant_one_opacity_is_not_alpha_test --lib`: OK。
- baseline scene45 512/128 との RMSE は `3947.85 (0.0602404)`。
- diff画像: `result/perf/diff_opt44_opacity_const_fold_vs_baseline_scene45_512_spp128.png`

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt44_opacity_const_fold_scene45_512_spp128.png` | 00m:15s:230ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt44_opacity_const_fold_scene41_512_spp128.png` | 00m:06s:765ms |

効果:

- opt36 の scene45 15.684s から 15.230s へ、約0.45s改善。
- scene45/scene41 比率は `15.230 / 6.765 = 2.25x`。

判断:

- 採用して先へ進む。
- alpha test 対象 material の誤判定は opacity-only bytecode 後にも残っていた構造的な無駄であり、今回の修正で必要な `material_checker_opacity` のみに絞れた。

## 43. Profile 34: NG expansion and bytecode evaluator state after opt44

ユーザー質問:

- scene45 の MaterialX material は本当に重い graph なのか。
- NG 展開で必要以上に深くなっていないか。
- bytecode 実行器そのものの最適化余地は残っているか。

追加した診断:

- `--profile-mtlx` 時のみ、MaterialX load 後に source document の node 数、flatten 後 node 数、上位 flat category を出すようにした。
- 通常 render では出力も処理も走らない。

NG / flatten の観察:

| material | source_nodes | flat_nodes | expansion | bytecode instrs |
| --- | ---: | ---: | ---: | ---: |
| Car_Paint | 62 | 151 | 2.44x | 109 |
| Copper_Satin | 49 | 118 | 2.41x | 74 |
| Bronze_Oxydized | 39 | 117 | 3.00x | 72 |
| material_checker_opacity | 34 | 139 | 4.09x | 45 |
| M_BrickPattern | 29 | 95 | 3.28x | 50 |
| Argentinian_Layered_Onyx | 14 | 81 | 5.79x | 33 |
| Acryl_Plastic | 1 | 65 | 65.00x | 24 |
| Diamond | 1 | 65 | 65.00x | 18 |
| Velvet | 1 | 65 | 65.00x | 10 |

観察:

- standard_surface 直書きの material は source node が 1 でも stdlib implementation 展開で 65 flat nodes になる。
- ただし compile 後の DCE / register allocation で bytecode は 10-24 命令程度まで落ちている。
- textured material でも最大は Car_Paint の 109 命令であり、数百から数千命令級の巨大 graph ではない。
- NG 展開の増幅はあるが、現状の支配要因は「中規模以下の bytecode が非常に高頻度に呼ばれる」こと。

opt44 後 512/128 profile:

- `render: 00m:36s:972ms` (profile overhead 込み、通常速度比較には使わない)
- `bytecode: 563038ms / 52032918 calls (2102624434 instrs)`
- `sample: 10065ms / 28482854`
- `eval_pdf: 20085ms / 38113893`
- `light_tree_precompute: 14941ms / 27524098`
- `opacity: 162ms / 4399838`

Opcode 上位:

| opcode | calls | total | per_call |
| --- | ---: | ---: | ---: |
| Image | 150191465 | 32726ms | 218ns |
| Arith | 728375994 | 23820ms | 33ns |
| MixValue | 203691865 | 6777ms | 33ns |
| LoadConst | 275071444 | 6344ms | 23ns |
| Extract | 159050823 | 5045ms | 32ns |
| LoadGeom | 106399292 | 3486ms | 33ns |

評価:

- 現在の bytecode 実行器は tree walk ではなく register VM で、DCE、linear-scan register allocation、scratch reuse、per-shading-vertex precompute も入っている。素朴な実装ではない。
- ただし、まだ interpreter なので `Instruction` enum dispatch、`Value` enum の tag 判定、`Operand::Reg/Const` 解決、`arith()` 内の動的 type / op 分岐が高頻度に残っている。
- 世の中の bytecode VM 最適化としては、typed bytecode、superinstruction、direct-threaded dispatch、constant propagation、native lowering / JIT などがある。今回特に効きそうなのは typed/specialized instruction と standard_surface 系 native lowering。
- 一方で NG 展開の折り畳みだけでは上限が限られる。すでに DCE 後の bytecode はそこまで巨大ではないため。

次:

- まず最大呼び出し数の `Arith` の evaluator を、op/type の分岐を外側へ出す形に書き換えて命令単価を下げる。
- その後、`standard_surface` / MaterialX BSDF network の native lowering が可能かを検討する。

## 44. Optimization 45: specialize Arith evaluator dispatch

ユーザー質問:

- bytecode 評価ループ自体の最適化は十分か。
- 既存の bytecode VM でよく知られた最適化のうち、未導入のものはあるか。

評価:

- 現在の MaterialX evaluator は tree walk ではなく register VM。
- compile 側には DCE、linear-scan register allocation、geometric load CSE、opacity-only bytecode などが入り、runtime 側も scratch を再利用している。
- profile OFF の通常 render では、`run_instruction_stream()` のループ自体に `Instant::now` や atomic counter は無い。
- 一方で、まだ interpreter なので以下は残っている。
  - `Instruction` enum の命令ディスパッチ。
  - `Operand::Reg/Const` の operand 解決。
  - `Value` tagged enum の型変換。
  - `Arith` などの命令内での動的 type / op 分岐。
- 一般的な bytecode VM 最適化としては typed opcode 化、operand predecode、superinstruction、direct-threaded dispatch、inline cache、hot graph native lowering / JIT などがある。
- Rust の portable な実装として direct-threaded dispatch / JIT は重い。まずは profile 上位で呼び出し回数が最大の `Arith` を typed/specialized opcode に近づけるのが低リスク。

opt44 profile の根拠:

| opcode | calls | total | per_call |
| --- | ---: | ---: | ---: |
| Image | 150191465 | 32726ms | 218ns |
| Arith | 728375994 | 23820ms | 33ns |
| MixValue | 203691865 | 6777ms | 33ns |
| LoadConst | 275071444 | 6344ms | 23ns |
| Extract | 159050823 | 5045ms | 32ns |
| LoadGeom | 106399292 | 3486ms | 33ns |

実装:

- `arith()` の closure-based scalar evaluator を廃止。
- `arith_vec2` / `arith_vec3` / `arith_vec4` / `arith_scalar` に分離。
- `Add` / `Subtract` / `Multiply` / `Divide` / `Min` / `Max` は glam の vector 演算に寄せた。
- `Modulo` / `Power` / `SafePower` / `Atan2` は component-wise 実装を維持。
- `divide` の 0 除算 NaN 挙動、`modulo` の floor 式、`safepower` の符号保持は変更していない。

検証:

- `cargo check`: OK。
- `cargo test material::mtlx:: --lib`: OK、214 tests。
- baseline scene45 512/128 との RMSE は `3946.28 (0.0602163)`。
- diff画像: `result/perf/diff_opt45_arith_specialized_vs_baseline_scene45_512_spp128.png`

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt45_arith_specialized_scene45_512_spp128.png` | 00m:14s:926ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt45_arith_specialized_scene41_512_spp128.png` | 00m:06s:587ms |

効果:

- opt44 の scene45 15.230s から 14.926s へ、約0.30s改善。
- scene45/scene41 比率は `14.926 / 6.587 = 2.27x`。
- scene41 も小幅に速くなっているため比率改善は小さいが、MaterialX scene45 の絶対時間は改善した。

判断:

- 採用して先へ進む。
- bytecode 評価器は最低限のVM最適化は入っているが、「十分」とは言えない。
- 次に効きそうなのは `Image` 命令の sampler 側、または frequent opcode の operand predecode / typed instruction 化。

## 45. Rejected: periodic non-UDIM image fast path

仮説:

- `image` / `tiledimage` の address mode 既定値は MaterialX では `periodic`。
- 非UDIMテクスチャでは下位の `Texture::sample_*` も `wrap_unit()` で periodic wrap する。
- そのため、`Instruction::Image` 側の `apply_address_modes()` と default operand 読みを、非UDIM/非Missingかつ `uaddress=vaddress=periodic` の場合に省略できる可能性がある。

実装実験:

- `ImageTexture::Color` / `ColorAlpha` / `Scalar` の periodic case だけ `sample_wrapping_image_texture()` へ分岐。
- `ImageTexture::Udim` / `Missing` は従来通り default を使う経路に残した。

検証:

- `cargo check`: OK。
- `cargo test material::mtlx:: --lib`: OK、214 tests。

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt46_periodic_image_fast_path_scene45_512_spp128.png` | 00m:15s:959ms |

判断:

- opt45 の 14.926s から大幅悪化。
- 追加分岐と helper 分割により、hot な `Image` 命令の最適化が阻害された可能性が高い。
- この変更は revert 済み。不採用。

## 46. Rejected: inline LoadConst registers into hot operands

仮説:

- VM 最適化として、operand predecode / constant operand 化が考えられる。
- 現状 `LoadConst` が profile で `275015798 calls / 6693ms` 出ているため、`LoadConst` の結果を使う hot opcode の `Operand::Reg` を `Operand::Const` に差し替えると命令数を減らせる可能性がある。

実装実験:

- compile の SSA 後、DCE 前に `LoadConst` の dst register と value_pool index を収集。
- `Arith`、`MixValue`、`Image`、`Extract`、`Clamp`、`Unary`、`Convert` など hot な inline operand と、operand pool 内の operand を `Operand::Const` に差し替えた。
- その後の既存 DCE で不要になった `LoadConst` を消す設計にした。

検証:

- `cargo check`: OK。
- `cargo test material::mtlx:: --lib`: OK、214 tests。

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt47_inline_load_const_operands_scene45_512_spp128.png` | 00m:15s:794ms |

判断:

- opt45 の 14.926s から悪化。
- `LoadConst` 命令を減らせても、hot opcode 側で `Operand::Const` の value_pool 読みが増え、register slot 読みより重くなった可能性が高い。
- 現在の register VM では、定数を一度 register slot に置いて読む形がこのシーンでは速い。
- この変更は revert 済み。不採用。

## 47. Checkpoint after bytecode evaluator optimizations

採用した内容:

- `Surface.opacity` が local constant 経由で 1.0 になる場合、alpha test 不要と判定する。
- `--profile-mtlx` 時に source graph と flatten 後 graph の node 数、展開率、上位 category を出す。
- `Arith` evaluator を scalar closure 方式から type/op specialized helper へ変更し、vector 演算は glam の vector op を使う。

不採用にした内容:

- periodic non-UDIM image fast path。
- `LoadConst` register の hot operand inline 化。

検証:

- `cargo check`: OK。
- `cargo test material::mtlx:: --lib`: OK、214 tests。
- `cargo test`: OK、lib 716 tests、main 2 tests、doc 0 tests。

現在の通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt45_arith_specialized_scene45_512_spp128.png` | 00m:14s:926ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt45_arith_specialized_scene41_512_spp128.png` | 00m:06s:587ms |

効果:

- opt36 checkpoint `4b2e886` の scene45 15.684s から 14.926s へ改善。
- scene45/scene41 比率は `14.926 / 6.587 = 2.27x`。

コミット:

- ここで `Optimize MaterialX opacity and arithmetic evaluation` として checkpoint commit を作る。

## 48. Profile 35: Arith op breakdown

目的:

- bytecode VM の typed opcode / superinstruction 化を検討するため、`Arith` の内訳を確認する。
- 通常 render には影響しない `--profile-mtlx` 専用の ArithOp 別カウンタを追加した。

検証:

- `cargo check`: OK。
- `cargo test material::mtlx:: --lib`: OK、214 tests。

profile:

- 出力: `result/perf/profile_opt48_arith_op_breakdown_scene45_512_spp128.png`
- `render: 00m:40s:962ms` (profile overhead 込み、通常速度比較には使わない)

ArithOp 内訳:

| op | calls | total | per_call |
| --- | ---: | ---: | ---: |
| Multiply | 469706620 | 14648ms | 31ns |
| Add | 108384812 | 3441ms | 32ns |
| Max | 63546409 | 2199ms | 35ns |
| Power | 40063078 | 2147ms | 54ns |
| Subtract | 34011908 | 1063ms | 31ns |
| Modulo | 7310341 | 265ms | 36ns |
| Divide | 5394748 | 187ms | 35ns |

判断:

- `Multiply` が圧倒的に多い。
- 次に `Add`、`Max`、`Power` が続く。
- bytecode format を変更する前に、`execute_instruction` の `Arith` match でこの4演算を専用分岐に逃がす実験を行う。

## 49. Rejected: specialize hot Arith ops in execute_instruction

仮説:

- `Multiply`、`Add`、`Max`、`Power` だけを `Instruction::Arith` の専用 match arm に分ければ、`arith()` 内の `ArithOp` 分岐を避けられる。
- bytecode enum 自体は変更しないので、typed opcode 化より低リスク。

実装実験:

- `execute_instruction()` に `op: ArithOp::Multiply` / `Add` / `Max` / `Power` の専用 arm を追加。
- 非matrixの場合は `arith_multiply` / `arith_add` / `arith_max` / `arith_power` に直接入るようにした。
- matrix の `Add` / `Multiply` は従来通り `arith_mat3` / `arith_mat4` へ渡した。

検証:

- `cargo check`: OK。
- `cargo test material::mtlx:: --lib`: OK、214 tests。

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt49_arith_hot_op_specialized_scene45_512_spp128.png` | 00m:15s:791ms |

判断:

- opt45 の 14.926s から悪化。
- `execute_instruction` の巨大 match にさらに細かい arm を増やすことで、i-cache や分岐予測の悪化が勝った可能性が高い。
- この変更は revert 済み。不採用。
- ArithOp 別 profile は今後の判断材料として残す。

## 50. Optimization 50: inline constant closure params and fold local Arith constants

仮説:

- opt48/49 で `Arith` 実行そのものは hot だが、runtime evaluator 側の細かい分岐追加は逆効果だった。
- 一方で closure parameter が compile-time constant なのに `ParamRef::Local` のまま残ると、BSDF walk のたびに bytecode register を読む。
- closure node の parameter を `ParamRef` literal に置き換えられれば、bytecode 実行回数ではなく closure traversal 側の局所的な読み出しと型変換を減らせる。
- MaterialX graph では constant arithmetic 経由の parameter もあるため、local constant 解析に `Instruction::Arith` の非matrix constant fold を追加する。

実装:

- `local_constants()` を `simplify_closure_nodes()` 前に呼び、closure node の parameter に渡す。
- `inline_closure_constant_params()` を追加し、`ParamRef::Local` が compile-time constant の場合だけ literal `ParamRef` に置き換える。
- `local_constants()` に非matrix `Instruction::Arith` の定数畳み込みを追加した。
- matrix は pool lifetime と参照型の扱いが絡むため対象外。

検証:

- `cargo check`: OK。
- `cargo test material::mtlx:: --lib`: OK、214 tests。
- baseline scene45 512/128 との RMSE は `3918.79 (0.0597951)`。
- diff画像: `result/perf/diff_opt50_closure_const_params_vs_baseline_scene45_512_spp128.png`

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt50_closure_const_params_scene45_512_spp128.png` | 00m:11s:645ms |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt50_closure_const_params_scene45_512_spp128_run1.png` | 00m:11s:934ms |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt50_closure_const_params_scene45_512_spp128_run2.png` | 00m:11s:779ms |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt50_closure_const_params_scene45_512_spp128_run3.png` | 00m:11s:850ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt50_closure_const_params_scene41_512_spp128_run1.png` | 00m:06s:636ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt50_closure_const_params_scene41_512_spp128_run2.png` | 00m:06s:893ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt50_closure_const_params_scene41_512_spp128_run3.png` | 00m:06s:648ms |

効果:

- opt45 の scene45 14.926s から初回 11.645s へ大きく改善。
- 3回平均では scene45 `11.854s`、scene41 `6.726s`。
- scene45/scene41 比率は平均で `1.76x`。

判断:

- 改善幅が大きく、画像差分も破綻していないため採用候補として残す。
- ただし `inline_closure_constant_params()` は追加コード量が大きいため、以降の監査ではこの変更自体も「効果が十分大きいので複雑性を許容できるか」という観点で見る。

## 51. Rejected: extend constant folding to MixValue and Clamp

仮説:

- opt50 の closure constant parameter 化が大きく効いたため、`MixValue` や `Clamp` も compile-time constant fold すればさらに local constant 化できる可能性がある。

実装実験:

- `local_constants()` に `Instruction::MixValue` と `Instruction::Clamp` の定数畳み込みを追加。
- 対応する spec test の期待値も一時的に確認した。

検証:

- `cargo check`: OK。
- `cargo test material::mtlx:: --lib`: OK、214 tests。

通常render測定:

| scene | 条件 | 出力 | render |
| --- | --- | --- | --- |
| 45 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt51_mix_clamp_constant_fold_scene45_512_spp128.png` | 00m:11s:684ms |
| 41 | 512x512, 128spp, integrator=mis, depth=16 | `result/perf/opt51_mix_clamp_constant_fold_scene41_512_spp128.png` | 00m:07s:075ms |

判断:

- opt50 の scene45 初回 11.645s よりわずかに悪く、scene41 も悪化した。
- 改善幅が通常の実行分散より小さいため、新しい基準では採用しない。
- この変更は revert 済み。不採用。

## 52. Audit: adopted optimizations below the new significance bar

新しい基準:

- 0.1s程度、または1%未満の差は単発計測では採用理由にしない。
- そのような差を評価する場合は複数回計測し、平均で見る。
- 差が分散に埋もれる場合、コードの複雑性を増やす変更は採用しない。
- 正しさ修正や明確な構造改善は速度差だけで revert しないが、速度最適化としての主張は分けて扱う。

ログからの監査候補:

| opt | 内容 | ログ上の効果 | 監査判断 |
| --- | --- | --- | --- |
| 11 | lazy light-tree query | opt10 18.505s に対して 18.550s / 18.697s で悪化 | 最有力の再検証候補。構造的には不要precompute削減だが、scene45では効果なし。 |
| 12 | `may_emit` guard | opt11 と同程度、明確な改善なし | guard は小さいが、単体速度改善としては弱い。 |
| 13 | single-lobe sample fast path | opt12 18.633s から 18.531s、約0.55% | 新基準では単発採用には弱い。再検証候補。 |
| 14 | GGX reflection LUT light-tree summary | opt13 18.531s から 18.344s、約1.0% | 後続の正しさ修正で置き換わっている可能性が高く、現行差分確認が必要。 |
| 21 | power-of-two texture wrapping | opt18 17.977s から 17.720s、約1.4% | 小さいが1%超。実装も局所的なので優先度低。 |
| 25 | geometric load CSE | opt21 17.720s から 17.637s、約0.47% | 新基準では採用根拠が弱い。ただし命令数は大きく減っているため、再検証して判断する。 |
| 45 | Arith evaluator dispatch | opt44 15.230s から 14.926s、約2.0% | 1%超だが追加コード量がある。現行の大きなopt50後にまだ効いているか確認余地あり。 |

次の作業:

- まず opt11 / opt13 / opt25 を一時的に無効化して、現行opt50基準との差を複数回で見る。
- 差が小さい、または無効化しても悪化しないものは実装の複雑性を確認し、revert候補にする。
- profile 実行は通常render測定と混ぜず、必要な場合だけ順番に実行する。

## 53. Audit correction: exclude structural/correctness changes from speed-only revert candidates

ユーザー指摘を受けた監査基準の補正:

- opt11 の light-tree query lazy build は scene45 では LightTree category が支配的なため効果が見えない。
- しかし論理的には、LightTree query / light-tree precompute は LightTree category が選ばれた時だけ必要であり、Environment / Directional light sample のために事前構築する必要はない。
- そのため opt11 は速度差だけでrevertする候補から外す。
- むしろ `lazy` という実験的な名前を残すより、通常の MIS compensated light sampling がこの挙動になるよう整理する余地がある。

Directional Albedo LUT の扱い:

- Dielectric / GeneralizedSchlick / Sheen などの Directional Albedo LUT は、単なる速度最適化ではなく、根拠の弱い Fresnel(cos_o) 近似を置き換えて layer throughput を正しくするための修正。
- render時間の改善が小さい、または悪化していても、正しさ修正として保持する。
- 速度監査のrevert候補からは外す。

残す監査候補:

| opt | 内容 | 理由 |
| --- | --- | --- |
| 12 | `may_emit` guard | 正しさは保った小さなguardだが、速度効果は明確でない。 |
| 13 | single-lobe sample fast path | ログ上の改善が約0.55%で、新基準では単発採用として弱い。 |
| 21 | power-of-two texture wrapping | 約1.4%だが単発。実装は局所的で低複雑性。 |
| 25 | geometric load CSE | 約0.47%。命令数削減は大きいが、速度採用根拠は弱い。 |
| 45 | Arith evaluator dispatch | 約2%で1%超。ただし追加実装量があるため、他候補後に必要なら確認する。 |

## 54. Audit result: revert opt13 single-lobe sample fast path

対象:

- opt13 `sample_needs_eval_pdf`。
- 単一ローブMaterialX sample pathで、sample済み `candidate.weight` / `candidate.pdf` を使い、sample後の `eval_pdf` 再計算を省く実装。

ログ上の問題:

- opt12 18.633s から opt13 18.531s で、改善は約0.55%。
- 単発測定であり、新基準では採用根拠として弱い。
- 実装上は `CompiledMaterial::sample_needs_eval_pdf` field、compile walker、runtime branch、test fixture field が増えていた。

一時無効化測定:

| 条件 | 出力 | render |
| --- | --- | --- |
| opt13無効化 run1 | `result/perf/audit_disable_opt13_sample_fast_path_scene45_512_spp128_run1.png` | 00m:11s:561ms |
| opt13無効化 run2 | `result/perf/audit_disable_opt13_sample_fast_path_scene45_512_spp128_run2.png` | 00m:11s:784ms |
| opt13無効化 run3 | `result/perf/audit_disable_opt13_sample_fast_path_scene45_512_spp128_run3.png` | 00m:11s:845ms |

平均:

- opt13無効化平均: `11.730s`。
- 直前の現行opt50平均: `11.854s`。

復帰後の確認測定:

| 条件 | 出力 | render |
| --- | --- | --- |
| opt13復帰 run1 | `result/perf/audit_current_after_opt13_restore_scene45_512_spp128_run1.png` | 00m:12s:077ms |
| opt13復帰 run2 | `result/perf/audit_current_after_opt13_restore_scene45_512_spp128_run2.png` | 00m:11s:721ms |
| opt13復帰 run3 | `result/perf/audit_current_after_opt13_restore_scene45_512_spp128_run3.png` | 00m:11s:716ms |

平均:

- opt13復帰平均: `11.838s`。
- 無効化との差は `0.108s`、約0.9%。

revert実装:

- `CompiledMaterial::sample_needs_eval_pdf` を削除。
- `closure_sample_needs_eval_pdf()` を削除。
- MaterialX sample後は thin-walled transmission を除き、常に `eval_pdf_closure_cached()` でweightを計算する単純な形に戻した。
- test fixture / scene fixture の `sample_needs_eval_pdf` field を削除。

revert後測定:

| 条件 | 出力 | render |
| --- | --- | --- |
| opt13 revert run1 | `result/perf/audit_revert_opt13_sample_fast_path_scene45_512_spp128_run1.png` | 00m:12s:156ms |
| opt13 revert run2 | `result/perf/audit_revert_opt13_sample_fast_path_scene45_512_spp128_run2.png` | 00m:11s:659ms |
| opt13 revert run3 | `result/perf/audit_revert_opt13_sample_fast_path_scene45_512_spp128_run3.png` | 00m:11s:982ms |

平均:

- opt13 revert平均: `11.932s`。
- opt13復帰平均との差は `0.094s`、約0.8%悪化。
- この差は実行分散内であり、速度最適化として保持する根拠にはならない。

検証:

- `cargo check`: OK。
- `cargo test material::mtlx:: --lib`: OK、214 tests。

判断:

- 新基準では、0.1s未満から0.1s程度の差を理由に compile/runtime構造を複雑にする変更は採用しない。
- opt13はrevertした状態を採用して先へ進む。

## 55. Audit result: keep opt25 geometric load CSE

対象:

- opt25 `ensure_geometric_kind_local()` による MaterialX bytecode 内 geometric load CSE。

ログ上の問題:

- opt21 17.720s から opt25 17.637s で、改善は約0.47%。
- 単発測定であり、速度だけなら新基準では採用根拠が弱い。

一時無効化測定:

| 条件 | 出力 | render |
| --- | --- | --- |
| opt25無効化 run1 | `result/perf/audit_disable_opt25_geometric_cse_scene45_512_spp128_run1.png` | 00m:11s:811ms |
| opt25無効化 run2 | `result/perf/audit_disable_opt25_geometric_cse_scene45_512_spp128_run2.png` | 00m:12s:040ms |
| opt25無効化 run3 | `result/perf/audit_disable_opt25_geometric_cse_scene45_512_spp128_run3.png` | 00m:11s:656ms |

平均:

- opt25無効化平均: `11.836s`。
- opt13 revert後の直前平均: `11.932s`。
- scene45速度だけでは差は分散内で、明確な速度改善とは言えない。

保持理由:

- opt25はruntime hot pathに分岐を増やす変更ではなく、compile時に同一の geometric input を同じ virtual register に束ねるCSE。
- ログの profile では `LoadGeom` calls が opt25 前 `396M` から opt25 後 `219M` に減っており、命令数削減としては明確。
- 実装は `ensure_geometric_kind_local()` に局所化されており、アーキテクチャを大きく複雑にしていない。

判断:

- scene45の通常render時間だけを採用根拠にはしない。
- ただし bytecode命令数を大きく減らす低複雑性のcompile CSEなので保持する。

## 56. Audit result: keep opt12, opt21, opt45

opt12 `may_emit` guard:

- `MtlxMaterial::le()` の先頭で、compile済み `may_emit` がfalseなら `evaluate_le()` を呼ばないだけの短絡。
- ログ上の速度改善は明確ではないが、実装は小さく、非発光materialでemission walkをしない構造は自然。
- revert候補から外す。

opt21 power-of-two texture wrapping:

- `wrap_index()` で `size.is_power_of_two()` の時だけ bitmask にする局所変更。
- ログ上は約1.4%改善で、1%未満ではない。
- 実装複雑性が低く、fallbackの `rem_euclid` も保持しているためrevertしない。

opt45 Arith evaluator dispatch:

- ログ上は opt44 15.230s から opt45 14.926s で約2.0%改善。
- 1%未満の微小最適化ではない。
- `arith_scalar` / `arith_vec2` / `arith_vec3` / `arith_vec4` への分離は、runtime evaluator の構造として追える範囲に収まっている。
- 現時点ではrevert候補にしない。

## 57. Verification after audit changes

最終状態:

- opt13 `sample_needs_eval_pdf` はrevert。
- opt11 light-tree query lazy build は構造改善として保持。
- Directional Albedo LUT 系は正しさ修正として保持。
- opt12 / opt21 / opt25 / opt45 は保持。

検証:

- `cargo check`: OK。
- `cargo test material::mtlx:: --lib`: OK、214 tests。
- `cargo test`: OK、lib 716 tests、main 2 tests、doc 0 tests。

メモ:

- この監査では profile と scene41 を同時実行していない。
- 速度比較はすべて `render:` 行だけを使った。
- opt13 revert後のscene45平均は現行との差が1%未満なので、速度改善としては主張しない。採用理由は複雑性削減。

## 58. Profile audit after opt50 and opt13 revert

目的:

- opt50 と opt13 revert によりコードパスが変わったため、既存profilerが現在のhot pathを漏らしていないか確認する。

確認結果:

- `run_instructions()` は full material bytecode と opacity-only bytecode の両方を `PROF_BYTECODE_*` に集計していた。
- そのため、`opacity` 行は `surface_opacity_at*()` のclosure opacity walkだけを示し、opacity bytecode実行時間は `bytecode` 行に混ざっていた。
- `evaluate_le()` は専用timerがなかった。ただし scene45 profile では `le` 実評価は0回で、今回のhotspotではなかった。
- `precompute_shading()` wrapper全体のtimerがなく、scratch allocation / dalbedo cache allocation / bytecode実行のinclusive costが見えていなかった。

実装:

- `run_instruction_stream()` に `BytecodeProfileKind::Full | Opacity` を渡し、full bytecode と opacity bytecode を分離集計する。
- `precompute_shading()` wrapperのinclusive profile counterを追加。
- `evaluate_le()` のprofile counterを追加。
- compile profileに、slot割当後に残っている「既存 `local_constants()` で定数と判定できるdst命令数」を表示する。
- `local_constants()` はSSA前提では問題なかったが、slot割当後diagnosticにも使うため、unknown dstで古い定数情報をclearするようにした。

検証:

- `cargo check`: OK。

profile:

- 出力: `result/perf/profile_opt59_profiler_audit_scene45_512_spp128.png`
- `render: 00m:26s:334ms` (profile overhead込み、通常速度比較には使わない)

主要結果:

| counter | calls | instrs | thread total |
| --- | ---: | ---: | ---: |
| bytecode total | 33,371,348 | 1,164,349,537 | 331,527ms |
| full bytecode | 28,484,341 | 1,110,592,460 | 317,695ms |
| opacity bytecode | 4,887,007 | 53,757,077 | 13,832ms |
| precompute inclusive | 28,484,341 | - | 324,933ms |
| light_tree_precompute | 27,391,156 | - | 15,967ms |
| sample | 28,484,294 | - | 14,379ms |
| eval_pdf | 54,664,312 | - | 33,623ms |
| le | 0 | - | 0ms |

判断:

- profileは主要な現行コードパスを概ね捉えている。
- ただし profile時間はthread累計かつ `Instant::now()` / atomic overhead込みなので、通常renderの絶対速度比較には使わない。
- hotspot帰属としては、残りは full bytecode precompute、closure sample/eval_pdf/light_tree_precompute traversal、Image/Arith/MixValue系命令が中心。

## 59. Experiment: bypass all MaterialX texture sampling after opt50

目的:

- 以前のtexture bypass実験はopt50前だったため、現行でtextureがまだ数秒級の上限を持つか確認する。

一時変更:

- `Image` はdefault valueを返す。
- `HextiledImage` / `HextiledNormalMap` はtexture sampleとhextile blendを飛ばしてdefaultを返す。
- 画質は壊れるため、速度上限を見るだけの実験。

通常render測定:

| 条件 | 出力 | render |
| --- | --- | --- |
| texture bypass run1 | `result/perf/experiment_opt60_bypass_mtlx_textures_current_scene45_512_spp128_run1.png` | 00m:10s:757ms |
| texture bypass run2 | `result/perf/experiment_opt60_bypass_mtlx_textures_current_scene45_512_spp128_run2.png` | 00m:10s:782ms |
| texture bypass run3 | `result/perf/experiment_opt60_bypass_mtlx_textures_current_scene45_512_spp128_run3.png` | 00m:11s:112ms |

結果:

- 平均: `10.884s`。
- 現行平均 `11.932s` からの上限改善は約 `1.05s`。

判断:

- texture sampling はまだhotspotだが、現行では全部潰しても3秒級には届かない。
- textureだけに大きな更新余地が残っているわけではない。
- 一時変更はrevert済み。不採用。

## 60. Experiment: general fold of currently detectable uniform bytecode constants

目的:

- constなuniform値のbytecode畳み込みがまだ大きく残っているか確認する。

調査:

- compile profileに `const_dst_instrs` / `const_non_load` を追加。
- 修正前のdiagnosticはslot割当後のregister再利用に対して古い定数情報が残っていたため、`local_constants()` でunknown dstをclearするよう修正した。
- 修正後のcompile probeでは、残るconst_non_loadは多くのmaterialで0、Car_Paintなど一部で数個から十数個程度。

一時実装:

- SSA段階で `local_constants()` が定数と判定できる非`LoadConst`命令を `LoadConst` に置き換える。
- その後DCEをもう一度走らせる。

compile probe:

- 出力: `result/perf/profile_opt63_constant_fold_probe_scene45_1_spp1.png`
- 命令数は少し減ったが大幅ではない。
- 例:
  - Car_Paint: `100 -> 96`
  - Copper_Satin: `71 -> 69`
  - material_checker_opacity: `41 -> 39`

通常render測定:

| 条件 | 出力 | render |
| --- | --- | --- |
| constant fold experiment | `result/perf/experiment_opt63_constant_fold_scene45_512_spp128_run1.png` | 00m:11s:941ms |

判断:

- 現行平均 `11.932s` と同等。
- const uniform foldingの未処理分は、少なくとも既存 `local_constants()` で検出できる範囲では数秒級ではない。
- 実装複雑性に対して効果がないため、一時変更はrevert済み。
- diagnostic用の `local_constants()` clear修正とcompile統計は残す。

## 61. Experiment: skip post-sample eval_pdf traversal upper bound

目的:

- MaterialX `sample()` は lobe sample 後に正しいweight/pdfを得るため `eval_pdf_closure_cached()` でclosure tree全体を再評価する。
- これを完全に省けた場合に数秒級の余地があるか確認する。

一時変更:

- thin-walled transmission以外で、`eval_pdf` を呼ばずに `candidate.pdf` / `candidate.weight` をそのまま使う。
- layer / mixture の正しいMIS weightではなくなるため、速度上限のみを見る実験。

通常render測定:

| 条件 | 出力 | render |
| --- | --- | --- |
| skip sample eval_pdf | `result/perf/experiment_opt64_skip_sample_eval_pdf_scene45_512_spp128_run1.png` | 00m:11s:562ms |

判断:

- 現行平均 `11.932s` から約0.37s改善に留まる。
- sample後のeval_pdfを完全に消しても3秒級の余地はない。
- 正しさも壊れるため、一時変更はrevert済み。

## 62. Experiment: skip BSDF-hit LightTree query for area-light MIS pdf

目的:

- `light_tree_precompute` が約27M回走っているため、direct-light query と BSDF-hit MIS pdf query の重複が数秒級か確認する。

一時変更:

- `emitted_radiance_from_bsdf_sample_area()` で `light_tree::build_query()` を省き、`pdf_for_triangle_hit(scene, None, ...)` を使う。
- MIS pdfが正しくなくなるため、速度上限だけを見る実験。

通常render測定:

| 条件 | 出力 | render |
| --- | --- | --- |
| skip BSDF-hit LightTree query | `result/perf/experiment_opt65_skip_bsdf_hit_lighttree_query_scene45_512_spp128_run1.png` | 00m:11s:806ms |

判断:

- 現行平均との差は分散内。
- `light_tree_precompute` の大半はdirect-light sampling側で、BSDF-hit MIS pdf queryの単純な再利用/省略では数秒級にならない。
- 一時変更はrevert済み。

## 63. Current normal render after profiler audit changes

通常render測定:

| 条件 | 出力 | render |
| --- | --- | --- |
| current run1 | `result/perf/opt66_profiler_audit_current_scene45_512_spp128_run1.png` | 00m:12s:137ms |
| current run2 | `result/perf/opt66_profiler_audit_current_scene45_512_spp128_run2.png` | 00m:11s:957ms |
| current run3 | `result/perf/opt66_profiler_audit_current_scene45_512_spp128_run3.png` | 00m:11s:898ms |

結果:

- 平均: `11.997s`。
- 前回平均 `11.932s` との差は約0.5%で分散内。

現時点の判断:

- 現行コードには、opt50のように単独で約3秒改善する明確な未処理箇所は見えていない。
- texture全無効化上限は約1.05s。
- sample後eval_pdf全省略上限は約0.37s。
- BSDF-hit側LightTree query省略上限は分散内。
- 既存 `local_constants()` で検出できるuniform constant folding未処理分は通常renderでは効果なし。
- 大きく改善するには、単発の小最適化ではなく、MaterialX Standard Surface / Layer closureをより直接的なnative evaluatorへ落とす、またはclosure tree全体をtyped/specialized closure programへ変換するような大きい設計変更が必要そう。

## 64. Profile path coverage audit

目的:

- 最適化でMaterialX評価のコードパスが変わっているため、profilerが現在の主要パスを取り逃していないか確認する。

確認した入口:

- `Material::precompute_shading()` -> `MtlxMaterial::precompute_shading()` -> full bytecode。
- `Material::sample()` -> `MtlxMaterial::sample()` -> `sample_closure_cached()` と sample後の `eval_pdf_closure_cached()`。
- `Material::eval()` -> `MtlxMaterial::eval()` -> `eval_closure_cached()`。
- `Material::pdf()` -> `MtlxMaterial::pdf()` -> `pdf_closure_cached()`。
- `Material::eval_pdf()` -> `MtlxMaterial::eval_pdf()` -> `eval_pdf_closure_cached()`。
- `Material::le()` -> `MtlxMaterial::le()` -> `evaluate_le()`。
- `Material::light_tree_precompute()` -> `MtlxMaterial::light_tree_precompute()` -> `light_tree_precompute_closure_cached()`。
- alpha test path: `Scene::closest_hit()` -> `Material::any_hit()` -> `MtlxMaterial::any_hit()` -> opacity bytecode と opacity closure。

見直し結果:

- full material bytecode と opacity-only bytecode は `bytecode split` で別々に計測できるようにした。
- `precompute_shading` は full bytecodeを含むinclusive timeとして計測できるようにした。
- `evaluate_le` は専用カウンタを追加した。scene 45のprofileでは `0 calls` で、現シーンのhot pathではないことを確認した。
- opacityは、opacity bytecode実行時間とopacity closure walk時間を分けて読めるようにした。
- `sample` / `eval` / `pdf` / `eval_pdf` / `light_tree_precompute` / `pre_dalbedo` は既存カウンタで現在のclosure walker経路を捕捉している。
- `any_hit` ラッパ全体の専用inclusive counterも追加した。これでalpha test入口、opacity bytecode、opacity closure walkを分けて読める。

判断:

- 現在のprofile出力は、今回の最適化後に存在する主要なMaterialX評価パスを概ね捕捉できている。
- 直近のprofileでは opacity bytecode が約13.8s/profile-overhead、opacity closure walkが約0.19s/profile-overheadだったため、`any_hit` ラッパ自体が数秒級の未発見hotspotになる可能性は低い。ただし今後のコードパス変化に備え、入口単位でも確認できる形にした。

検証:

| 条件 | 出力 | render |
| --- | --- | --- |
| profile with any_hit counter | `result/perf/profile_opt67_any_hit_profile_scene45_512_spp128.png` | 00m:27s:653ms |

主なprofile出力:

- bytecode total: `355562ms / 33364587 calls / 1164058204 instrs`
- bytecode full: `340354ms / 28475891 calls / 1110282548 instrs`
- bytecode opacity: `15208ms / 4888696 calls / 53775656 instrs`
- precompute inclusive: `352160ms / 28475891 calls`
- light_tree_precompute: `16705ms / 27382143 calls`
- any_hit inclusive: `18287ms / 4888696 calls`
- opacity closure walk: `185ms / 4888696 calls`
- le: `0ms / 0 calls`

追加判断:

- `any_hit` inclusiveは opacity bytecode を含んだ入口時間として見えるようになった。
- opacity closure walkは軽く、alpha test経路の主成分は opacity bytecode。
- `le` はこのsceneでは呼ばれておらず、未計測hotspotではない。

## 65. Refactor uniform constant folding into an explicit pass

目的:

- 既存の `local_constants()` はclosure param inlineやopacity判定用の補助解析であり、bytecode命令列そのものを体系的に簡約するpassではなかった。
- uniform式の簡約が中途半端に見える状態を避けるため、pure命令を明示的にconstant foldingするcompile passを追加する。

実装:

- `fold_constant_instructions()` を追加し、closure param inlineとDCEの前に実行する。
- opacity/fullそれぞれのDCE後にも同じpassを再実行し、foldで不要化した命令を再DCEする。
- SSA命令列を順に走査し、入力operandがすべてcompile-time constantで、結果がinline `Value` として表現できる命令を `LoadConst` に置き換える。
- その後の既存 `inline_closure_constant_params()` とDCEで、畳み込まれた定数をclosure nodeへinlineし、不要になった元計算を削除する。
- `local_constants()` は引き続き解析用途として残し、fold済み命令列に対する定数状態を読む。
- profile compile出力に `const_folded` を追加し、各MaterialX materialで何命令がfoldされたか確認できるようにした。

fold対象:

- arithmetic / unary / convert / logical / compare / ifelse / mix / clamp / smoothstep / extract。
- reflect / refract / rotate / dot / cross / distance / facingratio / luminance / combine / switch。
- place2d / latlong / ramp / split / blackbody / artistic_ior / hair補助 / roughness補助 / transformcolor。
- blend / merge / mask / premult / unpremult / contrast / range / remap / hsvadjust / saturate / colorcorrect / checkerboard。

foldしない対象:

- texture image / hextiled image / geometry load / object-world transform / matrix pool参照が必要なmatrix演算。
- noise / random / curve / triplanar / normalmap / bump / heighttonormal。
- これらはshading vertex、texture、derivative、matrix pool、または手続き評価の扱いが絡むため、今回のpure inline `Value` folding passからは外す。

ドキュメント整理:

- `result/perf/materialx_optimization_log.md` はレンダリング結果ではないため `MATERIALX_OPTIMIZATION_LOG.md` へ移動。
- `result/perf/materialx_optimization_summary.md` は `MATERIALX_OPTIMIZATION_REPORT.md` へ移動。

検証:

- `cargo check`: 成功。
- `cargo test material::mtlx --lib`: 222件成功。
- `cargo test`: library 716件、binary 2件、doc 0件成功。

profile render:

| 条件 | 出力 | render |
| --- | --- | --- |
| profile | `result/perf/opt68_constant_fold_pass_scene45_512_spp128.png` | 00m:21s:790ms |
| profile after post-DCE fold pass | `result/perf/opt69_constant_fold_after_dce_profile_scene45_512_spp128.png` | 00m:22s:126ms |

profile結果:

- bytecode total: `223631ms / 33374188 calls / 939301607 instrs`
- bytecode full: `211499ms / 28483596 calls / 885505095 instrs`
- bytecode opacity: `12132ms / 4890592 calls / 53796512 instrs`
- precompute inclusive: `220445ms / 28483596 calls`
- any_hit inclusive: `15186ms / 4890592 calls`

post-DCE fold profileは命令数・時間とも同等で、DCE後に追加で大きく畳める残りはなかった。

通常render測定:

| 条件 | 出力 | render |
| --- | --- | --- |
| scene45 run1 | `result/perf/opt68_constant_fold_pass_scene45_512_spp128_run1.png` | 00m:11s:296ms |
| scene45 run2 | `result/perf/opt68_constant_fold_pass_scene45_512_spp128_run2.png` | 00m:11s:356ms |
| scene45 run3 | `result/perf/opt68_constant_fold_pass_scene45_512_spp128_run3.png` | 00m:11s:329ms |
| scene41 run1 | `result/perf/opt68_constant_fold_pass_scene41_512_spp128_run1.png` | 00m:06s:649ms |

結果:

- scene45平均: `11.327s`。
- 前回現行平均 `11.997s` から約 `0.67s` 改善。
- scene45 / scene41 は `11.327 / 6.649 = 1.70x`。

判断:

- この変更は単なる測定分散ではなく、bytecode命令数を約11.64億から約9.39億に減らしている。
- 速度改善幅は約5.6%で、コード構造も `local_constants()` の暗黙用途から明示passへ分離されるため採用する。
