# MaterialX 1.39.4 監査レポート

このレポートは `MATERIALX_AUDIT.md` の監査メモをもとに、MaterialX 1.39.4 サブセット実装について確認した内容、仕様違反として修正した内容、追加したテスト、問題ないと判断した実装領域を日本語で再整理したものです。

## 監査ステータス

- 現在の未解決項目: なし。
- 監査メモの現在行数: `MATERIALX_AUDIT.md` 2918 行。
- 仕様書: MaterialX 1.39.4 の markdown 仕様書 7 ファイルを読了。
- 参照実装: MaterialX 1.39.4 upstream の MDL 実装、生成 MDL 実装 `.mtlx`、必要な GLSL/OSL/C++ 参照実装を確認。
- ローカル実装: loader、compile、runtime、BSDF、`MtlxMaterial`、integrator、light tree、scene 連携、テスト期待値を監査。
- 最終検証: `cargo fmt`、`cargo check`、全体 `cargo test` が成功。最終時点では 712 library tests、2 binary tests、0 doc tests が成功。
- 追加の末尾行数突合後検証: `cargo test scene_loader::mtlx_loader` 76 件成功、`cargo test material::mtlx` 220 件成功、`cargo test bsdf::mtlx` 35 件成功。

## 参照した仕様と実装

### MaterialX 仕様書

- `MaterialX.Specification.md` 1-1582 行。
- `MaterialX.StandardNodes.md` 1-1271 行。
- `MaterialX.PBRSpec.md` 1-513 行。
- `MaterialX.GeomExts.md` 1-525 行。
- `MaterialX.NPRSpec.md` 1-74 行。
- `MaterialX.Proposals.md` 1-337 行。
- `MaterialX.Supplement.md` 1-264 行。

### MaterialX upstream MDL / 生成実装

- `libraries/targets/genmdl.mtlx`。
- `libraries/nprlib/genmdl/nprlib_genmdl_impl.mtlx`。
- `libraries/pbrlib/genmdl/pbrlib_genmdl_impl.mtlx`。
- `libraries/stdlib/genmdl/stdlib_genmdl_impl.mtlx`。
- `source/MaterialXGenMdl/mdl/materialx/core.mdl`。
- `source/MaterialXGenMdl/mdl/materialx/hextile.mdl`。
- `source/MaterialXGenMdl/mdl/materialx/hsv.mdl`。
- `source/MaterialXGenMdl/mdl/materialx/noise.mdl`。
- `source/MaterialXGenMdl/mdl/materialx/pbrlib.mdl`。
- `source/MaterialXGenMdl/mdl/materialx/pbrlib_1_6.mdl`、`pbrlib_1_7.mdl`、`pbrlib_1_8.mdl`、`pbrlib_1_9.mdl`、`pbrlib_1_10.mdl`。
- `source/MaterialXGenMdl/mdl/materialx/sampling.mdl`。
- `source/MaterialXGenMdl/mdl/materialx/stdlib.mdl`。
- `source/MaterialXGenMdl/mdl/materialx/stdlib_1_6.mdl`、`stdlib_1_7.mdl`、`stdlib_1_8.mdl`、`stdlib_1_9.mdl`、`stdlib_1_10.mdl`。
- 必要に応じて `mx_heighttonormal_vector3.osl`、`mx_heighttonormal_vector3.glsl`、`HeightToNormalNodeMdl.cpp`、`mx_microfacet_sheen.glsl`、`mx_sheen_bsdf.glsl`、`mx_blackbody.glsl`、`mx_blackbody.osl`、`mx_artistic_ior.glsl`、`mx_artistic_ior.osl`、`MDL_spec_1.10.2.txt` も突合対象にした。

## 既知の許容ポリシー

- sRGB と linear 以外の色空間は、将来の OCIO 対応まで警告つきの一時フォールバックとして許容。
- volume と light ノードは現時点では未対応でよい。volume material は警告または passthrough/transparent 扱い。
- SSS は現時点では警告つきで Burley diffuse にフォールバックしてよい。ただし無視する入力も型・接続の検証は行う。
- measured EDF、displacement、animated image、cubic image filter、spectral rendering、blur convolution、dynamic height-to-normal sample-grid は、警告または明示エラーでよい。
- 上記の許容は、壊れた接続や参照を黙って無視することを許すものではない。

## 修正した仕様違反と改善内容

### 1. MaterialX loader / parser / resolver / library / flatten

- `types.rs`
  - 数値 array パースで malformed element を `filter_map` 的に落としていた経路を修正し、integer/float/vector/color array の不正要素をエラーにした。
  - 空の `stringarray` を 1 個の空文字列ではなく空配列として扱うようにした。
  - MaterialX の escape 規則に従い、stringarray 内の escaped comma、semicolon、backslash を保持するようにした。
  - matrix33 の row-major 入力保持、型 round-trip、boolean/float/color/vector literal の挙動をテストで固定した。

- `parser.rs`
  - root `version` が欠落または malformed でも黙って処理される経路を廃止し、構造エラーにした。
  - `typedef`、`geompropdef`、`nodedef`、`implementation`、`input`、`output` などの malformed element が `Option`/`filter_map` により落ちる挙動を廃止した。
  - `name`、`type`、`node`、`nodedef`、token `type` などの必須属性を必須として扱うようにした。
  - malformed boolean 属性を `false` 扱いせず、`true/false/1/0` 以外をエラーにした。
  - `geompropdef.index` の malformed integer を `None` にせずエラーにした。
  - `geompropdef` の `uniform`、array type、standard geomprop、space、index、type mismatch を仕様どおり検証するようにした。
  - `geomprop="viewdirection"` を vector3 standard geomprop として許容し、NPR の `Vworld` を扱えるようにした。
  - `geomprop="geomcolor"` は float/color3/color4 を許容するようにした。
  - `xi:include` の `href` 欠落をエラーにした。
  - XInclude された child document が root `namespace` や `colorspace` を省略した場合、include 元から継承するようにした。
  - document/root namespace の適用対象に nodedef `node` を含めるようにした。
  - element-level `namespace` は `<nodedef>` と `<nodegraph>` に適用し、仕様にない `<implementation namespace>` への適用は取り消した。
  - input/output の binding 属性について、`value`、`nodename`、`nodegraph`、`interfacename`、`defaultgeomprop` の競合をエラーにした。
  - `output` 属性だけがあり `nodename`/`nodegraph` を持たない input/output をエラーにした。
  - `defaultgeomprop` を uniform input や vector2/vector3 以外に指定した場合をエラーにした。
  - token `value` と `interfacename` の同時指定を ambiguous としてエラーにした。

- `library.rs`
  - nodedef inheritance の missing parent と cycle をエラーにした。
  - 必須 standard library file の欠落をエラーにした。
  - requested version が default version に黙ってフォールバックする挙動を廃止した。
  - MaterialX の omitted minor rule に従い、`major` と `major.0` を同一として扱うようにした。
  - nodedef matching で unknown input を許容しないようにした。
  - target-specific nodedef と nodegraph を universal renderer path で使用しないようにした。
  - universal `<implementation nodegraph=...>` を direct universal functional nodegraph より優先するようにした。

- `resolver.rs`
  - explicit `nodedef="..."` を名前だけで受け入れず、declared `node` category、input signature、output type と node use を照合するようにした。
  - local document nodedef lookup でも同じ category/signature gate を適用し、explicit nodedef mismatch が library nodedef に黙って落ちる挙動を廃止した。

- `flatten.rs`
  - missing `interfacename` が `FlatInput::Empty` になる経路を廃止し、`FlattenError::Missing` にした。
  - invalid typed literal を string や empty value にせずエラーにした。
  - missing required nodedef input、missing required nodegraph input を zero default にせずエラーにした。
  - nodegraph output に `nodename`/`nodegraph` がない場合、empty input から zero constant へ落ちる挙動を廃止しエラーにした。
  - unresolved disabled-node nodedef、missing nodegraph implementation、unknown nodegraph/nodedef input をエラーにした。
  - common `disable` control input を nodedef data socket matching から除外した。
  - shader-semantic `value=""` は明示的な empty shader connection として扱うようにした。
  - omitted/empty `surfaceshader` は passthrough empty closure としてコンパイルされ、`any_hit` で透明扱いされるようにした。
  - explicitly empty `backsurfaceshader` と `displacementshader` が不要な materialization や displacement warning を出さないようにした。
  - `volumematerial` は現在の no-volume constraint に従い、警告つき passthrough surface へ flatten するようにした。
  - local nodedef overload resolution で input name/type、requested output type、requested version、default-version preference を照合するようにした。
  - target-specific nodedef/nodegraph を universal path から除外した。
  - node reference と nodegraph reference で multi-output の `output` 指定を検証し、不明な output 名をエラーにした。
  - single-output への余分な `output` は仕様どおり無視するようにした。
  - custom nodegraph materialization で常に最初の output を使う挙動を廃止し、requested output を materialize するようにした。
  - node materialization cache key に requested output を含め、multi-output の別 output が alias しないようにした。
  - `type="multioutput"` を output type として誤用せず、aggregate multi-output use として扱い、requested output に基づいて型を確定するようにした。
  - disabled multi-output node で requested output の `defaultinput/default` を使うようにした。
  - disabled passthrough constant/materialized node の型を aggregate `type` ではなく selected output type にした。
  - functional nodegraph implementation が nodedef の全 output set と一致することを検証するようにした。
  - functional nodegraph の direct child input を仕様違反としてエラーにした。
  - nodedef token と node use token override を保持し、required token、unknown token、type validation、`[token]` filename substitution を実装した。
  - inherited nodedef token を inheritance resolution でコピーするようにした。
  - unsupported custom defaultgeomprop を UV0 や zero に黙って落とさずエラーにした。
  - geometry extension の `<asset>` など未対応 filename token は警告を出しつつ token text を保持するようにした。
  - `<UDIM>` と `<UVTILE>` は flatten では保持し、texture collection/runtime で展開する方針を明確にした。
  - unconnected non-shader input が unsupported type で zero value を持たない場合、`FlatInput::Empty` ではなくエラーにした。

- `build.rs`
  - image colorspace が sRGB/linear 以外の場合、黙って linear 扱いせず警告を出すようにした。
  - MaterialX image node の default fallback 仕様に従い、image/UDIM load failure は warning にとどめ、runtime default に流す方針を確認した。
  - same filename が先に color3、後で color4 として使われる場合、RGB cache の存在で alpha collection を落とさないようにした。
  - UDIM tile sets が RGB/alpha/scalar payload を merge するとき、既存 payload を破棄しないようにした。
  - scalar texture と color texture が同じ filename の場合、float/integer image output は scalar texture を優先するようにした。

### 2. texture / image / colorspace / animation

- `image`、`tiledimage`、`latlongimage`
  - float output が color texture から ACEScg luminance を読む挙動を修正し、StandardNodes の "first N channels" に従って Red channel を読むようにした。
  - color/vector4 output で file に alpha がない場合、足りない 4 番目 channel を 1.0 ではなく 0.0 で埋めるようにした。
  - scalar `image`/`tiledimage` の `<UDIM>`/`<UVTILE>` を scalar tile として scan/load するようにした。
  - `latlongimage` を regular UV image として扱うのをやめ、official `NG_latlongimage` と MDL に従い、`viewdir` と `rotation` から latlong UV を生成し、periodic U、mirror V、linear filter で sample するようにした。
  - `filtertype`、`uaddressmode`、`vaddressmode` を strict static string として検証し、dynamic connection や malformed 値が default に黙って落ちないようにした。
  - `file` socket は static filename/string を要求し、dynamic file connection が empty filename になる silent fallback を廃止した。
  - cubic filtering は警告つきで linear filter に落とす現方針を維持した。
  - unknown filter/address string はエラーにした。
  - `{frame}` と `{0Nframe}` の filename substitution は frame 0 固定の animated-image warning を出すようにした。
  - non-default `framerange`、`frameoffset`、`frameendaction` は animated image 未対応 warning または malformed static value error にした。

- colorspace
  - document、nodegraph、node、material、nodedef、input scope の `colorspace` inheritance を parser/flatten で処理するようにした。
  - color3/color4 literal は supported sRGB-like spaces から linear working space へ変換するようにした。
  - color4 alpha は変換せず保持するようにした。
  - `colorspace="none"` は warning なしの no-conversion とした。
  - unsupported `transformcolor` space は OCIO 未対応制約に従い warning を出して identity 扱いにした。

- `hextiledimage`
  - color4 default alpha と sampled alpha を保持するようにした。
  - color4 alpha は 3 サンプル平均に Schlick gain を適用し、MDL の `hextiledimage` に合わせた。
  - color3/vector3 output は luminance helper ではなく RGB channel semantics に従うようにした。

- `hextilednormalmap`
  - missing texture を flat normal に黙って落とす挙動を廃止し、MaterialX/MDL の `default` socket を返すようにした。
  - `default`、`normal`、`tangent`、`bitangent` socket を instruction に持たせるようにした。
  - missing frame socket は `Nworld`/`Tworld`/`Bworld` を明示 load するようにした。
  - explicit zero frame input を missing default と誤判定しないようにした。
  - per-tile normal conversion、axis rotation、gradient blend は MDL/GLSL と同じ raw normalize 方針にした。

- `triplanarprojection` / `triplanarblend`
  - `triplanarprojection` は official standard-library nodegraph expansion で実装されることを確認した。
  - six overloads が parser、stdlib flatten、compile を通ることをテストした。
  - `triplanarblend.filtertype=cubic` は warning、unknown filter は error とした。

### 3. StandardNodes math / matrix / transform / channel

- numeric math
  - `modulo` は MDL `core::mx_mod` の `x - y * floor(x/y)` に合わせた。負 divisor と zero divisor の NaN/inf propagation も raw にした。
  - `sqrt`、`ln`、`asin`、`acos` は defensive clamp や finite fallback を廃止し、native math の NaN/inf を保持するようにした。
  - `fract` は Rust `f32::fract()` ではなく `x - floor(x)` 型の MDL/GLSL/OSL 挙動に合わせた。
  - `safepower` は `sign(in1) * pow(abs(in1), in2)` にした。
  - `floor`/`ceil`/`round` の integer output は float operation 後に integer constructor で変換する MDL 挙動にした。
  - `atan2`、trigonometric、power、sqrt/ln/exp、sign、absval などの runtime と default を確認した。

- vector math
  - `magnitude`、`dotproduct`、`distance` が vector2/vector3/vector4 の declared dimension を保持するようにした。
  - `normalize` は zero vector を特別扱いせず raw normalize behavior にした。
  - `min`/`max` は vector/color と float RHS を component-wise に扱うことを確認した。
  - `reflect` と `refract` は nodegraph/MDL と同じ unclamped dot/raw normal 方針を確認した。

- transform
  - `transformpoint`、`transformvector`、`transformnormal` の missing `fromspace`/`tospace` は object-to-world default とした。
  - `transformnormal.in` missing default は `(0,0,1)` とした。
  - transform space string は strict static string とし、dynamic connection を default に黙って落とさないようにした。
  - `transformmatrix` は declared output vector type を保持し、vector2M3 と vector3M4 の temporary appended `1.0` を MDL と同じにした。
  - invalid vector2+matrix44 や vector4+matrix33 を coercion せず reject するようにした。

- matrix
  - `transpose` と `invertmatrix` は declared output matrix type で overload selection することを確認した。
  - `determinant` が default-only matrix44 input を matrix33 と誤判定しないようにした。
  - `invertmatrix` は StandardNodes の仕様挙動に従い、MDL TODO stub より仕様を優先する方針を明示した。
  - `creatematrix` の matrix44 from vector3 rows は MDL と同じ `w = 0,0,0,1` 挿入にした。
  - `zero_param(Matrix33/Matrix44)` は identity ではなく zero matrix にした。identity default が必要なノードは nodedef default を明示的に使う。
  - matrix `add/subtract/multiply/divide` を compile/runtime 実装し、matrix divide は StandardNodes の `in1 * inverse(in2)` 方針にした。MDL には TODO stub があるため、この箇所は StandardNodes-over-MDL-TODO として記録した。

- channel / combine / extract / separate / convert
  - MaterialX `convert` は仕様上許される value convert overload のみ許可し、MaterialX node としての float-to-integer など unsupported conversion はエラーにした。
  - internal runtime conversion は floor/ceil/round integer outputs などのために保持した。
  - value-to-surfaceshader convert は shipped library nodegraph expansion に任せ、direct literal-to-closure fallback は禁止した。
  - `combine2` は declared `in1` type と declared output type を使って overload resolution するようにした。
  - `combine2/3/4` の channel copy と default を StandardNodes/MDL と照合した。
  - `extractrowvector` を実装し、matrix33→vector3、matrix44→vector4、identity default、static index validation、out-of-range error を入れた。
  - `extract` は generated MDL の switch/default branch と同じく、negative/out-of-range index で last channel を返すようにした。
  - `separate3/4` は color input に color output names、vector input に vector output names のみ許可するようにした。
  - multi-output pattern dispatch は missing/unknown output name を error にし、最初の output への silent selection を廃止した。

### 4. StandardNodes adjustment / compositing / conditionals / utility

- `range`
  - non-positive span に `1e-30` を代入する fallback を廃止し、reversed range や zero span は式どおり評価して inf/NaN を伝播させるようにした。
  - `doclamp` は static boolean とし、dynamic/malformed 値を error にした。

- `smoothstep`
  - cubic Hermite と bounds order を確認し、テストで固定した。

- `luminance`
  - ACEScg default coefficient と custom `lumacoeffs` を確認した。
  - color4 alpha preservation を確認した。

- `rgbtohsv` / `hsvtorgb`
  - `rgbtohsv` の低 chroma threshold を MDL と同じ `FLOAT_MIN` 相当の扱いにした。
  - `hsvtorgb` は MDL hue wrapping と channel formula に一致することを確認した。
  - color4 alpha preservation を確認した。

- `contrast`
  - stdlib nodegraph の `(in - pivot) * amount + pivot` に一致することを確認した。
  - color4 と float RHS overload をテストした。

- `hsvadjust`
  - default `amount=(0,1,1)` と output type preservation を修正した。
  - saturation/value clamp を廃止し、stdlib nodegraph/MDL helper と同じく pass-through するようにした。

- `saturate`
  - `mix(luminance(in), in, amount)` の nodegraph formula に一致することを確認した。
  - amount を clamp しないこと、custom `lumacoeffs`、color4 alpha preservation を確認した。

- `colorcorrect`
  - 実行順を stdlib nodegraph と同じ、hue-only HSV adjustment、luminance mix saturation、signed range gamma、lift、gain、contrast、exposure にした。
  - 以前の lift/gain/gamma/contrast 順序違い、`1 + contrast`、negative RGB clamp を修正した。

- `premult` / `unpremult`
  - missing input default を `(0,0,0,1)` にした。
  - `premult` は RGB に alpha を掛け alpha を保持する挙動を確認した。
  - `unpremult` は alpha が完全に `0.0` の場合のみ passthrough にし、tiny nonzero alpha は divide するようにした。
  - MDL 1.6 TODO stub ではなく StandardNodes と OSL/GLSL を優先する方針を記録した。

- blend compositing
  - `plus`、`minus`、`difference`、`burn`、`dodge`、`screen`、`overlay` を確認した。
  - vector output を reject するようにした。
  - burn/dodge の `FLOAT_EPS` branch を MDL と一致させた。
  - color4 alpha は ordinary channel として処理することを確認した。

- merge / mask compositing
  - `disjointover`、`in`、`mask`、`matte`、`out`、`over` は color4-only とし、non-color4 output を reject するようにした。
  - MDL formula、disjointover の epsilon guard、mix interpolation を確認した。
  - `inside` / `outside` は vector output を reject し、mask default と color4 alpha scaling を確認した。

- `mix`
  - value mix は float mix broadcast と same-type per-channel mix を扱い、vector4 output を color4 に潰さないようにした。
  - value mix は amount を clamp しない方針を確認した。
  - PBR `mix_bsdf` / `mix_edf` は MDL の `math::saturate` に合わせて mix weight を clamp するようにした。

- conditionals
  - `ifgreater` / `ifgreatereq` は float/integer selector、boolean-output、matrix/integer branch values を確認した。
  - `ifequal` は epsilon comparison を廃止し、StandardNodes と MDL の exact `==` にした。
  - closure `IfEqual` も `le`、sample、eval、pdf、light-tree summary、directional albedo、EDF evaluation、thin-walled transmittance の全 traversal で exact equality にした。
  - `ifequal` default は `value1=0`、`value2=0` とし、`ifgreater` の default を誤用しないようにした。
  - `switch` は branch defaults、matrix outputs、integer selector type を保持し、selector clamp behavior をテストで固定した。
  - shipped stdlib nodegraph と StandardNodes が `which >= 10` で衝突する箇所は StandardNodes clamp semantics を採用する方針を記録した。

- `blur`
  - 未実装のまま passthrough せず、explicit unsupported error にした。

- `heighttonormal`
  - constant height は encoded flat normal `(0.5,0.5,1.0)` にした。
  - dynamic height は derivative/sample-grid 未実装のため error にした。
  - `scale` と `texcoord` は constant-height path でも検証するようにした。
  - stale runtime instruction が実行された場合は panic するようにし、absolute height を gradient とみなす誤った fallback を廃止した。

- logical nodes / `dot`
  - `and`、`or`、`xor`、`not` の boolean defaults と truth table を確認した。
  - `not` は input default false により output default true であることを確認した。
  - value `dot` は boolean/integer/matrix を含む supported value types を passthrough するようにした。
  - closure `dot` は surface/BSDF/EDF closure passthrough とし、missing shader input は zero closure とした。

### 5. Geometry / NPR / procedural nodes

- geometric/application nodes
  - `geomcolor` は vertex color がない場合 white ではなく MaterialX default black を返すようにした。
  - `frame` は host frame/time plumbing がないため nodedef default 1 を返すことを確認した。
  - `time` は nodedef default 0 を返すことを確認した。
  - real `<geompropvalue>` / `<geompropvalueuniform>` は `FlatNodeKind::Geometric` ではなく Pattern node として扱い、`geomprop`、`default`、output type を尊重するようにした。
  - `geompropvalue.geomprop` は strict static string にした。
  - `ViewDirection` は MaterialX NPRSpec/MDL 1.8 に従い camera-to-surface direction を返すようにし、以前の surface-to-camera `sv.wo` 逆向きを修正した。
  - geometric `space` は strict static string にし、dynamic connection を default 扱いしないようにした。
  - `viewdirection` は default world space、他の geometric space nodes は default object space とした。

- NPR
  - `facingratio` は official nodegraph と同じく、`faceforward=true` で `abs(dot(view, normal))`、`false` で `-dot(view, normal)`、`invert` で `1 - value` にした。
  - `gooch_shade` は primitive shading closure として誤分類せず、1.39.4 official color3 nodegraph を使うようにした。
  - native `GoochShadeKernel` は diffuse mix direction と specular term を official nodegraph に合わせた。

- procedural shape/tile nodegraphs
  - `line`、`circle`、`cloverleaf`、`hexagon`、`grid`、`crosshatch`、`tiledcircles`、`tiledcloverleafs`、`tiledhexagons` は native direct lowering ではなく official stdlib nodegraph expansion で実装されることを確認した。
  - nested official nodegraphs が compile まで通ることをテストした。

- ramps / splits
  - `ramp4` は official `NG_ramp4_*` nodegraph と同じ top-to-bottom mix に修正した。
  - `ramplr`/`ramptb` と `splitlr`/`splittb` の omitted right/bottom inputs は nodedef/MDL と同じ zero default にした。
  - `splittb` は StandardNodes/GLSL/OSL と MDL が衝突しているが、監査条件の MDL equivalence に従い、MaterialX 1.39.4 generated MDL の x-axis step を採用した。

- noise / random / checker
  - `worleynoise` distance helpers、top-2/top-3 ordering、metric branch、jitter、neighborhood loops を MDL と照合した。
  - `worleynoise2d/3d.style` と `unifiednoise2d/3d.style` は enum `0=Distance`、`1=Solid` を static integer として処理し、string enum name も compatibility として許容した。
  - invalid style、dynamic style、dynamic `unifiednoise.type` を error にした。
  - `fractal2d/3d` vector2 outputs は MDL と同じ `(fbm(p), fbm(p + offset))` にした。
  - `bits_to_01` は high-bit hash で MDL signed-int conversion に合わせた。
  - `fbm` は `octaves <= 0` で zero を返す MDL empty loop と一致するようにした。
  - runtime `octaves` clamp は min 1 ではなく min 0 にした。
  - `randomfloat` は official nodegraph の cellnoise seed mapping に合わせた。
  - `randomcolor` は seed offsets `ceil(seed + 413.3)`, `ceil(seed + 1522.4)`, `ceil(seed + 1813.8)` と HSV-to-RGB に合わせた。
  - `checkerboard` は UV tiling 後に `uvoffset` を subtract する official nodegraph と一致させた。

- hextile helper
  - hash constants、seed offset、skew/inverse-skew、random rotation sign、scale/offset interpolation、weight normalization、Schlick gain を MDL/GLSL と突合した。
  - local scale/weight guard を削除し、MDL/GLSL と同じ式にした。
  - `normals_to_gradient` の denominator floor を MDL `FLOAT_MIN` 意図に合わせた。
  - `gradient_blend_3_normals` は fallback normal ではなく raw normalize とした。

### 6. PBR BSDF / EDF / VDF / shader / utilities

- diffuse/translucent/Gooch
  - Burley diffuse、Oren-Nayar diffuse、translucent は sample/eval/pdf contract と `sample.weight = f * cos / pdf` を確認した。
  - Burley diffuse の rough behavior と smooth Lambert limit をテストで固定した。
  - Translucent は opposite hemisphere の diffuse transmission と same-side zero を確認した。

- dielectric/conductor/generalized Schlick
  - `dielectric_bsdf` defaults を PBRSpec/pbrlib 1.39.4 に合わせて確認した。
  - `conductor_bsdf` defaults を PBRSpec/pbrlib 1.39.4 に合わせて確認した。
  - `generalized_schlick_bsdf` defaults を PBRSpec/pbrlib 1.39.4 に合わせて確認した。
  - unsupported `distribution`、invalid `scatter_mode` は error にした。
  - optional normal/tangent frame を sample/eval/pdf/light-tree precompute で一貫して使うことを確認した。
  - MaterialX roughness vectors は BSDF layer では GGX alpha として扱い、light-tree へ追加 square なしで渡す方針を確認した。
  - `generalized_schlick_bsdf` / `generalized_schlick_edf` の `exponent` を 1.0 未満で clamp する挙動を廃止した。
  - front/back eta selection と RT branch sampling を確認した。

- sheen
  - `sheen_bsdf` defaults は PBRSpec/pbrlib の `roughness=0.3` を採用し、MDL function default `0.2` より nodedef/spec を優先した。
  - invalid `mode` は default に落とさず error にした。
  - Conty-Kulla と Zeltner path を確認した。
  - Zeltner sheen は未由来の 32x32 table を削除し、MaterialX generated GLSL の rational fits で directional albedo、`aInv`、`bInv` を計算するようにした。
  - roughness clamp `[0.01,1.0]` を generated GLSL と一致させた。

- Chiang hair
  - `chiang_hair_bsdf` defaults は PBRSpec/pbrlib と一致することを確認した。
  - `cuticle_angle` は PBRSpec の `[0,1]` range に clamp して 0.5 を no tilt に map するようにした。
  - `normal` socket を closure に追加し、sample/eval/pdf/light-tree で一貫して適用するようにした。
  - invalid `normal` input は error にした。
  - `curve_direction` と native hair basis を確認した。

- SSS / VDF / measured EDF
  - `subsurface_bsdf` は現制約に従い warning つき Burley diffuse fallback とした。
  - SSS fallback でも `radius` と `anisotropy` は type/connection checked にした。
  - `measured_edf` は warning-level unsupported とし、`uniform_edf` fallback へ進む前に `normal` と `file` を検証するようにした。
  - `absorption_vdf` と `anisotropic_vdf` は warning-level unsupported zero/no-volume とし、declared inputs を検証してから fallback するようにした。

- EDF
  - `uniform_edf`、`conical_edf`、`generalized_schlick_edf` defaults を PBRSpec/pbrlib と照合した。
  - `conical_edf.normal` が runtime `le` で無視されていた問題を修正した。
  - EDF radiance convention は MaterialX radiant emittance と renderer radiance の `pi` convention bridge として確認した。
  - `add_edf` / `mix_edf` は MDL の shape/intensity decomposition と一致させるため、`EdfTerms { shape, intensity }` を導入した。
  - `add_edf` は shape と intensity を別々に足し、`mix_edf` は別々に mix、`multiply_edf` は intensity scaling とした。
  - `closure_max_emission` も同じ shape/intensity model に合わせ、light-tree/emitter registration の underestimation を避けるようにした。

- PBR closure combinators / shaders
  - `add_bsdf` は MaterialX 1.39.4 MDL の equal 0.5/0.5 `df::clamped_mix` 挙動に合わせた。
  - `add_bsdf` の sample PMF、eval、pdf、directional albedo、thin-walled transmittance traversal を MDL 相当にした。
  - `surface.thin_walled` は boolean 以外を false 扱いせず error にした。
  - `surface_unlit.transmission` は MDL と同じく saturate してから emission attenuation と transmission color に使うようにした。
  - `MtlxMaterial::eval` / `pdf` は thin-walled transmission の opposite-side direction に対して zero を返し、sample の straight-through delta transmission と整合させた。
  - volume は警告/no-volume、light node は surface material 内で unsupported error という方針を確認した。

- PBR utilities
  - `roughness_anisotropy` は MDL clamp/aspect formula と roughness squaring に合わせた。
  - `glossiness_anisotropy` は nodedef default glossiness 1.0 を使い、`roughness_anisotropy(1 - glossiness, anisotropy)` とした。
  - `roughness_dual` は vector2 input、negative-y mirroring、per-channel squaring/clamping を扱うようにした。
  - `chiang_hair_roughness` は MDL の variance formula と TT/TRT longitudinal scaling に合わせた。
  - `deon_hair_absorption_from_melanin` は linear pigment mix ではなく MDL の logarithmic melanin/pigment mapping にした。
  - extra clamp していた `melanin_redness` と pigment clamp を削除し、MDL に合わせた。
  - `chiang_hair_absorption_from_color` は color を `[0.001,1.0]` に clamp して logarithmic absorption conversion するようにし、azimuthal roughness の余計な lower clamp は削除した。
  - `blackbody` は renderer-specific Planck sample/luminance normalize ではなく generated GLSL/OSL の Planckian-locus approximation、temperature clamp `[1667,25000]`、XYZ-to-Rec.709、non-negative RGB clamp に合わせた。
  - `artistic_ior` は MaterialX 1.39.4 nodedef default を使い、multi-output selection を検証するようにした。

### 7. Material / Scene / Integrator / Light tree 連携

- `MtlxMaterial`
  - front/back compiled graph selection を `front_face` に基づいて active material から行うよう確認した。
  - `precompute_shading`、`sample`、`eval`、`pdf`、`le`、`any_hit`、light-tree precompute が active front/back graph を使うようにした。
  - `may_emit` は front/back の OR、`max_emission` は front/back max、`has_alpha_test` は front/back opacity/passthrough を合成するようにした。
  - `any_hit` は active front/back material の passthrough/opacity-test を評価するようにした。
  - light-tree precompute は diffuse-only proxy ではなく、diffuse、glossy、BTDF lobes を保持し、importance は合算するようにした。
  - MaterialX `Mtlx` は `is_pure_emitter` false のままとし、conservative emitter を NEE/MIS direct-light sampling から除外しない方針を確認した。

- `Material`
  - top-level material dispatch で `precompute_shading`、`prepare_shading_vertex`、`sample`、`eval`、`pdf`、`le`、`may_emit`、`is_pure_emitter`、`max_emission`、`has_alpha_test`、`any_hit`、light-tree precompute/importance を確認した。
  - native `Emissive` だけを pure emitter とする設計を確認した。

- `Scene`
  - MaterialX scratch capacity が front graph だけでなく back graph `num_registers` も含むようにした。
  - `closest_hit` alpha-test traversal で `MtlxScratch` checkpoint/restore を行うことを確認した。
  - opacity/any-hit traversal では material normal mapping を実行せず、full shading では `prepare_shading_vertex` を使うことを確認した。
  - `ShadingVertex` の `front_face`、face-forwarded `ng`/`ns`、frame、UV/position/normal derivatives、object/world transforms、MaterialX register handles を確認した。
  - emissive triangle registration は `may_emit` / `max_emission` を使い、MaterialX front/back emission を conservative に扱うようにした。
  - area-light area と solid-angle PDF を確認した。

- Integrator / light transport
  - PT は `precompute_shading` 後に `le`/`sample` を呼び、throughput に `sample.weight` を使うことを確認した。
  - NEE/MIS は direct-light estimation を BSDF sampling より前に行い、randomly selected BSDF branch の `DELTA` flag に direct lighting が左右される問題を修正した。
  - pure emitter だけを direct-light sampling から除外するようにした。
  - direct-light contributions で `material.eval` 後に `abs(ns·wi)` を使い、BTDF の below-surface direct lighting を保持するようにした。
  - BSDF-sampled emitter hits、environment hits、area/environment/delta light PDF、reverse solid-angle PDF、target_triangle propagation を確認した。

- Light tree
  - MaterialX light-tree precompute は diffuse/glossy/BTDF lobes を扱うようにした。
  - conductor/dielectric/generalized Schlick の roughness/alpha と directional albedo を使う lobe construction を確認した。
  - multiple glossy/BTDF roughness matrices は reflectance-weighted alpha-squared averaging で merge することを確認した。
  - dielectric transmission を BTDF lobe として保持することを確認した。
  - `may_emit` / `max_emission` による light-tree registration、leaf PMF、reverse PMF、zero importance behavior を確認した。

### 8. 不要コメントと監査メモ整理

- コードを読めば分かるだけのコメントを複数削除した。
- `compiled.rs` の stale operand-layout comments を削除または修正した。
- `compile.rs`、`mtlx_material.rs`、`generalized_schlick.rs`、`mod.rs` などで assertion/test name と重複するコメントを削除した。
- scratch lifetime や `precompute_shading` の非自明な register lifetime contract を説明するコメントは保持した。
- `MATERIALX_AUDIT.md` に Current Audit Index を追加し、古い `Follow-up required` は履歴として残しつつ、現在の未解決項目は Current Open Items に集約した。
- 行数突合で明示範囲が弱かった末尾候補を再読し、`parser.rs` 1259-1451/1451、`resolver.rs` 151-191/191、`flatten.rs` 2191-3253/3253、`build.rs` 561-698/698、`compiled.rs` 1139-1155/1155、`compile.rs` 6296-6332/6332、`runtime.rs` 5166-5178/5178、`generalized_schlick.rs` 404-408/408、`chiang_hair.rs` 502/502 を閉じた。

## 追加・更新したテスト

### Loader / parser / resolver / flatten

- `float_array_rejects_malformed_element`
- `integer_array_rejects_malformed_element`
- `vector_array_rejects_malformed_element`
- `string_array_supports_materialx_escape_convention`
- `malformed_version_is_an_error`
- `missing_required_input_attribute_is_an_error`
- `malformed_geompropdef_index_is_an_error`
- `invalid_geompropdef_semantics_are_errors`
- `malformed_boolean_attribute_is_an_error`
- `ambiguous_input_binding_attributes_are_errors`
- `ambiguous_output_binding_attributes_are_errors`
- `xinclude_missing_href_is_an_error`
- `colorspace_inherits_from_document_and_node_scope`
- `element_namespace_qualifies_nodedef_and_nodegraph_names`
- `xinclude_children_inherit_root_namespace_and_colorspace`
- `xinclude_children_precede_parent_content_and_inherit_fileprefix`
- `token_type_is_required`
- `token_cannot_bind_value_and_interface`
- `explicit_nodedef_must_match_node_category_and_signature`
- `missing_input_interface_errors_instead_of_empty_value`
- `missing_output_interface_errors_instead_of_empty_value`
- `missing_required_nodedef_input_errors_instead_of_zero_default`
- `missing_required_nodegraph_input_errors_instead_of_zero_default`
- `nodegraph_output_without_nodename_errors_instead_of_zero_default`
- `local_nodedef_overload_matches_input_types`
- `target_specific_nodegraph_is_not_used_as_universal_implementation`
- `target_specific_nodedef_is_not_used_as_universal_definition`
- `target_specific_nodedef_is_not_universal_match`
- `nodegraph_for_nodedef_ignores_target_specific_graphs`
- `nodegraph_implementation_precedes_direct_universal_graph`
- `multi_output_node_requires_valid_output_name`
- `custom_multi_output_nodegraph_uses_requested_output`
- `multi_output_nodegraph_reference_requires_valid_output_name`
- `disabled_multi_output_node_uses_requested_output_default`
- `functional_nodegraph_outputs_must_match_nodedef_outputs`
- `functional_nodegraph_child_inputs_are_rejected`
- `nodedef_token_override_substitutes_filename_token`
- `missing_required_nodedef_token_errors`
- `missing_unsupported_input_type_errors`
- `empty_shader_value_flattens_to_empty_connection`
- `volumematerial_flattens_to_empty_passthrough_surface`
- `unsupported_geometry_filename_token_is_preserved`
- `custom_defaultgeomprop_errors_instead_of_texcoord_fallback`
- `geompropvalue_flattens_to_pattern_node`
- `procedural_shape_nodegraphs_flatten_and_compile`
- `triplanarprojection_nodegraphs_flatten_and_compile`
- `gooch_shade_uses_official_color3_nodegraph`
- `flatten_disable_passes_through_default_input`
- `flatten_distance_unit_centimeter_scales_to_meter_base`
- `flatten_unknown_unit_errors`
- `invalid_numeric_literal_errors_instead_of_becoming_string`
- `local_nodedef_version_treats_missing_minor_as_zero`

### Image / texture / colorspace

- `spec_image_float_reads_red_channel_not_luminance`
- `spec_udim_float_uses_scalar_tile_red_channel`
- `color4_use_adds_alpha_when_color3_loaded_same_file_first`
- `unsupported_image_colorspace_uses_temporary_linear_fallback`
- `matches_udim_4digit_mari_style`
- `matches_uvtile_mudbox_style`
- `spec_latlongimage_uses_viewdir_rotation_nodegraph`
- `spec_hextiledimage_color4_preserves_sampled_alpha`
- `spec_image_cubic_and_animated_inputs_warn_without_silent_filter_fallback`
- `spec_image_string_enums_reject_dynamic_or_malformed_values`
- `inherited_srgb_color_value_is_linearized`
- `spec_transformcolor_spaces_reject_dynamic_strings`

### StandardNodes math / matrix / transform / channel

- `spec_absval_sign_floor_ceil_round_fract`
- `spec_modulo_matches_mdl_floor_formula_for_negative_divisor`
- `spec_modulo_is_non_negative`
- `spec_unary_domains_match_mdl_without_clamping`
- `spec_safepower_preserves_negative_sign`
- `spec_floor_integer_output_compiles_to_integer_convert`
- `spec_atan2_in_radians`
- `spec_sin_cos_radians`
- `spec_sqrt_ln_exp`
- `spec_magnitude_vector2_compile_preserves_input_dimension`
- `spec_magnitude_vector4_includes_w_component`
- `spec_dotproduct_vector4_compile_preserves_input_dimension`
- `spec_dotproduct_vector4_includes_w_component`
- `spec_distance_vector4_includes_w_component`
- `spec_min_max_vector_and_float_rhs_are_componentwise`
- `spec_transformnormal_default_is_z_axis`
- `spec_transform_space_defaults_are_object_to_world`
- `spec_transform_spaces_accept_empty_defaults_but_reject_dynamic_strings`
- `spec_transformmatrix_matches_mdl_vector_append_rules`
- `spec_transformmatrix_declared_matrix44_default_selects_m4_overload`
- `spec_geometric_normal_object_space_uses_raw_transform_normal`
- `spec_geometric_normalized_vectors_use_raw_normalize`
- `spec_geometric_space_defaults_and_static_string_validation`
- `spec_matrix_add_compiles_and_evaluates`
- `spec_matrix_divide_multiplies_by_inverse`
- `spec_matrix_transpose_output_compiles`
- `spec_determinant_declared_matrix44_default_selects_m4_overload`
- `spec_creatematrix_vector3_matrix44_sets_mdl_w_components`
- `spec_creatematrix_vector3_matrix44_compiles_with_vec3_rows`
- `spec_extractrowvector_reads_matrix_rows`
- `spec_extractrowvector_static_bad_index_errors`
- `spec_extractrowvector_vector4_compiles_as_vector4`
- `spec_combine2_overload_uses_declared_default_input_type`
- `spec_combine_remaining_overloads_copy_channels`
- `spec_extract_returns_indexed_channel`
- `spec_separate4_vector4_outw_extracts_w_channel`
- `spec_separate_output_names_are_type_specific`
- `spec_multi_output_pattern_requires_valid_output_name`
- `spec_convert_boolean_integer_scalar_rules`
- `spec_convert_boolean_to_integer_output_compiles`
- `spec_convert_runtime_float_to_integer_truncates_like_mdl_constructor`
- `spec_convert_float_to_integer_node_errors`
- `spec_convert_color3_to_color4_adds_alpha_one`
- `spec_convert_vector3_to_vector4_adds_w_one`

### Adjustment / compositing / conditional / utility

- `spec_range_no_clamp`
- `spec_range_doclamp_clamps_input_to_inlow_inhigh`
- `spec_range_doclamp_uses_output_bounds_order`
- `spec_range_doclamp_rejects_dynamic_boolean`
- `spec_smoothstep_cubic_hermite`
- `spec_luminance_instruction_uses_custom_lumacoeffs`
- `spec_luminance_uses_acescg_coefficients`
- `spec_luminance_color4_preserves_alpha`
- `spec_rgbtohsv_low_chroma_matches_mdl_thresholds`
- `spec_rgbtohsv_color4_preserves_alpha`
- `spec_hsvtorgb_color4_preserves_alpha`
- `spec_contrast_color4_float_amount_and_pivot_broadcast`
- `spec_hsvadjust_default_amount_is_noop`
- `spec_hsvadjust_does_not_clamp_saturation_or_value`
- `spec_saturate_uses_luminance_mix_without_clamping_amount`
- `spec_saturate_color4_preserves_alpha`
- `spec_colorcorrect_saturation_uses_luminance_mix_nodegraph`
- `spec_colorcorrect_color4_preserves_alpha`
- `spec_colorcorrect_lift_gain_contrast_matches_nodegraph`
- `spec_premult_unpremult_default_input_alpha_is_one`
- `spec_unpremult_zero_alpha_passes_rgb_through`
- `spec_unpremult_tiny_nonzero_alpha_divides`
- `spec_premult_multiplies_rgb_by_alpha`
- `spec_blend_vector_output_is_rejected`
- `spec_blend_plus_adds_fg_scaled_by_mix`
- `spec_blend_minus_subtracts_fg`
- `spec_blend_remaining_ops_match_standard_formulas`
- `spec_blend_burn_dodge_mdl_epsilon_branches_return_zero`
- `spec_merge_and_mask_vector_outputs_are_rejected`
- `spec_merge_over_porter_duff`
- `spec_merge_remaining_ops_match_mdl_formulas`
- `spec_mask_inside_scales_by_mask`
- `spec_mask_outside_uses_one_minus_mask`
- `spec_mask_color4_scales_alpha_channel_too`
- `spec_mix_vector4_preserves_vector_type`
- `spec_mix_color4_supports_float_and_per_channel_mix`
- `spec_mix_bsdf_clamps_mix_like_mdl`
- `spec_compare_greater_selects_in_true_or_in_false`
- `spec_compare_greatereq_selects_in_true_on_equality`
- `spec_compare_bool_returns_bool`
- `spec_compare_equal_uses_exact_mdl_equality`
- `spec_ifequal_default_values_select_in1`
- `spec_closure_ifequal_uses_exact_mdl_equality`
- `spec_switch_floor_and_clamp_selector`
- `spec_switch_matrix_output_compiles`
- `spec_ifgreater_matrix_output_compiles`
- `spec_ifgreater_integer_output_compiles`
- `spec_ifelse_picks_true_or_false_branch`
- `spec_blur_errors_instead_of_passthrough`
- `spec_heighttonormal_constant_height_is_flat`
- `spec_heighttonormal_dynamic_height_errors`
- `spec_heighttonormal_invalid_scale_errors`
- `spec_logical_and_or_xor`
- `spec_logical_not_default_is_true`
- `spec_boolean_logical_nodes_compile`
- `spec_dot_boolean_output_compiles`
- `spec_dot_matrix44_output_compiles`

### Geometry / NPR / procedural

- `spec_geomcolor_defaults_to_black`
- `spec_geompropvalue_boolean_default_compiles`
- `spec_geompropvalue_geomprop_requires_static_string`
- `spec_viewdirection_matches_materialx_camera_to_surface_direction`
- `spec_facingratio_matches_official_nodegraph_formula`
- `eval_matches_materialx_nodegraph_diffuse_mix_direction`
- `spec_ramp4_mixes_top_to_bottom_like_nodegraph`
- `spec_splittb_uses_mdl_x_axis_step`
- `spec_fractal2d_vector2_uses_scalar_offset_channel`
- `spec_fractal3d_vector2_uses_scalar_offset_channel`
- `bits_to_01_matches_mdl_signed_int_mapping`
- `fbm_zero_octaves_matches_mdl_empty_loop`
- `spec_fractal_zero_octaves_matches_mdl_empty_loop`
- `spec_randomfloat_float_matches_nodegraph_cellnoise`
- `spec_randomfloat_integer_matches_nodegraph_cellnoise`
- `spec_randomcolor_matches_nodegraph_seed_offsets`
- `spec_checkerboard_subtracts_uvoffset`
- `spec_worleynoise_integer_style_one_compiles_to_solid`
- `spec_worleynoise_invalid_integer_style_errors`
- `spec_unifiednoise_worley_uses_integer_style`
- `spec_unifiednoise_connected_type_errors`
- `spec_trianglewave_matches_stdlib_nodegraph`
- hextile 系: `cargo test hextile` で `hextile` helper、hextiled image、hextiled normal map の回帰を確認。

### PBR / BSDF / EDF / VDF / MaterialX material

- `spec_dielectric_bsdf_compile_defaults_match_1394`
- `spec_dielectric_bsdf_invalid_scatter_mode_errors`
- `spec_conductor_bsdf_compile_defaults_match_1394`
- `spec_conductor_bsdf_invalid_distribution_errors`
- `spec_generalized_schlick_bsdf_compile_defaults_match_1394`
- `spec_generalized_schlick_bsdf_invalid_scatter_mode_errors`
- `schlick_exponent_below_one_is_preserved`
- `spec_sheen_bsdf_compile_defaults_match_1394`
- `spec_sheen_bsdf_invalid_mode_errors`
- `zeltner_dir_albedo_matches_materialx_glsl_fit`
- `zeltner_sample_weight_matches_f_cos_over_pdf`
- `zeltner_distinct_from_conty_kulla_at_rough`
- `spec_chiang_hair_bsdf_compile_defaults_match_1394`
- `spec_chiang_hair_bsdf_normal_input_is_checked`
- `cuticle_angle_matches_mdl_radians`
- `spec_chiang_hair_roughness_matches_mdl_variance_formula`
- `spec_deon_hair_absorption_from_melanin_matches_mdl_log_mapping`
- `spec_deon_hair_absorption_from_melanin_does_not_clamp_redness`
- `spec_chiang_hair_absorption_from_color_clamps_color_like_mdl`
- `spec_chiang_hair_absorption_from_color_does_not_clamp_beta`
- `spec_subsurface_bsdf_warns_and_falls_back_to_burley_diffuse`
- `spec_subsurface_bsdf_ignored_inputs_are_type_checked`
- `spec_uniform_edf_compile_defaults_match_1394`
- `spec_conical_edf_compile_defaults_match_1394`
- `spec_conical_edf_uses_normal_socket`
- `spec_measured_edf_warns_and_checks_file_socket`
- `spec_generalized_schlick_edf_compile_defaults_match_1394`
- `spec_add_edf_matches_mdl_unbounded_shape_and_intensity_add`
- `spec_add_edf_max_emission_matches_mdl_shape_intensity_bound`
- `spec_vdf_nodes_warn_to_zero_but_validate_inputs`
- `spec_add_bsdf_matches_mdl_equal_mix`
- `spec_surface_unlit_defaults_emit_white`
- `spec_surface_unlit_saturates_transmission_like_mdl`
- `spec_surface_thin_walled_requires_boolean`
- `spec_empty_surface_root_is_passthrough`
- `thin_walled_standard_surface_is_recognized`
- `thin_walled_transmission_eval_and_pdf_are_delta_zero`
- `thin_walled_back_face_specular_selection_prob_matches_front`
- `front_back_material_flags_are_combined`
- `any_hit_uses_active_back_material_passthrough`
- `light_tree_precompute_keeps_conductor_as_glossy_lobe`
- `light_tree_precompute_keeps_dielectric_transmission_as_btdf_lobe`
- `direct_light_nee_keeps_transmission_below_surface`
- `direct_light_mis_keeps_transmission_below_surface`
- `spec_roughness_anisotropy_matches_mdl_formula`
- `spec_glossiness_anisotropy_inverts_then_squares_roughness`
- `spec_roughness_dual_accepts_vector2_and_mirrors_negative_y`
- `spec_blackbody_matches_generated_glsl_planckian_locus`
- `spec_artistic_ior_compile_defaults_match_nodedef`

## 問題ないと確認した領域

### 仕様書とライブラリ定義

- MaterialX 1.39.4 の 7 つの markdown 仕様書を全範囲で確認した。
- `stdlib_defs.mtlx` 1-5110 行を、image/tiled/hextiled/triplanar、constant、ramp/split、geometry/application/NPR、math/matrix/transform、normal/bump/heighttonormal、adjustment、compositing、conditional、convert/combine/extract/separate、blur、logical、organization `dot` に分けて確認した。
- `pbrlib_defs.mtlx` 1-462 行を、BSDF/EDF/VDF/shader/light/displacement、closure combinator、roughness utility、blackbody、artistic IOR、hair utility に分けて確認した。
- `pbrlib_ng.mtlx`、`nprlib_defs.mtlx`、`nprlib_ng.mtlx` を確認し、`viewdirection`、`facingratio`、`gooch_shade`、`glossiness_anisotropy` の nodegraph と local implementation を照合した。

### Loader / document handling

- `types.rs`、`parser.rs`、`resolver.rs`、`library.rs`、`flatten.rs`、`build.rs` は strict parsing、nodedef resolution、inheritance、XInclude、nodegraph materialization、tokens、file paths、image defaults、colorspace warning、unsupported geometry filename tokens、invalid reference fallback 禁止の観点で確認した。
- parser の残る `unwrap_or` は `fileprefix`、namespace、colorspace など optional inheritance/default scope に限られ、壊れた参照の fallback ではないことを確認した。
- standard library loading は必要ファイル欠落で error になることを確認した。

### Runtime / compile / bytecode

- `compiled.rs` の `Value::Empty` は typed accessor で panic し、silent zero conversion されないことを確認した。
- every instruction operand contract は compile/runtime と対応し、stale operand comments は削除または修正した。
- static string / static boolean / filename sockets は dynamic connection や malformed value を silent default に落とさないことを確認した。
- MaterialX value node の numeric semantics は MDL または StandardNodes/生成 shader 実装と突合した。
- unsupported blur、dynamic heighttonormal/bump、VDF、measured EDF、SSS fallback、cubic/animated image は警告または明示エラーで処理され、invalid connection/reference を黙って通さないことを確認した。

### BSDF / EDF / sampling contract

- Burley diffuse、Oren-Nayar diffuse、Translucent、Dielectric、Conductor、Generalized Schlick、Sheen、Chiang hair は `sample`、`eval`、`pdf`、sample weight、normal/tangent override、roughness/alpha、front/back side、transmission/reflection branch を確認した。
- `sample.weight = f * abs(cos) / pdf` contract と pdf weighting を確認した。
- closure `mix`、`layer`、`add`、`multiply`、conditional/switch traversal が eval/pdf/sample/le/light-tree に同じ選択規則を使うことを確認した。
- EDF の `le` は shape/intensity decomposition によって MDL model と整合することを確認した。

### Integrator / light / scene

- `MtlxMaterial::{sample, eval, pdf, le}` と PT/NEE/MIS integrator の結合を確認した。
- NEE/MIS は mixed MaterialX closure の direct-light sampling を random branch delta flag で落とさないことを確認した。
- area light、environment、point、directional、spot、light tree traversal の PDF と reverse PDF、target triangle propagation を確認した。
- alpha-test/any-hit/passthrough と `MtlxScratch` checkpoint/restore を確認した。
- `ShadingVertex` の `front_face`、`ng`、`ns`、frame、object/world transform、ray differential fallback を確認した。
- MaterialX emitter registration は `may_emit` / `max_emission` によって conservative に行われることを確認した。

### Tests

- `src/material/mtlx/spec_tests.rs` 1-9077 行を再読し、helper、expected values、assertions、unsupported-policy tests、failure cases が仕様/MDL/生成参照実装に合っていることを確認した。
- テストは単なる coverage ではなく、invalid enum、dynamic static socket、unsupported warnings/errors、invalid references/output names、numeric MDL equivalence、roughness utilities、surface_unlit convention、BSDF/EDF mix/add、blackbody、noise、matrix、transform、image、heighttonormal などの期待値自体を確認した。
- 現時点で incorrect test expectation、未対応の newly found issue、silent fallback を許す test hole は見つかっていない。

## 実行した主な検証コマンド

- `cargo fmt`
- `cargo check`
- `cargo test`
- `cargo test scene_loader::mtlx_loader`
- `cargo test material::mtlx`
- `cargo test material::mtlx::spec_tests`
- `cargo test bsdf::mtlx`
- `cargo test light_tree`
- `cargo test scene::`
- `cargo test material::tests`
- 個別 regression test filter: `spec_...`、`thin_walled...`、`direct_light...`、`hextile`、`normalmap`、`bump`、`heighttonormal`、`edf`、`vdf`、`subsurface`、`chiang`、`sheen`、`dielectric`、`conductor`、`generalized_schlick` など。

## 最終判断

- 監査メモ上の Current Open Items はなし。
- 仕様違反として見つかった silent fallback、default mismatch、MDL 数式不一致、socket validation 漏れ、front/back/light-tree/integrator 連携の問題は修正済み。
- volume/light 未対応、SSS Burley fallback、measured EDF/VDF/displacement/animated image/cubic filter/spectral/dynamic heighttonormal などは、監査で定めた許容ポリシーの範囲内で警告または明示エラーとして扱われている。
- color space は sRGB/linear のみを現時点の既知制約として許容し、OCIO 対応まで unsupported spaces は警告つきの一時処理とした。
- 最終の全体テストと対象別テストは成功している。
