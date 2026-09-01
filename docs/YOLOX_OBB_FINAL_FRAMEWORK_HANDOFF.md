# YOLOX-OBB final framework handoff

Status: final framework documentation handoff for Issue [#28](https://github.com/shapovalovei/YOLOX-OBB/issues/28).

This document records the final maintained framework contract and the
cross-project actions that follow from it. The behavioral framework baseline
is `0695e2834ff5716dfd51fcd6ac7b2300b7bc27ce`, the verified `origin/main`
revision before this documentation change. The documentation commit is
separate and changes no framework behavior.

## Executive summary

### Final baseline

`0695e2834ff5716dfd51fcd6ac7b2300b7bc27ce` (`fix: correct OBB MixUp boundary geometry`, merged PR [#42](https://github.com/shapovalovei/YOLOX-OBB/pull/42)) is the final behavioral framework baseline for this handoff. It is the verified current `origin/main` revision and contains the accepted framework work covered below.

### Audit result

The maintained framework now has documented and tested fixes for:

- OBB label geometry at augmentation boundaries;
- finite, positive OBB target validity;
- numerically stable KLD loss arithmetic;
- training-only decoded-dimension gradients;
- classification of CUDA assignment fallback errors;
- dynamic raw ONNX export, DOTA evaluation, and rotated-IoU packaging;
- optional MixUp visible-polygon geometry.

Earlier accepted OBB work also established rotated assignment geometry,
angle-bin geometry, KLD argument order, OBB label preservation through MixUp,
singleton postprocess behavior, NumPy 2 compatibility, and CPU retry after a
genuine assignment CUDA OOM.

### Deferred or not proven

The following are not framework-completion claims:

- true projective perspective semantics;
- supported CUDA autocast FP16 behavior on the tested T4/PyTorch baseline;
- QNN, Android GPU, and Core ML target validation;
- model-quality impact of the framework fixes;
- a checkpoint, final mobile artifact, or SDK device qualification under the
  final framework baseline.

### Critical interpretation

Framework correctness evidence is not the same as model-quality evidence.
Several changes are plausibly quality-relevant, but improvement or loss must
be measured by a new controlled training experiment. Historical checkpoints
remain useful baselines; they are not equivalent to a model trained under the
final framework semantics.

## Scope and evidence convention

The framework owns generic OBB geometry, training and inference behavior,
export, evaluation adapters, and packaging. The training repository owns a
concrete checkpoint and its conversion/provenance. The SDK repository owns
artifact packaging, runtime integration, source mapping, device qualification,
and release decisions.

Statements in this handoff use the following classifications where uncertainty
matters:

- **FACT** — directly supported by merged framework history, maintained tests,
  closed Issue handoffs, or current public downstream repository evidence.
- **INFERENCE** — an engineering consequence of those facts, not a measured
  model-quality result.
- **UNVERIFIED** — evidence still required before the corresponding claim or
  action is accepted.

The material framework research and validation trail includes the closed
framework audit handoff [#23](https://github.com/shapovalovei/YOLOX-OBB/issues/23),
boundary-geometry validation [#24](https://github.com/shapovalovei/YOLOX-OBB/issues/24),
KLD validation [#25](https://github.com/shapovalovei/YOLOX-OBB/issues/25),
assignment-fallback validation [#26](https://github.com/shapovalovei/YOLOX-OBB/issues/26),
assignment-memory validation [#27](https://github.com/shapovalovei/YOLOX-OBB/issues/27),
and completed CUDA assignment validation [#44](https://github.com/shapovalovei/YOLOX-OBB/issues/44),
and perspective validation [#32](https://github.com/shapovalovei/YOLOX-OBB/issues/32).
The merged PRs and exact revisions are recorded in the ledger.

## Final OBB contract

### Dataset and target representation

The DOTA annotation path emits the six-field source row:

```text
[xmin, ymin, xmax, ymax, angle_deg, class_id]
```

The first four values are the axis-aligned envelope encoding of an oriented
rectangle: its center plus or minus the canonical long/short extents divided
by two. They are not the four vertices of an axis-aligned image rectangle and
must not be treated as polygon corners. The angle is in degrees.

After preprocessing, padded training labels use:

```text
[class_id, center_x, center_y, width, height, angle_deg]
```

where `width` is the canonical long side and `height` is the canonical short
side. Class zero and angle zero are valid values; row validity is not inferred
from a positive sum.

The maintained OBB geometry path converts transformed corners to a canonical
minimum-area rectangle, selecting long side then short side and normalizing
the angle to the maintained `[-90, 90)` representation. The prediction
representation is:

```text
[center_x, center_y, width, height, angle_deg]
```

with class fields following the objectness field in a prediction row.

The maintained framework image preprocessing resizes by the minimum ratio to
the requested input dimensions, places the resized image at the top-left of a
114-filled canvas, converts BGR to RGB, applies `/255` plus the configured
mean/std normalization, and returns float32 NCHW data. A concrete downstream
artifact may move some of these operations into a wrapper; that is a separate
artifact contract.

Horizontal augmentation candidate checks remain an HBB filter for historical
compatibility. The surviving OBB itself is derived from the visible polygon.

### Angles and flips

The geometry helpers and KLD loss use degree-valued angles at their public
boundary and convert to radians only inside trigonometric numerical
calculations. Horizontal and vertical flips preserve the rectangle while
negating the angle, with the established `90`-degree boundary handling.
Predicted angles are decoded from the angle logit to the corresponding
degree-valued interval. Canonicalization, rather than an arbitrary width/height
swap, establishes the final long/short geometry.

### Visible-polygon augmentation

For transformed OBBs, the maintained operation is:

```text
actual transformed OBB polygon
  -> intersection with the image canvas
  -> canonical minimum-area OBB
```

This is the contract implemented by PR [#31](https://github.com/shapovalovei/YOLOX-OBB/pull/31), merge `40bac1a4f7a48b2f89cdc3f29a5e6d7fe3bef823`, for perspective/affine augmentation and by PR [#42](https://github.com/shapovalovei/YOLOX-OBB/pull/42), merge `0695e2834ff5716dfd51fcd6ac7b2300b7bc27ce`, for the optional MixUp path. Empty or degenerate visible polygons are rejected. This prevents pointwise vertex clipping from inventing a box that does not describe the visible object.

### KLD contract

The asymmetric KLD call is prediction first and target second:

```text
D_KL(prediction || target)
```

The maintained implementation uses degree-valued angles at its interface,
log-domain stable arithmetic, and promotes FP16/BF16 sensitive calculations to
FP32. The supported domain is finite, positive target geometry and the
validated stride-8 raw interval `[-92, 46]` recorded in [Issue #33](https://github.com/shapovalovei/YOLOX-OBB/issues/33); behavior outside the underlying numerical limits is not claimed. The implementation does not use a hard gradient-killing floor/cap or `nan_to_num` repair. Exact-zero decoded dimensions still have the limiting behavior of the underlying exponential and do not acquire a fabricated gradient.

### Decode boundaries

The framework has three intentionally distinct decode paths.

#### Training decode

The differentiable training path decodes width and height as:

```text
exp(raw_wh + log(stride))
```

This is mathematically equivalent to `exp(raw_wh) * stride` in ordinary
finite arithmetic, but avoids an avoidable FP32 backward overflow near
positive subnormal decoded dimensions. This change is limited to training
gradient computation.

#### Eager inference decode

The maintained eager inference path remains:

```text
xy = (raw_xy + grid) * stride
wh = exp(raw_wh) * stride
angle_deg = (sigmoid(angle_logit) - 0.5) * 180
```

Its literal formulation was intentionally unchanged by the training-gradient
fix.

#### External raw decoder

An exported raw model is decoded outside the neural-network graph using the
same grid/stride equations. Objectness and class fields in the maintained
export are already probabilities; the external consumer must not apply a
second sigmoid to those fields. Rotated polygon construction and rotated NMS
remain outside the raw neural-network graph.

### Raw ONNX export

Static export remains the default. The opt-in dynamic export supports dynamic
batch and square height/width dimensions, with positive dimensions divisible
by 32. The maintained tests cover 320, 416, and 512 square inputs and batch 2
at 416. Non-square dynamic export and additional backend behavior are not part
of the proven contract.

The exported raw output has shape:

```text
[B, N, 6 + C]
```

and row fields:

```text
[tx, ty, tw, th, angle_logit, objectness_probability,
 class_probability_0, ...]
```

For strides 8, 16, and 32, `N` is the sum of the square feature-map areas.
For example, 416 produces `N=3549`. Export disables decode inside the model;
the raw output contract is therefore independent of the training-only decode
arithmetic.

The dynamic export work is PR [#12](https://github.com/shapovalovei/YOLOX-OBB/pull/12), merge `5b5a80eb6e2530f4b71ad7d753666e2b96b3e107`.

The rotated-IoU packaging work proves the native extension is carried through
editable, wheel, and source-distribution installs in the validated packaging
environment. A downstream framework training/evaluation environment that
imports the native rotated IoU/NMS implementation must build or install that
extension. The extension is not part of the ONNX/TFLite neural-network graph;
a mobile consumer may keep decode and rotated NMS in its own external runtime.
Cross-platform native-build parity is not established by this framework
handoff.

### Postprocess and NMS

Raw export does not promise a rotated postprocess graph. External consumers
own grid decode, score composition as required by their artifact contract,
polygon construction, and rotated NMS. The framework’s DOTA result writer
also remains an external adapter: it writes the eight polygon coordinates in
the DOTA result format, applying the maintained coordinate offset required by
that format, and does not fabricate AP values.

### Input shape and layout boundaries

The framework ONNX boundary is NCHW, with the static default and the opt-in
square dynamic contract described above. A downstream TFLite/LiteRT artifact
may expose a different layout and preprocessing contract. Framework NCHW
export semantics must not be conflated with a concrete mobile artifact’s
NHWC, dtype, normalization, or wrapper behavior.

## Framework change ledger

The table distinguishes code correctness from downstream quality evidence.
“Quality impact unmeasured” means the framework behavior is tested but no
trained-model comparison was established by that framework work.

| Change | Issue / PR / merge | Previous behavior | Final behavior | Training correctness | Inference/runtime | Export/API impact | Quality evidence |
|---|---|---|---|---|---|---|---|
| Earlier OBB geometry baseline | [Issue #1](https://github.com/shapovalovei/YOLOX-OBB/issues/1); commits `dc89feea825be653bc48e391dc53cac02216bfdc`, `cf29a90929e29391805ddda9cac3882b583a2a66`, `1602f7de25d97a473cf2530383b0661b300889ab`, `06944aa62505dbaf20fa946f1b8e7c58acea646f` | Rotated assignment, angle-bin, MixUp-label, and singleton paths had contract gaps | Rotated candidate geometry, angle-bin geometry, OBB label preservation, and singleton postprocess are regression-covered | Yes for affected training paths | Yes for singleton postprocess | No raw layout change | Correctness proven; quality impact unmeasured |
| Earlier compatibility fixes | [Issue #4](https://github.com/shapovalovei/YOLOX-OBB/issues/4); commit `5f8b2832c013ba03828e2f60f29cdb85471de7a6` | Rotated postprocess depended on NumPy behavior that changed in NumPy 2 | Finite integer polygon conversion and output-shape handling are explicit | No | Yes | No | Not applicable |
| KLD argument order | [Issue #3](https://github.com/shapovalovei/YOLOX-OBB/issues/3); commit `791b40dbe80bf53662e3f4659194ceb0dd032a54` | Asymmetric KLD operands were reversed | Prediction is the first operand and target the second | Yes | No | No | Correctness proven; quality impact unmeasured |
| Dynamic raw ONNX export | [Issue #5](https://github.com/shapovalovei/YOLOX-OBB/issues/5), [PR #12](https://github.com/shapovalovei/YOLOX-OBB/pull/12), `5b5a80eb6e2530f4b71ad7d753666e2b96b3e107` | Export was static by default | Static default retained; opt-in dynamic B/H/W supports square dimensions divisible by 32; raw decode remains external | No | Yes | Raw shape/layout contract documented; non-square dynamic behavior unproven | Correctness proven; model quality not applicable |
| DOTA evaluator | [Issue #13](https://github.com/shapovalovei/YOLOX-OBB/issues/13), [PR #16](https://github.com/shapovalovei/YOLOX-OBB/pull/16), `1f56536162f7172e7d01c027441e7e907b143891` | Evaluation could imply internal metrics that were not actually computed | Real external DOTA result generation/evaluation adapter; CPU-safe behavior; no fabricated AP or best-checkpoint claim | No | Infrastructure | Evaluation return contract is explicit | Correctness proven; quality not claimed |
| Rotated-IoU packaging | [Issue #14](https://github.com/shapovalovei/YOLOX-OBB/issues/14), [PR #17](https://github.com/shapovalovei/YOLOX-OBB/pull/17), `13815a864061e167d300063f06a3ed9d62d70d34` | Native rotated-IoU extension was not reliably included in distributions | Editable, wheel, and sdist packaging include the extension sources and package path; native fixtures cover IoU/NMS | No | Runtime dependency availability | Packaging only; no model I/O change | Correctness proven; quality not applicable |
| Visible-polygon augmentation | [Issue #24](https://github.com/shapovalovei/YOLOX-OBB/issues/24), [Issue #29](https://github.com/shapovalovei/YOLOX-OBB/issues/29), [PR #31](https://github.com/shapovalovei/YOLOX-OBB/pull/31), `40bac1a4f7a48b2f89cdc3f29a5e6d7fe3bef823` | Boundary OBB vertices were clipped independently, producing geometrically false survivors | Actual visible convex polygon is intersected with the canvas and refit to a canonical minimum-area OBB | Yes; label population and geometry can change | No inference output change | Six-field source-label contract preserved | Correctness proven; quality impact unmeasured |
| OBB target validity | [Issue #34](https://github.com/shapovalovei/YOLOX-OBB/issues/34), [PR #35](https://github.com/shapovalovei/YOLOX-OBB/pull/35), `f3cc029c2a0fb36ff3607c6fe3e6efc342d47fe0` | Invalid fallback rows could reach assignment/KLD; validity was inferred from row sums | Finite positive width/height checks are enforced in preprocessing and head defense; no silent repair of malformed rows | Yes; invalid labels are excluded rather than repaired | No | Training label contract clarified; public prediction layout unchanged | Correctness proven; quality impact unmeasured |
| KLD numerical stability | [Issue #25](https://github.com/shapovalovei/YOLOX-OBB/issues/25), [Issue #33](https://github.com/shapovalovei/YOLOX-OBB/issues/33), [PR #36](https://github.com/shapovalovei/YOLOX-OBB/pull/36), `69c2c02cd5a7d8ae84bbce47834a22b973db12e5` | Valid extreme inputs could produce unstable values/gradients; hard floors/caps were considered | Log-domain arithmetic, degree-aware angles, FP32 promotion for FP16/BF16 sensitive arithmetic, and no hard gradient-killing floor/cap | Yes | No ordinary inference contract change | No raw export change | Correctness proven; quality impact unmeasured |
| Training decode gradients | [Issue #37](https://github.com/shapovalovei/YOLOX-OBB/issues/37), [Issue #38](https://github.com/shapovalovei/YOLOX-OBB/issues/38), [PR #39](https://github.com/shapovalovei/YOLOX-OBB/pull/39), `c8d8b5611cab1b2f706a1638f53891b77b93dce0` | `exp(raw)*stride` could overflow an intermediate in backward near positive subnormal dimensions | Training uses `exp(raw + log(stride))`; eager inference and external raw decode remain literal `exp(raw)*stride` | Yes | No eager inference change | Raw ONNX contract unchanged | Correctness proven; quality impact unmeasured |
| CUDA assignment fallback classification | [Issue #26](https://github.com/shapovalovei/YOLOX-OBB/issues/26), [Issue #40](https://github.com/shapovalovei/YOLOX-OBB/issues/40), [PR #41](https://github.com/shapovalovei/YOLOX-OBB/pull/41), `4f577cac81d7b6a796461ff6a4713c392b651d1c` | Any `RuntimeError` could trigger cache clearing and CPU retry | Only validated modern or legacy CUDA OOM signatures retry on CPU; ordinary errors propagate | No successful-assignment change | Failure semantics only | No | Correctness proven; no model-quality claim |
| Optional MixUp geometry | [Issue #30](https://github.com/shapovalovei/YOLOX-OBB/issues/30), [PR #42](https://github.com/shapovalovei/YOLOX-OBB/pull/42), `0695e2834ff5716dfd51fcd6ac7b2300b7bc27ce` | Cropped axis-aligned envelope could retain an angle that did not describe the visible object | Transformed corners are clipped as a visible polygon and refit to a canonical OBB | Yes only when MixUp is enabled | No | Six-field contract and compositing behavior preserved | Correctness proven; no maintained-recipe effect while MixUp is disabled |
| Perspective validation | [Issue #32](https://github.com/shapovalovei/YOLOX-OBB/issues/32) | Non-zero parameter selected a perspective warp without adding projective terms | Defect documented; no implementation performed | No current recipe effect | Image warp path differs, but no projective label transform exists | No API change | Defect proven; quality impact unmeasured |
| Assignment memory/equivalence validation | [Issue #27](https://github.com/shapovalovei/YOLOX-OBB/issues/27), [Issue #44](https://github.com/shapovalovei/YOLOX-OBB/issues/44), [final T4 handoff](https://github.com/shapovalovei/YOLOX-OBB/issues/44#issuecomment-5484241937) | GPU behavior was initially inferred from CPU concerns | CPU scaling was supplemented by bounded T4 CUDA VRAM/latency/OOM and GPU/CPU equivalence validation; no framework implementation change was needed | No implementation | Bounded CUDA behavior validated; FP16 autocast limitation recorded separately | No | Assignment correctness/resource evidence; no model-quality claim |

The compact representative from [#27](https://github.com/shapovalovei/YOLOX-OBB/issues/27)
is a CPU-only 416-pixel, 16-class dense-assignment matrix: increasing target
density `G` from 25 to 150 increased median latency from 192.9 ms to 1329.5 ms
and delta RSS from 26.4 MiB to 134.8 MiB. The completed [#44 T4 validation
handoff](https://github.com/shapovalovei/YOLOX-OBB/issues/44#issuecomment-5484241937)
now supplies the corresponding bounded CUDA evidence.

For assignment fallback, the accepted modern signal is
`torch.cuda.OutOfMemoryError`; the validated legacy signals are `RuntimeError`
messages beginning `CUDA out of memory` or `CUDA error: out of memory`. CPU,
MPS, generic memory, and unrelated runtime errors do not enter the retry path.

## CUDA assignment validation (#44)

Issue [#44](https://github.com/shapovalovei/YOLOX-OBB/issues/44) is closed as
Completed with accepted Verdict A. It completed the missing CUDA evidence on
the exact behavioral baseline
`0695e2834ff5716dfd51fcd6ac7b2300b7bc27ce`. The detailed factual evidence is
in the [final T4 handoff](https://github.com/shapovalovei/YOLOX-OBB/issues/44#issuecomment-5484241937),
accepted by the orchestrator in [comment 5484274489](https://github.com/shapovalovei/YOLOX-OBB/issues/44#issuecomment-5484274489).

### Environment and workload

- `torch.cuda.is_available(): True`; GPU: Tesla T4; VRAM: 15,360 MiB.
- Python: 3.13.15; PyTorch: 2.11.0+cu128; `torch.version.cuda`: 12.8.
- Primary 416: 3,549 anchors (`52^2 + 26^2 + 13^2`), FP32, `C=1,16`,
  `G=1,5,10,25,50,100,150`, deterministic sparse and dense regimes.
- All 28/28 primary cells completed. The reproduced sparse `K` sequence was
  `76, 375, 680, 1440, 2044, 2853, 2789`; dense `K` was 3,549 for every `G`.
- Secondary 1024: 21,504 anchors (`128^2 + 64^2 + 32^2`), `C=16`, selected
  `G=1,10,25,50` in sparse and dense regimes; all 8/8 selected FP32 cells
  completed. Sparse `K` was `75, 793, 1986, 3602`; dense `K` was 21,504 for
  every selected `G`.

### Memory, latency, and OOM

Timing used warmup and synchronized median measurements; allocator peaks were
reset between cells. No natural CUDA OOM occurred in any of the 36 bounded
FP32 cells, and no OOM was forced. The maximum 416 peak was 107.255 MiB
allocated / 168 MiB reserved at dense `G=150, C=16`; the maximum selected
1024 peak was 218.871 MiB / 344 MiB at dense `G=50, C=16`. The largest
synchronized medians were 705.331 ms for the 416 matrix and 199.202 ms for
the selected 1024 matrix. The corresponding peak allocated/reserved deltas
were 104.449 / 144 MiB and 214.084 / 320 MiB. These are bounded synthetic
stress measurements, not a product latency SLA or training-throughput claim.

### FP32 GPU/CPU fallback equivalence

For identical deterministic inputs, `get_assignments(mode="gpu")` and
`get_assignments(mode="cpu")` produced matching shapes, dtypes, devices, and
ordering. In the stable cases 416 `C=1,G=10` sparse, 416 `C=1,G=25` dense,
416 `C=16,G=25` dense, and 416 `C=16,G=100` dense, `num_fg`, `fg_mask`,
`matched_gt_inds`, and `gt_matched_classes` were exact;
`pred_ious_this_matching` was allclose, with maximum observed stable absolute
difference `1.788e-7`.

The bounded near-tie stress produced one 4-index `fg_mask` divergence at
indices `1272, 1274, 1326, 1327`. `num_fg` remained equal, matched GT fields
remained equal, and the relevant cost margin was zero/quantized; the cost
maximum absolute difference was `3.576e-7`. This is expected floating-point
tie sensitivity, not a fallback correctness defect.

### FP16 autocast

The maintained true CUDA autocast path was tested for 416 dense `C=1,G=25`,
`C=16,G=25`, and `C=16,G=100`. On this PyTorch/runtime it is unsupported:
`binary_cross_entropy` / `BCELoss` triggers the PyTorch autocast safety guard
before cost completion. This is not an OOM and not a GPU/CPU fallback
correctness defect. No source workaround was applied, and AMP training must
not be claimed as supported by this evidence.

As a separate direct-FP16-input diagnostic, all three selected cases completed
with FP32 KLD/BCE/cost intermediates and equivalent GPU/CPU fallback outputs:

| C | G | synchronized median ms | peak allocated/reserved MiB | major KLD/class/cost dtypes |
|---:|---:|---:|---:|:---|
| 1 | 25 | 96.119 | 12.748 / 30 | float32 |
| 16 | 25 | 96.951 | 21.690 / 30 | float32 |
| 16 | 100 | 372.644 | 74.747 / 116 | float32 |

The diagnostic does not substitute for a supported autocast result.

### Tests and evidence classification

The required focused suites passed: assignment fallback 8/8, OBB assignment
geometry 5/5, KLD prediction/target 5/5, KLD numerical stability 18/18, and
OBB target validity 11/11. Full discovery reported 101 total, 91 passed,
6 skipped, 2 known `_polyiou` import errors, and 2 unrelated MixUp failures
under Python 3.13/PyTorch 2.11. MixUp is disabled in the current controlled
training recipe; those full-suite failures are outside #44 acceptance and
were not repaired.

**FACT:** The T4 gate passed, all bounded FP32 cells completed without natural
OOM, stable fallback outputs were equivalent, and the focused suites passed.

**INFERENCE:** Current bounded FP32 CUDA assignment behavior is acceptable;
the evidence does not justify an assignment-memory optimization phase or a
fallback-correctness implementation phase. The near-tie result is explained
by zero/quantized cost margins.

**UNVERIFIED:** Model quality, convergence, training throughput, behavior
beyond the tested matrix, and a supported CUDA autocast FP16 result remain
unestablished.

### Final #44 decision

**A — CURRENT CUDA BEHAVIOR ACCEPTABLE.** No assignment-memory optimization
phase is justified by the bounded evidence, and no fallback correctness
implementation phase is required. The true CUDA FP16 autocast limitation is
recorded for compatibility awareness, not promoted automatically to a new
implementation recommendation. See the [orchestrator acceptance](https://github.com/shapovalovei/YOLOX-OBB/issues/44#issuecomment-5484274489).

## Earlier quality-relevant framework fixes

The pre-audit OBB baseline remains part of the final framework contract. The
material accepted changes are:

- OBB-aware rotated candidate geometry in assignment (`dc89feea825be653bc48e391dc53cac02216bfdc`);
- CPU retry after the earlier assignment CUDA-OOM path (`a0bcf4a284441ba88a8618dc96b5958a8b60d418`), now
  narrowed by [PR #41](https://github.com/shapovalovei/YOLOX-OBB/pull/41);
- singleton rotated postprocess shape handling (`06944aa62505dbaf20fa946f1b8e7c58acea646f`);
- angle-bin geometry and angle preservation (`cf29a90929e29391805ddda9cac3882b583a2a66` and `1602f7de25d97a473cf2530383b0661b300889ab`);
- OBB-aware augmentation geometry from [Issue #2](https://github.com/shapovalovei/YOLOX-OBB/issues/2) (`80b48ea90bb281948af76bf5e9d86eb1e671586c`), later completed for visible
  boundary polygons by [PR #31](https://github.com/shapovalovei/YOLOX-OBB/pull/31);
- corrected prediction/target KLD order (`791b40dbe80bf53662e3f4659194ceb0dd032a54`); and
- NumPy 2 rotated postprocess compatibility (`5f8b2832c013ba03828e2f60f29cdb85471de7a6`).

These are framework correctness or infrastructure changes. They do not by
themselves establish that a trained detector improves.

## Training-project handoff

### Which final changes can alter training?

| Framework change | Training consequence |
|---|---|
| Visible-polygon augmentation [#29](https://github.com/shapovalovei/YOLOX-OBB/issues/29) | Changes surviving boundary-label geometry and label population. Quality-relevant. |
| Target validity [#34](https://github.com/shapovalovei/YOLOX-OBB/issues/34) | Excludes malformed/non-positive rows before assignment and KLD instead of repairing them. Quality- and stability-relevant. |
| KLD stability [#33](https://github.com/shapovalovei/YOLOX-OBB/issues/33) | Changes loss arithmetic and extreme-value gradients while preserving the intended KLD direction. Quality- and stability-relevant. |
| Training decode [#38](https://github.com/shapovalovei/YOLOX-OBB/issues/38) | Changes a training-only backward arithmetic path near subnormal dimensions. Quality- and stability-relevant. |
| Optional MixUp [#30](https://github.com/shapovalovei/YOLOX-OBB/issues/30) | Changes labels only if the recipe explicitly enables MixUp. The maintained DOTA policy has `enable_mixup=False`. |
| Earlier assignment/angle geometry | Applies to the affected OBB training paths; a new final-framework run includes the corrected behavior. |
| Assignment error classification [#40](https://github.com/shapovalovei/YOLOX-OBB/issues/40) | Does not change successful ordinary assignments. It prevents unrelated runtime errors from being misclassified as CUDA OOM. |
| Evaluator, packaging, and dynamic export | Infrastructure or artifact-interface work; does not alter training labels or loss semantics. |

### Retraining decision

The decisions below concern producing a model genuinely trained under the
final framework semantics, not merely executing an existing checkpoint.

| Change | Decision |
|---|---|
| [#29](https://github.com/shapovalovei/YOLOX-OBB/issues/29), visible-polygon labels | **Required** for a final-semantics candidate when boundary examples are in the training data. |
| [#34](https://github.com/shapovalovei/YOLOX-OBB/issues/34), target validity | **Required** for a final-semantics candidate. |
| [#33](https://github.com/shapovalovei/YOLOX-OBB/issues/33), KLD arithmetic | **Required** for a final-semantics candidate. |
| [#38](https://github.com/shapovalovei/YOLOX-OBB/issues/38), training decode | **Required** for a final-semantics candidate. |
| [#30](https://github.com/shapovalovei/YOLOX-OBB/issues/30), optional MixUp | **Only if enabled**; otherwise no separate retraining requirement from this item. |
| Earlier corrected assignment/angle paths | **Recommended** to include in the same fresh controlled run; separate retraining is not attributed to infrastructure-only fixes. |
| [#40](https://github.com/shapovalovei/YOLOX-OBB/issues/40), fallback classification | **Not required** as a separate quality experiment; pin the final revision for future runs. |
| Evaluator, packaging, dynamic export | **Not required** to retrain. |

Overall recommendation: run a fresh controlled training experiment pinned to
the final behavioral framework revision before making a final-framework model
quality claim. Training is a downstream future phase and is not being started
as part of Issue #28 or PR #43. Do not assume that the new run will outperform
the historical champion.

### Historical training provenance

The public `card-detector-training` repository was checked at main commit
`9ad5eacc888faa29c51c695176615fef5d08da8c`. Its durable [EXP02 experiment
record](https://github.com/shapovalovei/card-detector-training/blob/9ad5eacc888faa29c51c695176615fef5d08da8c/docs/experiments/exp02.md),
[patched-assignment record](https://github.com/shapovalovei/card-detector-training/blob/9ad5eacc888faa29c51c695176615fef5d08da8c/docs/experiments/exp02_patched_assignment_ab.md),
and [mobile export records](https://github.com/shapovalovei/card-detector-training/tree/9ad5eacc888faa29c51c695176615fef5d08da8c/docs/experiments)
establish the following:

- The original EXP02 scratch experiment
  `20260823_yolox_obb_nano_416_combined_scratch_adamw_exp02` used framework
  source `5f8b2832c013ba03828e2f60f29cdb85471de7a6`, training project commit
  `5cc1baf920af7c51aea43155ea6d68624f77a94e`, a 416-pixel experimental
  Nano-style detector, one card-number class, and a recipe with MixUp off.
  Its dataset archive hash is
  `1a94ede727e5a1e1844a3ac68a3e2bd221467df8ada5006b42af791341e55537`.
  The original EMA checkpoint remains the recorded historical champion with
  SHA-256
  `a668c839dfb1ad58019032f57b91954b3fc54e72853c4b632770479b1d56b201`.
- The patched assignment continuation ([training Issue #1](https://github.com/shapovalovei/card-detector-training/issues/1))
  `20260827_yolox_obb_nano_416_exp02_patched_assignment` used source
  `9528bf1b574d8bf3c3979d3eaec3293a6f4116fc`, reached a partial later phase,
  and was paused. Its authoritative recorded epoch-34 checkpoint hash is
  `0437d0396571ae9f22264feb48c0e2910c16b3c39bec58c231cec9708964ec3c`.
  It did not produce a new champion or complete the durable real-world
  validation gate.
- The recorded V4 comparison is historical evidence, not a promoted final
  model. The controlled run `20260827_yolox_obb_mix_exp02_v4_4k` has the
  durable verdict **MIX LOSES** against the frozen EXP02 champion on primary
  mean-localization quality. Its public provenance also predates the final
  fixes in #29, #34, #33, and #38.
- The public export records trace the historical EXP02 artifacts to source
  `5f8b2832c013ba03828e2f60f29cdb85471de7a6`. INT8/PTQ candidates were not
  adopted: the recorded Percentile candidate failed the predefined quality
  gates against the frozen FP32 control. No QAT result is established.

These records are sufficient to classify historical checkpoints as valid
historical baselines, but not as models trained under the final framework.
Where a historical metric is not recorded in the cited durable repository
documents, it is not established here.

### Future training baseline

New final-framework training must pin:

```text
0695e2834ff5716dfd51fcd6ac7b2300b7bc27ce
```

This is the behavioral framework baseline. The current documentation branch
and its later merge commit will contain only Markdown changes, so the training
repository may use the behavioral SHA above as the reproducibility pin. Its
training provenance metadata must be updated in the training repository as a
separate change; this handoff does not modify that repository.

### Recipe policy

No recipe change is implied by this handoff. The durable EXP02 and current
controlled training recipe records MixUp disabled, and the inspected
maintained recipes use `perspective=0.0`. Keep those settings for a controlled
comparison unless a separate experiment authorizes a recipe change. Enabling
MixUp or perspective is not a consequence of the framework fixes.

### Post-retrain validation

The training repository should use its established sequence:

1. Run the normal validation used for checkpoint selection.
2. Evaluate the selected candidate with the durable `real-val160` at 416
   contract where that remains the applicable experiment gate.
3. Compare with the same historical champion and preserve the same dataset,
   recipe, and evaluation definitions.
4. Report geometry/localization metrics and failure/ranking analysis already
   maintained by the training project; do not substitute framework unit tests
   for model-quality evidence.
5. Record training stability, non-finite loss/gradient absence, assignment
   failure classification, and available target-validity counters as run
   evidence.

## Export ownership

### YOLOX-OBB owns

- generic raw export correctness;
- static and opt-in dynamic ONNX framework behavior;
- generic decode and postprocess semantics; and
- framework-level rotated geometry and NMS correctness.

### card-detector-training owns

- the concrete trained checkpoint;
- FP32/FP16 mobile export;
- TFLite/LiteRT conversion;
- INT8/PTQ/QAT and calibration;
- artifact hashes and conversion provenance; and
- model-quality and model-side parity evidence.

## SDK-project handoff

### Framework I/O contract impact

The recent quality-relevant fixes do not change the raw model’s input shape,
raw output shape, raw field order, degree angle semantics, external decode, or
rotated-NMS boundary. The training decode change is deliberately not an eager
inference or ONNX raw-output change. Target validity and visible-polygon
augmentation affect how a future checkpoint is trained, not how an existing
raw artifact is decoded.

The dynamic export feature is opt-in and does not change the static default.
It adds a square dynamic ONNX option; it does not authorize a mobile consumer
to assume non-square inputs or a different output layout.

**FACT:** training-only framework changes matter to the SDK after a newly
trained/exported artifact is adopted, unless a framework inference/export
contract itself changes. The fixes in this handoff do not by themselves
require SDK source changes.

### Current and historical mobile artifacts

The public SDK mainline was checked at commit
`dc00d57f66f14741f1639ceb6c89bf4833b4f2b9`. Its current public default is
`card-detector-yolo26-v1`, not EXP02. The [current model manifest](https://github.com/shapovalovei/react-native-scanner-sdk/blob/dc00d57f66f14741f1639ceb6c89bf4833b4f2b9/models/model-manifest.json)
records this current artifact as:

| Field | Current SDK mainline value |
|---|---|
| Input | RGB `uint8`, NHWC `[1,640,640,3]` |
| Output | Float32 `[1,6,8400]`, `ClassicAnchors6x8400` channel-major |
| Output fields | `[cx, cy, w, h, confidence, angle]` |
| Angle | Radians |
| Artifact | `card-detector-yolo26-v1-u8-f32.tflite` |
| SHA-256 | `a3a61ec82d900e59a990ca6a3101af6c9b6dfce1fb2550b41ee338f4bd045ac1` |
| Size | 9,984,332 bytes |
| Delegate metadata | Manifest preference hints `android-gpu` with `default`/`cpu` fallbacks; public acceptance retains canonical CPU behavior |

The historical EXP02 mobile wrapper is documented in the training and SDK
repositories, but is not the current public SDK default:

| Field | Historical EXP02 wrapper contract |
|---|---|
| Input | RGB `uint8`, NHWC `[1,416,416,3]` |
| Preprocessing | Top-left aspect fit, pad 114; normalization inside the wrapper |
| Output | Float32 `[1,3549,7]` |
| Raw fields | `tx,ty,tw,th,angle_logit,objectness_sigmoid,class_sigmoid` |
| Decode | External grid/stride decode; `xy` and `exp(raw_wh)*stride`; angle in degrees |
| NMS | External rotated NMS |
| FP32 wrapper SHA-256 | `3af534950f8138cab89500657de0cf57d07ffb1304ea2ad08bc868f725216d80` |
| FP16-weight wrapper SHA-256 | `469ff586d990226e823c89c60ea7cc9ffbf43f18c1d6b4b08b75d1b88725d887` |
| Framework provenance | Historical source `5f8b2832c013ba03828e2f60f29cdb85471de7a6` |

The EXP02 wrapper’s model-side parity records do not establish SDK or device
qualification. The SDK repository has no EXP02 entry in its public mainline
manifest; [SDK Issue #363](https://github.com/shapovalovei/react-native-scanner-sdk/issues/363)
tracks the downstream qualification gate and [SDK Issue #366](https://github.com/shapovalovei/react-native-scanner-sdk/issues/366)
tracks the open inference-path audit. Adopting a newly trained artifact
requires a separate SDK qualification and manifest/package change.

### SDK code action checklist

For the framework changes in this handoff, the current evidence supports:

| SDK area | Framework-driven action now |
|---|---|
| Preprocessing, RGB order, normalization, resize/aspect, top-left padding | **No change required.** Verify against the selected artifact only if an artifact is adopted. |
| Raw decode and angle decode | **No change required.** The maintained raw contract is unchanged; artifact-specific adapters remain SDK work. |
| Rotated NMS | **No change required.** It remains external to the raw graph. |
| ROI/source mapping | **No change required from this framework phase.** It is SDK-owned and artifact/qualification-specific. |
| Model packaging and manifest | **No current change.** Change only after a concrete artifact is accepted. |
| Runtime delegates, threading, and lifecycle | **No framework-driven change.** Target qualification is separate. |

Therefore: **no framework-driven SDK code change is required until a new
artifact is selected and passes model-quality and runtime gates**. Existing
SDK work on OBB-aware session acceptance, source mapping, reader crop policy,
and reader long-axis normalization is downstream integration work and is not
an implication of these framework commits.

### Accelerator watch items

| Item | Current status | Consequence |
|---|---|---|
| Framework mobile-accelerator research [#15](https://github.com/shapovalovei/YOLOX-OBB/issues/15) | Generic risk inventory; target proof required | Not a framework defect or SDK requirement. |
| QNN target validation [#19](https://github.com/shapovalovei/YOLOX-OBB/issues/19) | Blocked by unavailable matching compiler/hardware evidence | No QNN compatibility or failure claim. |
| LiteRT GPU Android [SDK #215](https://github.com/shapovalovei/react-native-scanner-sdk/issues/215), [#237](https://github.com/shapovalovei/react-native-scanner-sdk/issues/237), [#302](https://github.com/shapovalovei/react-native-scanner-sdk/issues/302) | Target-specific GPU semantic divergence was investigated and rejection was retained | No public delegate promotion; no generic framework requirement. |
| Core ML iOS [SDK #216](https://github.com/shapovalovei/react-native-scanner-sdk/issues/216), [#221](https://github.com/shapovalovei/react-native-scanner-sdk/issues/221), [#236](https://github.com/shapovalovei/react-native-scanner-sdk/issues/236) | Delegate research/qualification is closed without public Core ML promotion | Deferred; current public behavior remains CPU and no framework action follows. |
| Heterogeneous CPU/GPU [SDK #381](https://github.com/shapovalovei/react-native-scanner-sdk/issues/381) | Open research | Requires real delegate/device evidence; no scheduler requirement is proven. |
| Near-vertical EXP02 behavior [SDK #382](https://github.com/shapovalovei/react-native-scanner-sdk/issues/382) | Open upstream model/training limitation handoff | Do not add an SDK workaround without an accepted model/runtime cause. |
| EXP02 versus YOLO26 benchmark [SDK #374](https://github.com/shapovalovei/react-native-scanner-sdk/issues/374) | Open model-comparison work; not a framework correctness result | Do not select or replace an artifact from this benchmark until it is complete. |
| EXP02 distance/scale behavior [SDK #380](https://github.com/shapovalovei/react-native-scanner-sdk/issues/380) | Open research; failure boundary is not established as a framework defect | Deferred; requires the SDK qualification path and controlled physical evidence. |
| EXP02 rotated-card OCR [SDK #377](https://github.com/shapovalovei/react-native-scanner-sdk/issues/377) | Open downstream reader/crop investigation after detector geometry work | No framework action; keep reader policy separate from detector framework changes. |

An operator-support concern is not a device failure. A missing Qualcomm target
or toolchain is a validation blocker, not evidence of a generic framework bug.

## Downstream action matrix

| Framework item | Training action | Retrain | Re-export | SDK code | Replace artifact | Validation | Owner |
|---|---|---|---|---|---|---|---|
| Visible-polygon labels #29/#31 | Pin final SHA and run final-semantics candidate | Required for final-semantics candidate | After retrain | No | Only after qualification | Model geometry and real-world gate | card-detector-training |
| Target validity #34/#35 | Pin final SHA; preserve valid-label reporting | Required for final-semantics candidate | After retrain | No | Only after qualification | Stability and target-validity evidence | card-detector-training |
| KLD stability #33/#36 | Pin final SHA and retrain candidate | Required for final-semantics candidate | After retrain | No | Only after qualification | Loss/gradient stability plus model quality | card-detector-training |
| Training decode #37/#38/#39 | Pin final SHA and retrain candidate | Required for final-semantics candidate | After retrain | No | Only after qualification | Finite gradients and model quality | card-detector-training |
| Earlier assignment/angle fixes | Included by final pin | Recommended in same run | After retrain if artifact adopted | No | Only after qualification | Assignment and geometry analysis | card-detector-training |
| CUDA assignment/fallback #26/#40/#41/#44 | Use final pin; preserve current fallback boundary | No separate run | No | No | No | T4 validation complete; no optimization or fallback-correctness phase justified | YOLOX-OBB |
| Optional MixUp #30/#42 | No action while disabled; enable only by experiment | Only if enabled | After a qualifying retrain | No | Only after qualification | MixUp-enabled label/quality gate | card-detector-training |
| Dynamic raw ONNX #5/#12 | No retrain | No | Only when choosing a dynamic ONNX artifact | No | After concrete export review | Shape/layout/runtime parity | card-detector-training |
| DOTA evaluator #13/#16 | No retrain | No | No | No | No | Use real external evaluation | YOLOX-OBB / card-detector-training |
| Rotated-IoU packaging #14/#17 | No retrain | No | No | No | No | Install/package checks | YOLOX-OBB |
| Perspective #32 | No current action | No | No | No | No | Deferred contract decision | YOLOX-OBB |
| Assignment memory/equivalence #27/#44 | No optimization action; preserve current implementation | No | No | No | No | 28/28 primary and 8/8 selected secondary FP32 cells; stable GPU/CPU equivalence | YOLOX-OBB |

## Provenance model

Every future promoted artifact should preserve this chain:

```text
YOLOX-OBB behavioral SHA
  -> training repository commit and run ID
  -> dataset manifest/version
  -> checkpoint SHA-256
  -> export configuration
  -> mobile artifact SHA-256
  -> SDK commit/release
  -> device/runtime qualification evidence
```

Ownership is split as follows:

| Provenance field | Owner | Current handoff value |
|---|---|---|
| Behavioral framework SHA | YOLOX-OBB | `0695e2834ff5716dfd51fcd6ac7b2300b7bc27ce` |
| Training commit/run ID | card-detector-training | Required for a future final-framework run; historical EXP02 records use their documented source/run values |
| Dataset manifest/version | card-detector-training | Required for every future promoted run |
| Checkpoint SHA-256 | card-detector-training | Historical champion: `a668c839dfb1ad58019032f57b91954b3fc54e72853c4b632770479b1d56b201` |
| Export configuration | card-detector-training | Required for every future concrete export |
| Mobile artifact SHA-256 | card-detector-training | Historical EXP02 wrapper and current YOLO26 hashes are recorded above; no final-framework artifact exists yet |
| SDK commit/release | react-native-scanner-sdk | Required at artifact adoption; current public mainline evidence is `dc00d57f66f14741f1639ceb6c89bf4833b4f2b9` |
| Device/runtime evidence | react-native-scanner-sdk | Required for promotion; not established for a final-framework artifact |

Missing historical values remain missing; they must not be inferred from a
filename, chat transcript, or an unverified local artifact.

## Backward compatibility

| Question | Classification |
|---|---|
| Source compatibility | **Maintained for the covered fixes.** No public OBB field-order or angle-unit change was introduced; dynamic export is opt-in. |
| Model artifact compatibility | **The raw contract remains usable for historical artifact evaluation.** Current public SDK mainline does not package EXP02, so execution of an EXP02 artifact in the public mainline is not established; artifact adoption still requires the consumer’s own parity checks. |
| Training semantic compatibility | **Not equivalent.** A checkpoint trained before #29, #34, #33, and #38 does not become a final-framework checkpoint retroactively. |

## Known limitations and deferred work

| Limitation | Evidence/status | Owner | Next trigger |
|---|---|---|---|
| True projective perspective semantics | [#32](https://github.com/shapovalovei/YOLOX-OBB/issues/32) proves non-zero `perspective` does not add projective matrix terms; maintained recipes inspected use `0.0`; not fixed | YOLOX-OBB | Explicit augmentation-contract decision and implementation Issue |
| True CUDA autocast FP16 support | [#44](https://github.com/shapovalovei/YOLOX-OBB/issues/44) tested the maintained path on T4/PyTorch 2.11; BCE triggers the autocast safety guard | YOLOX-OBB | Separate authorized compatibility research only if AMP becomes a requirement |
| QNN hardware validation | [#15](https://github.com/shapovalovei/YOLOX-OBB/issues/15), [#19](https://github.com/shapovalovei/YOLOX-OBB/issues/19) identify risk but lack target compiler/session evidence | YOLOX-OBB / target owner | Real target compiler, partition/fallback trace, and parity run |
| Android GPU target validation | SDK [#215](https://github.com/shapovalovei/react-native-scanner-sdk/issues/215), [#237](https://github.com/shapovalovei/react-native-scanner-sdk/issues/237), and [#302](https://github.com/shapovalovei/react-native-scanner-sdk/issues/302) retain target-specific GPU rejection; no public promotion | react-native-scanner-sdk | New target validation only if the supported-device requirement changes |
| Core ML target validation | SDK [#216](https://github.com/shapovalovei/react-native-scanner-sdk/issues/216), [#221](https://github.com/shapovalovei/react-native-scanner-sdk/issues/221), and [#236](https://github.com/shapovalovei/react-native-scanner-sdk/issues/236) close the current research/qualification path without public Core ML promotion | react-native-scanner-sdk | Deferred public support; no framework action |
| Training quality under final framework | No final-framework retraining result is recorded | card-detector-training | Fresh controlled run pinned to `0695e283...` |
| Concrete final mobile artifact | Historical EXP02 artifacts predate the final framework; no replacement is selected | card-detector-training | Accepted checkpoint plus reproducible export |
| SDK device qualification | EXP02 integration/qualification remains downstream work; current public mainline uses YOLO26 | react-native-scanner-sdk | Accepted artifact and SDK qualification Issue |
| Heterogeneous CPU/GPU scheduling | SDK [#381](https://github.com/shapovalovei/react-native-scanner-sdk/issues/381) is open research | react-native-scanner-sdk | Real delegate evidence and performance requirement |

## Rollout / phase status

| Phase | Status |
|---|---|
| Framework behavioral stabilization | Complete |
| CUDA assignment validation | Complete / Verdict A |
| Framework documentation handoff | This PR |
| Downstream training | Future, separately authorized; not started here |
| Downstream export/SDK | Only after a concrete artifact decision and qualification |

## Recommended rollout sequence

1. Pin `0695e2834ff5716dfd51fcd6ac7b2300b7bc27ce` in the training repository’s
   new experiment provenance.
2. Run a fresh controlled training experiment under the same intended recipe,
   with MixUp disabled and perspective at `0.0` unless separately authorized.
3. Select a checkpoint using the training repository’s normal validation
   procedure.
4. Compare it with the historical champion using the durable real-world
   validation contract and geometry/localization analysis.
5. Only after acceptance, export concrete FP32/FP16 mobile candidates in the
   training repository.
6. Verify model-side export/runtime parity and record artifact hashes.
7. Update the SDK artifact and manifest only after the candidate is accepted;
   do not change preprocessing or decode without parity evidence.
8. Run SDK and device qualification, including any target delegate work as a
   separate validation gate.
9. Promote the artifact only after model-quality and runtime gates pass.
