# Maintainer guide

This guide describes the maintained YOLOX-OBB fork at the current checkout. It is a framework guide, not a training recipe for a particular production model and not a mobile-runtime qualification report.

## Repository map

| Area | Purpose |
| --- | --- |
| `yolox/` | model, data, augmentation, assignment, losses, decode, evaluation, and training infrastructure |
| `DOTA_devkit_YOLO/` | DOTA/VOC-style OBB helpers and the native polygon-IoU extension |
| `exps/` | experiment templates; concrete runs and selected artifacts belong in the training project |
| `tools/` | training, evaluation, and export entry points |
| `tests/` | maintained local unit/regression tests |
| `demo/` | backend examples and reference integrations; these pages are not a blanket OBB qualification matrix |
| `docs/` | maintained guidance, reference pages, and historical project records |

## Installation and native extensions

Create an isolated environment with an importable PyTorch, a C/C++ compiler, Python development headers, and compatible setuptools. The root `setup.py` builds two extensions from checked-in sources: `yolox._C` and `DOTA_devkit_YOLO._polyiou`. `MANIFEST.in` includes the sources, so a clean Git checkout is expected to be source-only; compiled `.so` files are build outputs. The tracked `requirements.txt` is a broad inherited dependency list, not a validated modern compatibility matrix; it includes old export pins such as `onnx==1.8.1`, `onnxruntime==1.8.0`, and `onnx-simplifier==0.3.5`.

If you choose to install that broad list, run it before the root package build:

```bash
python -m pip install -r requirements.txt
python -m pip install -v -e . --no-build-isolation
```

The package setup evaluates PyTorch while defining the native extensions. Make PyTorch importable before invoking the root install. Core/native setup is separate from optional/backend/export dependencies, and the requirements file does not establish a supported environment matrix. SWIG is relevant to regenerating the checked-in `DOTA_devkit_YOLO/polyiou_wrap.cxx`, not to the normal build of that checked-in wrapper.

On some current Python/pip/setuptools combinations, editable installation can recurse through the legacy `setup.py develop` frontend before the extension build completes. The source-preserving fallback is:

```bash
python -m pip install -v . --no-build-isolation --no-deps
```

This is a compatibility fallback, not a promise that every modern environment is supported. The exact OS/Python/PyTorch/pip matrix remains unverified. A prior native-build validation covered macOS x86_64, Python 3.11.11, and PyTorch 2.2.2; it should not be generalized to other matrices.

Run the [authoritative native smoke check](testing.md#native-smoke-check) immediately after either installation path. It launches Python outside the checkout, prints all imported module origins, accepts the exact intended source checkout for an editable build or the intended environment install roots for a non-editable build, and rejects unrelated origins.

The current training/evaluation import path also requires `Apex`; it is not installed by `requirements.txt`, so provision it separately when using those tools. `pycocotools` is needed for COCO-related evaluation paths; it does not provide the rotated OBB extension.

The nested `DOTA_devkit_YOLO/setup.py` is a historical standalone build path. It is not the normal installation command for this repository; use the root package build so both extensions are handled together.

## OBB representation and validity

The dataset-side source annotation convention is:

```text
[xmin, ymin, xmax, ymax, angle_degrees, class_id]
```

The XML field names are historical VOC storage names. The converter fits `cv2.minAreaRect`, canonicalizes the long and short sides, and stores `xmin = center_x - long_width / 2`, `xmax = center_x + long_width / 2`, `ymin = center_y - short_height / 2`, and `ymax = center_y + short_height / 2`, together with the angle. These four values encode OBB center plus canonical long/short dimensions; they are not the HBB envelope of the rotated polygon and are not polygon corners. After the training transform, labels are represented as:

```text
[class_id, center_x, center_y, width, height, angle_degrees]
```

The public angle is in degrees and is normalized to `[-90, 90)`. Width is the canonical long side and height the short side. Internal trigonometric calculations use radians where the implementation requires them. Class `0` and angle `0` are valid values; zero-filled padded rows are not labels.

Training targets must have finite values, positive dimensions, and a visible post-transform minimum dimension greater than four pixels. OBB augmentation reconstructs rectangle corners, applies the image transform, clips the visible polygon, and fits a canonical minimum-area rectangle. Horizontal/vertical flips preserve the rectangle while updating the angle.

Mosaic combines four images when enabled and MixUp remains separately configurable. Do not infer a universal augmentation policy from one experiment: for example, the concrete DOTA experiment disables MixUp even though the base OBB experiment exposes it.

## Training lifecycle

The no-Mosaic boundary is logical-batch based and is assigned before prefetching. Let:

```text
M = len(train_loader)
B = max_epoch - no_aug_epochs
cutover = B * M
mosaic_for_ordinal = ordinal < cutover
```

For a configuration of `max_epoch=100` and `no_aug_epochs=15`, human epochs 1–85 use Mosaic and epochs 86–100 use no-Mosaic with the final L1 phase. Those numbers are derived from the configuration and loader length; they are not a universal fixed schedule.

The final Mosaic epoch and the no-Aug/L1 phase are aligned with this boundary. Prefetching may hold work emitted before the boundary, so the sampler emission ordinal—not the consumer's later observation—is the contract.

Generic `Trainer.save_ckpt()` persists `start_epoch`, model state, optimizer state, best-metric state, and AMP state when applicable. It does not persist a complete Mosaic/DataLoader execution state such as the logical batch ordinal, `M`, cutover ordinal, `max_epoch`, `no_aug_epochs`, sampler cursor, queued batches, worker state, or augmentation RNG. On process resume, `Trainer` reconstructs the Mosaic schedule from the current `start_epoch`, current experiment values, and current `len(train_loader)` before `DataPrefetcher` construction. Exact generic mid-epoch process resume is not guaranteed.

Rotated assignment uses OBB candidate geometry and dynamic-k matching. KLD prediction/target argument order is part of the API: `KLDloss.forward(pred, target)` is prediction-first, while the historical helper `compute_kld_loss(targets, preds)` is target-first. Keep degree/radian conversion and floating-point ordering intact when changing this code.

Training decode uses `exp(raw_wh + log(stride))`; eager inference uses `exp(raw_wh) * stride`. This algebraic distinction is intentional for gradient behavior. Exported models expose raw output for external decode/postprocess, so an integration must preserve the documented output layout and OBB angle conversion.

The trainer's CPU fallback is a CUDA-out-of-memory recovery path. It is not evidence that a CPU-only training invocation is supported. The current `perspective` configuration is not a qualification of projective augmentation behavior; that boundary remains unverified for a generic documentation claim.

## Evaluation and postprocess

The DOTA evaluator writes per-class polygon result files and returns timing/loader information rather than a complete internal DOTA AP score. Empty predictions are represented by empty per-class result files, allowing an evaluation run to complete without inventing detections.

Rotated polygon NMS is performed in the Python/native postprocess path using `DOTA_devkit_YOLO._polyiou`; it is not required to be embedded in an exported neural-network graph. The native extension is therefore a runtime capability for rotated evaluation and postprocess, not just an installation convenience.

`tools/eval.py` follows the CUDA-oriented experiment path. The direct evaluator supports CPU float32 operation, while CPU half precision is rejected. The exposed `--test` option should not be read as a portable unlabeled-DOTA workflow: the maintained DOTA experiment's evaluation loader is validation-oriented and obtains annotations from its configured XML layout.

The DOTA conversion and result-merging helpers contain machine-specific paths or lack stable command-line configuration. Treat them as source references requiring local configuration, not as reproducible one-command tooling. Record the dataset, experiment, checkpoint, and external evaluator provenance in the separate training project.

## Export and backend references

`tools/export_onnx.py` exports model outputs for external decode/postprocess. The normal static path uses a square input divisible by 32; `--dynamic-shape` changes the exported input-shape contract and should be tested by the consuming runtime. OBB decode, polygon construction, and rotated NMS remain outside the neural-network graph unless a specific integration establishes otherwise.

The maintained OBB ONNX notes are in [the ONNX reference](demo/onnx_readme.md). Other backend pages under `demo/` and `docs/demo/` are inherited examples or backend-specific references. Their presence does not prove OBB support, parity, performance, or device qualification. Use the current source, tests, and a target-specific validation task for those claims.

## Local validation

The maintained local suite is run with:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_*.py' -v
```

The tests are correctness evidence for covered geometry, augmentation, loader, assignment, loss, decode, export, and compatibility contracts. They do not prove trained-model quality, dataset coverage, throughput, memory limits, or physical/mobile behavior. Do not substitute GitHub CI for the local suite: the current workflow declares style checks, does not install the root package or execute the full unittest suite/native smoke, and references a `format_check.sh` file absent from this checkout.

The Sphinx tree can be built when its documentation dependencies are available:

```bash
make -C docs html
```

During this audit, dependency-gated runtime and documentation commands were not made to install packages, run training, use GPU/CUDA, download data, or evaluate a model. Keep such checks explicitly marked as unverified when the environment cannot provide their dependencies.

## Project boundary and evidence

Use YOLOX-OBB for generic framework correctness. Use [`card-detector-training`](https://github.com/shapovalovei/card-detector-training) for training runs, datasets, checkpoints, model-quality evidence, concrete export/quantization, and artifact provenance. Use [`react-native-scanner-sdk`](https://github.com/shapovalovei/react-native-scanner-sdk) for mobile packaging, delegates, device preprocessing/decode/NMS, camera/ROI behavior, and release qualification.

Issue and PR history can explain why a contract exists, but it is not a substitute for current source and tests. In particular, [the historical handoff](YOLOX_OBB_FINAL_FRAMEWORK_HANDOFF.md) records an older snapshot and must not be read as the current baseline.
