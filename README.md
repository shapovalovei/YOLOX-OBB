# YOLOX-OBB

YOLOX-OBB is a maintained fork of [YOLOX-OBB by buzhidaoshenme](https://github.com/buzhidaoshenme/YOLOX-OBB), with oriented bounding-box (OBB) support kept as a core framework contract. The original project and its license and attribution remain part of this repository's provenance; this is not the official Megvii YOLOX project.

This repository maintains generic OBB model, data, training, evaluation, and export behavior. It is not a model registry or a mobile SDK.

## Start here

- [Maintainer guide](docs/maintainer_guide.md): installation, native extensions, OBB contracts, training lifecycle, evaluation, export, and validation boundaries.
- [Quick run](docs/quick_run.md): the shortest maintained-fork setup and command path.
- [Custom OBB data](docs/train_custom_data.md): the current DOTA/VOC-style data layout and experiment workflow.
- [Local validation](docs/testing.md): maintained correctness evidence and its limits.
- [Sphinx documentation](docs/index.rst): the navigable documentation tree.

## Ownership boundaries

YOLOX-OBB owns generic framework and model correctness: OBB geometry, assignment, augmentation, model heads, decode, postprocess, rotated NMS, export behavior, and regression tests.

The related projects own different artifacts and integration layers:

| Project | Owns |
| --- | --- |
| [`card-detector-training`](https://github.com/shapovalovei/card-detector-training) | datasets, recipes, training runs, checkpoints, model-quality evidence, concrete exports/quantization, and artifact provenance |
| [`react-native-scanner-sdk`](https://github.com/shapovalovei/react-native-scanner-sdk) | mobile packaging and runtime integration, delegates, device preprocessing/decode/NMS, camera/ROI behavior, and release qualification |

Do not interpret downstream model or device results as generic framework guarantees.

## Installation and native capabilities

Use a clean virtual environment with a supported compiler toolchain and an importable PyTorch installation. The tracked `requirements.txt` is a broad inherited dependency list, not a validated modern compatibility matrix. It contains old export pins such as `onnx==1.8.1`, `onnxruntime==1.8.0`, and `onnx-simplifier==0.3.5`; exact environment compatibility is not guaranteed by that file. If you choose to use it, run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -v -e . --no-build-isolation
```

The root build compiles both native extensions used by maintained paths:

- `yolox._C` for the YOLOX native operators;
- `DOTA_devkit_YOLO._polyiou` for rotated polygon IoU/NMS support.

A clean Git checkout contains the native source files, not compiled `.so` artifacts. Build the extensions before training or evaluation and fail fast with the [authoritative native smoke check](docs/testing.md#native-smoke-check) rather than discovering the missing capability after a long run. SWIG is only needed when regenerating the checked-in `DOTA_devkit_YOLO/polyiou_wrap.cxx`; it is not the normal first-run build requirement. Core native setup requires importable PyTorch, a compiler, Python development headers, and compatible setuptools independently of optional/backend/export dependencies.

Some modern Python/pip/setuptools combinations do not handle this repository's editable-install frontend reliably. The source-preserving fallback is:

```bash
python -m pip install -v . --no-build-isolation --no-deps
```

That fallback has not been qualified for every OS, Python, PyTorch, and pip combination. Keep the exact interpreter and package versions in the environment record. Run the same [authoritative native smoke check](docs/testing.md#native-smoke-check) after either installation path; it launches Python outside the checkout, reports imported origins, and verifies the source provenance of any non-editable installation.

`Apex` is imported by the current training/evaluation launch path and must be available for those tools; it is not installed by `requirements.txt`. `pycocotools` is an additional COCO-evaluation dependency, not a replacement for the OBB native extension.

## Minimal OBB workflow

The maintained examples use a DOTA/VOC-style OBB dataset and an experiment file. Copy an existing experiment into a local file, edit its dataset paths/classes, then run:

```bash
cp exps/example/yolox_voc/yolox_dota_s_obb_kld.py exps/my_dota_obb.py
python tools/train.py -n yolox-s -f exps/my_dota_obb.py -d 1 -b 8 --fp16 -o
python tools/eval.py -n yolox-s -f exps/my_dota_obb.py -c /path/to/ckpt.pth.tar -b 8 --fp16
```

These commands are CUDA-oriented and require the experiment's data and model configuration. The DOTA evaluator writes per-class polygon result files for external evaluation; it does not itself report a complete DOTA AP score. See the [maintainer guide](docs/maintainer_guide.md) for CPU limits, empty predictions, export boundaries, and unsupported helper-script assumptions.

## Validation evidence

Run the maintained local unit/regression suite with:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_*.py' -v
```

The suite is correctness evidence for covered framework contracts. It does not prove that a trained checkpoint is better, that a particular export is faster, or that a mobile/device runtime is qualified. The current GitHub workflow declares style checks only, does not install the root package or run the test suite/native smoke, and references a `format_check.sh` file absent from this checkout. It is not authoritative full-suite execution. See [local testing](docs/testing.md).

## Provenance and historical material

The [DOTA helper README](DOTA_devkit_YOLO/README.md) and the files under `docs/` preserve useful original-project context, but inherited commands and benchmark tables are not current guarantees unless the maintained guide says so. The [framework handoff ledger](docs/YOLOX_OBB_FINAL_FRAMEWORK_HANDOFF.md) is explicitly a historical snapshot; current behavior is defined by the checked-out source and maintained tests.

## License

This project retains the original project's licensing and attribution. See [LICENSE](LICENSE) and the attribution in the historical documentation for details.
