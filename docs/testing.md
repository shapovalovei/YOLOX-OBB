# Local testing

The maintained test suite is local correctness evidence. It is separate from the current GitHub workflow, which declares style checks only, does not install the root package or run the test suite/native smoke, and references a `format_check.sh` file absent from this checkout.

## Run the maintained suite

From the repository root, with the runtime dependencies installed:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_*.py' -v
```

The suite is intentionally run through Python's unittest discovery so the repository's test modules and environment are used. A clean environment without PyTorch, NumPy, OpenCV, or other test dependencies will fail during collection; record that as an environment blocker rather than treating it as a passing or complete test run.

## What the tests establish

Covered tests provide regression evidence for selected framework contracts, including OBB geometry and angle handling, augmentation and Mosaic behavior, rectangular dimensions, assignment/loss/decode behavior, native-independent utility paths, and compatibility/export checks where their dependencies are available.

Passing local tests do not establish:

- model-quality improvement or a checkpoint recommendation;
- dataset completeness or external DOTA/COCO AP;
- GPU/CUDA compatibility across environments;
- throughput, memory limits, or long-running training stability;
- native extension availability unless the native smoke check also passes;
- ONNX/TensorRT/backend parity or mobile/device qualification.

## Native smoke check

Before a dependent run, verify both extensions from the repository root:

```bash
python - <<'PY'
from DOTA_devkit_YOLO import polyiou
import yolox._C

p = polyiou.VectorDouble([0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0])
assert abs(polyiou.iou_poly(p, p) - 1.0) < 1e-12
print("native extensions: OK")
PY
```

This catches a source-only checkout or an incomplete native build before training/evaluation. It is not a model-quality test.

## Documentation checks

When Sphinx dependencies are available:

```bash
make -C docs html
```

For documentation-only changes, also inspect the full diff, run `git diff --check`, verify relative Markdown links, and check that fenced code blocks are balanced. Commands shown in docs should be checked against the current CLI/source; if execution would require data, GPU/CUDA, a model, or a missing dependency, leave the command unexecuted and say why.
