# Local testing

The maintained test suite is local correctness evidence. It is separate from the current GitHub workflow, which declares style checks only, does not install the root package or run the test suite/native smoke, and references a `format_check.sh` file absent from this checkout. The tracked `requirements.txt` is a broad inherited dependency list, not a validated modern compatibility matrix; its exact environment compatibility remains unverified.

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

Run this from the repository root after installation. The shell captures the intended checkout, but launches Python from a fresh temporary directory outside the checkout so the current working directory cannot shadow an installed package:

```bash
repo_root="$(pwd -P)"
smoke_dir="$(mktemp -d "${TMPDIR:-/tmp}/yolox-obb-smoke.XXXXXX")"
trap 'rm -rf "$smoke_dir"' EXIT
(
  cd "$smoke_dir"
  YOLOX_REPO_ROOT="$repo_root" env -u PYTHONPATH python - <<'PY'
import os
import sysconfig
from pathlib import Path

import yolox
import yolox._C
from DOTA_devkit_YOLO import polyiou
import DOTA_devkit_YOLO._polyiou as polyiou_native

print("yolox.__file__:", yolox.__file__)
print("yolox._C.__file__:", yolox._C.__file__)
print("polyiou.__file__:", polyiou.__file__)
print("polyiou_native.__file__:", polyiou_native.__file__)

repo_root = Path(os.environ["YOLOX_REPO_ROOT"]).resolve()
source_roots = (repo_root / "yolox", repo_root / "DOTA_devkit_YOLO")
install_roots = []
for key in ("purelib", "platlib"):
    value = sysconfig.get_paths().get(key)
    if value:
        install_roots.append(Path(value).resolve())
install_roots = tuple(install_roots)

def under(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False

for module, source_root in (
    (yolox, source_roots[0]),
    (yolox._C, source_roots[0]),
    (polyiou, source_roots[1]),
    (polyiou_native, source_roots[1]),
):
    imported = Path(module.__file__).resolve()
    if not (
        under(imported, source_root)
        or any(under(imported, install_root) for install_root in install_roots)
    ):
        raise RuntimeError(f"unexpected import origin: {imported}")

p = polyiou.VectorDouble([0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0])
assert float(polyiou.iou_poly(p, p)) == 1.0
print("native extensions: OK")
PY
)
```

Interpret the printed paths rather than requiring one install style: an editable/source build may legitimately resolve into the exact intended checkout, while a non-editable install should resolve into the intended environment's install roots. Any path from an unrelated checkout or package location is a failure. This catches source-tree shadowing or an incomplete native build before training/evaluation; it is not a model-quality test.

## Documentation checks

When Sphinx dependencies are available:

```bash
make -C docs html
```

For documentation-only changes, also inspect the full diff, run `git diff --check`, verify relative Markdown links, and check that fenced code blocks are balanced. Commands shown in docs should be checked against the current CLI/source; if execution would require data, GPU/CUDA, a model, or a missing dependency, leave the command unexecuted and say why.
