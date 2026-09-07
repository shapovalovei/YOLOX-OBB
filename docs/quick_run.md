# Quick run

This is the short path for the maintained YOLOX-OBB fork. For contracts and boundaries, read the [maintainer guide](maintainer_guide.md) first.

## Install and verify

Use an isolated environment with importable PyTorch, a compiler, Python development headers, and compatible setuptools:

```bash
python -m pip install -r requirements.txt
python -m pip install -v -e . --no-build-isolation
```

The root build includes both `yolox._C` and `DOTA_devkit_YOLO._polyiou`. Verify them before a long run:

```bash
python - <<'PY'
from DOTA_devkit_YOLO import polyiou
import yolox._C

p = polyiou.VectorDouble([0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0])
assert abs(polyiou.iou_poly(p, p) - 1.0) < 1e-12
print("native extensions: OK")
PY
```

If the editable frontend fails in a current pip/setuptools environment, try the source-preserving fallback:

```bash
python -m pip install -v . --no-build-isolation --no-deps
```

The fallback is not a universal compatibility guarantee. The current training/evaluation import path also requires Apex.

## Run an experiment

Copy an OBB experiment and edit its dataset paths and classes before running. The following commands are CUDA-oriented and are source-verified examples, not a claim that a checkpoint or dataset is included:

```bash
cp exps/example/yolox_voc/yolox_dota_s_obb_kld.py exps/my_dota_obb.py
python tools/train.py -n yolox-s -f exps/my_dota_obb.py -d 1 -b 8 --fp16 -o
python tools/eval.py -n yolox-s -f exps/my_dota_obb.py -c /path/to/ckpt.pth.tar -b 8 --fp16
```

The DOTA evaluator writes polygon result files for external evaluation; it does not produce a complete internal DOTA AP score. See [custom OBB data](train_custom_data.md) and the [maintainer guide](maintainer_guide.md) for the dataset layout, phase schedule, and evaluator limits.

## Run local tests

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_*.py' -v
```

These tests are maintained correctness evidence, not model-quality or mobile qualification evidence. See [local testing](testing.md).
