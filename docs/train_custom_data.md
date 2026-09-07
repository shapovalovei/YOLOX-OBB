# Train on custom OBB data

This repository maintains the generic OBB framework. A concrete dataset, run, checkpoint, and model-quality result belong in the separate training project and are not included by this guide.

## Dataset layout

The maintained DOTA/VOC-style experiment expects a layout equivalent to:

```text
VOC2012/
├── Annotations/
├── ImageSets/Main/
│   ├── train.txt
│   └── val.txt
├── JPEGImages/
├── JPEGImages-val/
└── JPEGImages-test/
```

The annotation convention is OBB-specific. Source rows use `[xmin, ymin, xmax, ymax, angle_degrees, class_id]`; the training representation is `[class_id, center_x, center_y, width, height, angle_degrees]`. See [the maintainer guide](maintainer_guide.md) for validity and angle semantics.

## Convert or prepare data

`custom tools/DOTA2VOC_obb.py` is a repository helper, but its current main path contains machine-specific absolute directories and does not expose a stable portable CLI. Configure those paths locally before using it. Because the path contains a space, quote it when invoking the script:

```bash
python "custom tools/DOTA2VOC_obb.py"
```

The command is a source-verified invocation form and was not run during the documentation audit; it must not be read as proof that a dataset is present or conversion is complete.

## Configure an experiment

Copy an existing experiment into a local file and edit its data directory, split files, and class list. Do not overwrite a maintained example:

```bash
cp exps/example/yolox_voc/yolox_dota_s_obb_kld.py exps/my_dota_obb.py
```

The experiment controls augmentation, the no-Mosaic/L1 boundary, input dimensions, and evaluation behavior. The boundary is derived from `max_epoch`, `no_aug_epochs`, and the logical loader batch count; it is not a universal fixed epoch number.

## Train and evaluate

The current launch path is CUDA-oriented and imports Apex:

```bash
python tools/train.py -n yolox-s -f exps/my_dota_obb.py -d 1 -b 8 --fp16 -o
python tools/eval.py -n yolox-s -f exps/my_dota_obb.py -c /path/to/ckpt.pth.tar -b 8 --fp16
```

The example commands require a configured dataset, a compatible environment, and a checkpoint for evaluation. They were not executed during the documentation audit because that would require unavailable runtime dependencies and would cross the no-training/no-model-evaluation boundary.

## DOTA evaluation boundary

The evaluator writes per-class polygon result files for external DOTA evaluation and handles empty predictions as empty files. It does not calculate a complete DOTA AP score internally. The exposed `--test` option is not a portable unlabeled-DOTA workflow for the maintained experiment: its configured evaluation loader is validation-oriented and reads annotations from its XML layout.

Conversion, result merging, and DOTA evaluation helpers include hardcoded paths or historical assumptions. Record the exact dataset, experiment, checkpoint, external evaluator, and artifact hashes in `card-detector-training`; do not turn a helper script into a generic framework guarantee.

For export, mobile integration, or concrete checkpoint quality, use the owning downstream project. For framework contracts and local correctness tests, use the [maintainer guide](maintainer_guide.md).
