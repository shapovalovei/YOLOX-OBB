# DOTA helpers

This directory is based on [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) and retains the original helper attribution and reference material. In this maintained fork it is packaged from the repository root together with the YOLOX native extension.

## Installation

Install from the repository root; do not install a nested requirements file or rely on a compiled artifact being present in Git:

```bash
python -m pip install -r requirements.txt
python -m pip install -v -e . --no-build-isolation
```

This builds `DOTA_devkit_YOLO._polyiou`. Verify it with the [native smoke check](../docs/testing.md#native-smoke-check) before rotated evaluation or polygon NMS. The nested `setup.py` is retained as historical context, not the normal maintained installation path.

## Helper inventory

- `DOTA.py`: load and display DOTA annotations.
- `ImgSplit.py` and `ImgSplit_multi_process.py`: split images and labels.
- `ResultMerge.py`: merge detection-result text files; the current helper contains local path assumptions.
- `dota-v1.5_evaluation_task1.py`: DOTA 1.5 evaluation helper; configure paths before use.
- `YOLO_Transform.py`: convert DOTA annotations to YOLO HBB or long-side/OBB formats.
- `Draw_DOTA_YOLO.py`: visualize YOLO OBB labels after augmentation.

These scripts are source references and several contain hardcoded paths or historical assumptions. They do not form a stable portable CLI. For the maintained dataset layout and evaluation boundary, see [custom OBB data](../docs/train_custom_data.md).

## Usage examples

The examples below are historical helper invocations. They require local data and path configuration and were not treated as repository-wide validation commands:

```bash
python DOTA.py
python ImgSplit_multi_process.py
python ResultMerge.py
python dota-v1.5_evaluation_task1.py
python YOLO_Transform.py
python Draw_DOTA_YOLO.py
```

For the evaluation helper, configure values equivalent to:

```python
detpath = r"/.../evaluation_example/result_classname/Task1_{:s}.txt"
annopath = r"/.../evaluation_example/row_DOTA_labels/{:s}.txt"
imagesetfile = r"/.../evaluation_example/imgnamefile.txt"
```

The DOTA polygon format is `poly classname difficult`. The long-side format emitted by the transformer is `classid x_c y_c longside shortside angle`; check the maintained OBB guide for the framework's public degree convention before connecting another tool.

## Original references

The following references are retained from the source helper project:

- [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)
- [DOTA遥感数据集以及相关工具DOTA_devkit的整理](https://zhuanlan.zhihu.com/p/355862906)
- [DOTA数据格式转YOLO数据格式工具](https://zhuanlan.zhihu.com/p/356416158)

The original project also credited the author as:

```text
Name: 胡凯旋
```
