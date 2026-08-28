## YOLOX-ONNXRuntime in Python

This doc introduces how to convert your pytorch model into onnx, and how to run an onnxruntime demo to verify your convertion.

### Download ONNX models.

| Model | Parameters | GFLOPs | Test Size | mAP | Weights |
|:------| :----: | :----: | :---: | :---: | :---: |
|  YOLOX-Nano |  0.91M  | 1.08 | 416x416 | 25.3 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EfAGwvevU-lNhW5OqFAyHbwBJdI_7EaKu5yU04fgF5BU7w?e=gvq4hf)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_nano.onnx) |
|  YOLOX-Tiny | 5.06M     | 6.45 | 416x416 |32.8 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/ET64VPoEV8FAm5YBiEj5JXwBVn_KYHM38iJQ_lpcK2slYw?e=uuJ7Ii)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_tiny_32dot8.onnx) |
|  YOLOX-S | 9.0M | 26.8 | 640x640 |39.6 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/Ec0L1d1x2UtIpbfiahgxhtgBZVjb1NCXbotO8SCOdMqpQQ?e=siyIsK)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s.onnx) |
|  YOLOX-M | 25.3M | 73.8 | 640x640 |46.4 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/ERUKlQe-nlxBoTKPy1ynbxsBmAZ_h-VBEV-nnfPdzUIkZQ?e=hyQQtl)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_m.onnx) |
|  YOLOX-L | 54.2M | 155.6 | 640x640 |50.0 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/ET5w926jCA5GlVfg9ixB4KEBiW0HYl7SzaHNRaRG9dYO_A?e=ISmCYX)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_l.onnx) |
|  YOLOX-Darknet53| 63.72M | 185.3 | 640x640 |47.3 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/ESArloSW-MlPlLuemLh9zKkBdovgweKbfu4zkvzKAp7pPQ?e=f81Ikw)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_darknet53.onnx) |
|  YOLOX-X | 99.1M | 281.9 | 640x640 |51.2 | [onedrive](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/ERjqoeMJlFdGuM3tQfXQmhABmGHlIHydWCwhlugeWLE9AA)/[github](https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox.onnx) |


### Convert Your Model to ONNX

First, you should move to <YOLOX_HOME> by:
```shell
cd <YOLOX_HOME>
```
Then, you can:

1. Convert a standard YOLOX model by -n:
```shell
python3 tools/export_onnx.py --output-name yolox_s.onnx -n yolox-s -c yolox_s.pth
```
Notes:
* -n: specify a model name. The model name must be one of the [yolox-s,m,l,x and yolox-nane, yolox-tiny, yolov3]
* -c: the model you have trained
* -o: opset version, default 11. **However, if you will further convert your onnx model to [OpenVINO](https://github.com/Megvii-BaseDetection/YOLOX/demo/OpenVINO/), please specify the opset version to 10.**
* --no-onnxsim: disable onnxsim
* To customize a static input shape for an onnx model, modify the following code in `tools/export_onnx.py`:

    ```python
    dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])
    ```

### Dynamic OBB export (opt-in)

The existing static export remains the default. For an OBB experiment, pass
`--dynamic-shape` to export a model with this input contract:

```shell
python3 tools/export_onnx.py --dynamic-shape \
    --output-name yolox_obb_dynamic.onnx \
    -f exps/your_dir/your_obb_exp.py -c your_obb.pth
```

```text
[batch, 3, size, size]
```

The first dynamic-shape slice supports square `size` values divisible by 32.
The maintained regression contract covers `320`, `416`, and `512`, and also
batch size 2 at `416`.

The exported OBB output remains before OBB decode and rotated postprocess:

```text
[batch, predictions, 6 + num_classes]
[tx, ty, tw, th, angle_logit, objectness_probability, class_probabilities...]
```

Objectness and class fields are already probabilities. Decode the exported
output using the runtime square input size before calling
`postprocessobb_kld`:

```python
from yolox.utils.demo_utils import decode_obb_raw_outputs

decoded = decode_obb_raw_outputs(
    onnx_outputs,
    input_shape=(size, size),
    num_classes=num_classes,
)
```

The decoder preserves P3/P4/P5 level order, uses strides `8/16/32`, and
applies the maintained OBB semantics:

```text
xy = (txy + grid) * stride
wh = exp(twh) * stride
angle = (sigmoid(angle_logit) - 0.5) * 180 degrees
```

Decode and rotated NMS remain outside the ONNX graph. ONNX Runtime is the
reference runtime for this dynamic OBB path. Rectangular inputs are not part
of this first slice, and the TensorRT, OpenVINO, ncnn, and MegEngine examples
remain unchanged and do not establish dynamic OBB support.

2. Convert a standard YOLOX model by -f. When using -f, the above command is equivalent to:

```shell
python3 tools/export_onnx.py --output-name yolox_s.onnx -f exps/default/yolox_s.py -c yolox_s.pth
```

3. To convert your customized model, please use -f:

```shell
python3 tools/export_onnx.py --output-name your_yolox.onnx -f exps/your_dir/your_yolox.py -c your_yolox.pth
```

### ONNXRuntime Demo

Step1.
```shell
cd <YOLOX_HOME>/demo/ONNXRuntime
```

Step2. 
```shell
python3 onnx_inference.py -m <ONNX_MODEL_PATH> -i <IMAGE_PATH> -o <OUTPUT_DIR> -s 0.3 --input_shape 640,640
```
Notes:
* -m: your converted onnx model
* -i: input_image
* -s: score threshold for visualization.
* --input_shape: should be consistent with the shape you used for onnx convertion.
