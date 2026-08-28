# YOLOX-OBB
YOLOX in DOTA with KLD loss. (Oriented Object Detection)（Rotated BBox）基于YOLOX的旋转目标检测

> **Maintained fork of [buzhidaoshenme/YOLOX-OBB](https://github.com/buzhidaoshenme/YOLOX-OBB).**

This fork preserves the original project, license, history, and attribution.
It is maintained independently and contains additional regression-tested fixes.
It is not the official YOLOX-OBB project.

## Fixes in this maintained fork

### Correct OBB geometry through augmentation

The original `random_perspective`/Mosaic path treated encoded OBB center and
dimension fields as HBB corner coordinates. After image transforms this could
leave the rotated-box geometry inconsistent, including a stale angle. The fork
decodes the OBB into image-space corners, applies the same affine or
perspective transform as the image, clips the transformed geometry, and
reconstructs a canonical OBB. Mosaic no longer clips encoded dimensions as if
they were an axis-aligned rectangle.

### Correct KLD prediction/target order

The KLD training-head call now passes the prediction tensor first and the
target tensor second, matching the semantics of the KLD implementation and its
gradient direction.

Both fixes have repository-local regression coverage. From the repository root,
run:

```shell
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Installation 
1. Install YOLOX-OBB(You can refer to the installation of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX))
```shell
cd YOLOX-OBB
pip3 install -r requirements.txt
pip install -v -e . --no-build-isolation
```
2. Install pycocotools
```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
PyTorch must already be installed in the active environment because the root
package uses PyTorch C++ extension tooling. The root installation builds both
the YOLOX native extension and the rotated-IoU extension. A compatible C/C++
compiler and build toolchain are required. SWIG is not required for normal
installation because the generated Python and C++ wrapper sources are included
in the repository. SWIG is only needed by maintainers who intentionally
regenerate the wrapper from `polyiou.i`.

Verify the rotated-IoU capability with:
```shell
python - <<'PY'
import yolox.utils
from DOTA_devkit_YOLO import polyiou

p = polyiou.VectorDouble([0, 0, 2, 0, 2, 2, 0, 2])
print(polyiou.iou_poly(p, p))
PY
```
3. Install apex
```shell
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
cd -
```
## Data Preparation
1. Split images and annotations(You can refer to [DOTA_devkit_YOLO](https://github.com/hukaixuan19970627/DOTA_devkit_YOLO))
```shell
 python DOTA_devkit_YOLO/ImgSplit_multi_process.py
 ```
 2. Transform annotations into voc-like format
 
 * `This is a object in voc-like format annotation:`
 <img src="assets/voc-like .png" width="500" >
 
 ```shell
 python custom tools/DOTA2VOC_obb.py
 ```
 3. Organize Directories(All annotations of train-images and val-images must be put into Annotations folder)
 ```
 |--your_data
     |--VOC2012
         |--Annotations
             |-- xxx.xml
                 ... 
         |--ImageSets
             |--Main
                 |--train.txt
                 |--val.txt
                 |--test.txt
         |--JPEGImages
         |--JPEGImages-val
         |-JPEGImages-test
```
## Train 
1. Modify configs

　change the data path with yours in [yolox_dota_s_obb_kld.py](https://github.com/buzhidaoshenme/YOLOX-OBB/blob/main/exps/example/yolox_voc/yolox_dota_s_obb_kld.py)
```
data_dir = 'your_data_path'
```
2. Train
```
CUDA_VISIBLE_DEVICES=0,1 python3 tools/train.py -f exps/example/yolox_voc/yolox_dota_s_obb_kld.py -d 2 -b 16 --fp16 -c weights/yolox_s.pth.tar
```
## Val
1. get results
```
CUDA_VISIBLE_DEVICES=0,1 python tools/eval.py -f exps/example/yolox_voc/yolox_dota_s_obb_kld.py -d 2 -b 16 -c YOLOX_outputs/yolox_dota_s_obb_kld/latest_ckpt.pth
```
　The evaluator writes DOTA polygon result files to
`your_data/results/VOC2012/Main`. It does not compute AP internally and
returns `(None, None, timing_info)`; use the external DOTA evaluation tooling
below for canonical metrics.
 * `If test, you must comment line 151 'target = self.load_anno(index)' and uncomment line 152 'target = []' in dota_obb.py before run the above instruction. Because test-set has no annotations.`

2. Merge results(You can refer to [DOTA_devkit_YOLO](https://github.com/hukaixuan19970627/DOTA_devkit_YOLO))
```
python DOTA_devkit_YOLO/ResultMerge.py
```
3. Evaluation
```
python DOTA_devkit_YOLO/dota_v1.5_evaluation_task1.py(You can refer to [DOTA_devkit_YOLO](https://github.com/hukaixuan19970627/DOTA_devkit_YOLO))
```
 * `If test, you should upload your results to DOTA Evaluation Server.`

The direct `DOTAEvaluator` supports single-process CPU float32 evaluation.
CPU float16 evaluation is intentionally rejected. The `tools/eval.py` command
and distributed launcher remain CUDA-oriented.

## Unfortunately 
The historical external DOTA evaluation workflow reported 0.712 mAP@0.5 on
DOTA v1.0.

## Reference
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

[DOTA_devkit_YOLO](https://github.com/hukaixuan19970627/DOTA_devkit_YOLO)

[YOLOv5_DOTA_OBB](https://github.com/hukaixuan19970627/YOLOv5_DOTA_OBB)

[RotationDetection](https://github.com/yangxue0827/RotationDetection)
