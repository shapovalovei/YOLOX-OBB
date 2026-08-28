#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import sys
import tempfile
import time
from collections import ChainMap
from loguru import logger
from tqdm import tqdm

import numpy as np

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocessobb_kld,
    synchronize,
    time_synchronized,
)


def _get_model_device(model):
    """Return the device of the model parameters or buffers, if available."""
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device

    buffer = next(model.buffers(), None)
    if buffer is not None:
        return buffer.device

    return None


class DOTAEvaluator:
    """
    DOTA OBB result-generation and external-evaluation adapter.

    This evaluator writes polygon detections for the external DOTA evaluation
    workflow. It does not compute internal AP metrics.
    """

    def __init__(
        self,
        dataloader,
        img_size,
        confthre,
        nmsthre,
        num_classes,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.num_images = len(dataloader.dataset)

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
    ):
        """
        Run OBB inference and write results for external DOTA evaluation.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            metric_1 (None): internal AP is not computed by this adapter.
            metric_2 (None): internal AP is not computed by this adapter.
            timing_info (str): timing and external-evaluation information.
        """
        # TODO half to amp_test
        model = model.eval()
        model_device = _get_model_device(model)
        if model_device is not None and model_device.type == "cpu" and half:
            raise ValueError(
                "DOTA evaluator CPU half precision is not supported; use half=False."
            )
        if half:
            model = model.half()
        ids = []
        data_list = {}
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        num_batches = len(self.dataloader)
        timed_batch_limit = max(num_batches - 1, 0)
        timed_batch_count = 0

        if trt_file is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("TensorRT DOTA evaluation requires CUDA.")

            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            trt_device = torch.device("cuda", torch.cuda.current_device())
            x = torch.ones(
                1, 3, test_size[0], test_size[1], device=trt_device
            )
            model(x)
            model = model_trt
            model_device = trt_device

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                if model_device is None:
                    model_device = imgs.device
                if model_device.type == "cpu" and half:
                    raise ValueError(
                        "DOTA evaluator CPU half precision is not supported; use half=False."
                    )
                input_dtype = torch.float16 if half else torch.float32
                imgs = imgs.to(device=model_device, dtype=input_dtype)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < timed_batch_limit
                if is_time_record:
                    timed_batch_count += 1
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())
                    # [[batch, n_anchors_all, 4 + 1 + 180 + 80], ...] list

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = outputs.cpu()
                outputs = postprocessobb_kld(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                ) # # #
                # (x1,y1,x2,y2,x3,y3,x4,y4, score, class_pred)
                #(x1, y1, x2, y2, x3, y3, x4, y4, conf, class)
                #(x1, y1, x2, y2, obj_conf, class_conf, class_pred)

                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            data_list.update(self.convert_to_voc_format(outputs, info_imgs, ids))

        if model_device is None:
            model_device = torch.device("cpu")
        statistics = torch.tensor(
            [inference_time, nms_time, timed_batch_count],
            dtype=torch.float32,
            device=model_device,
        )
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = ChainMap(*data_list)
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_voc_format(self, outputs, info_imgs, ids):
        """Restore image-scale polygon detections for DOTA result writing."""
        predictions = {}
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                predictions[int(img_id)] = (None, None, None)
                continue
            output = output.cpu()  # polygon, score, class
            bboxes = output[:, 0:8]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale

            cls = output[:, 9]
            scores = output[:, 8]

            predictions[int(img_id)] = (bboxes, cls, scores)
        return predictions  # image id -> (polygons, classes, scores)

    def evaluate_prediction(self, data_dict, statistics):
        # data_dict: image id -> (polygons, classes, scores)
        if not is_main_process():
            return None, None, None

        logger.info("Evaluate in main process...")

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        timed_batch_count = statistics[2].item()

        if timed_batch_count > 0 and self.dataloader.batch_size:
            a_infer_time = (
                1000
                * inference_time
                / (timed_batch_count * self.dataloader.batch_size)
            )
            a_nms_time = (
                1000
                * nms_time
                / (timed_batch_count * self.dataloader.batch_size)
            )
            time_info = ", ".join(
                [
                    "Average {} time: {:.2f} ms".format(k, v)
                    for k, v in zip(
                        ["forward", "NMS", "inference"],
                        [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                    )
                ]
            )
        else:
            time_info = (
                "Timing unavailable: no batch was eligible for latency statistics."
            )

        info = (
            time_info
            + "\nInternal AP was not computed; run the external DOTA evaluation workflow.\n"
        )

        all_boxes = [
            [[] for _ in range(self.num_images)] for _ in range(self.num_classes)
        ]
        for img_num in range(self.num_images):
            bboxes, cls, scores = data_dict[img_num]
            if bboxes is None:
                for j in range(self.num_classes):
                    all_boxes[j][img_num] = np.empty([0, 9], dtype=np.float32)
                continue
            for j in range(self.num_classes):
                mask_c = cls == j
                if sum(mask_c) == 0:
                    all_boxes[j][img_num] = np.empty([0, 9], dtype=np.float32)
                    continue

                c_dets = torch.cat((bboxes, scores.unsqueeze(1)), dim=1)
                all_boxes[j][img_num] = c_dets[mask_c].numpy()

            sys.stdout.write(
                "im_eval: {:d}/{:d} \r".format(img_num + 1, self.num_images)
            )
            sys.stdout.flush()

        with tempfile.TemporaryDirectory() as tempdir:
            self.dataloader.dataset.evaluate_detections(all_boxes, tempdir)
        return None, None, info
