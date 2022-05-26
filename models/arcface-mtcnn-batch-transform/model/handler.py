from __future__ import absolute_import, division, print_function

import numpy as np
import mxnet as mx
import os
from mtcnn_detector import MtcnnDetector
from utils import get_model, get_input, get_feature

import io
from PIL import Image

import numpy as np

threshold = 1
margin = 0
det_threshold = [0.7, 0.8, 0.9]
model_name = "resnet100.onnx"


class Handler(object):
    def __init__(self):
        self.initialized = False

    def initialize(self, context):
        self.initialized = True
        model_path = os.getenv("MODEL_PATH", "/opt/ml/model")
        full_model_path = model_path + "/" + model_name
        print("model path", model_path)
        print("full model path", full_model_path)

        if len(mx.test_utils.list_gpus()) == 0:
            ctx = mx.cpu()
        else:
            ctx = mx.gpu(0)

        self.detector = MtcnnDetector(
            model_folder=model_path,
            ctx=ctx,
            num_worker=1,
            accurate_landmark=True,
            threshold=det_threshold,
        )

        self.model = get_model(ctx, full_model_path)

    def inference(self, data):
        resp = []
        for input in data:
            body = input.get("body")
            img = Image.open(io.BytesIO(body)).convert("RGB")
            input = get_input(self.detector, np.array(img)[:, :, ::-1].copy())
            if input is None:
                resp.append([])
                continue

            inp, bboxes = input
            feats = get_feature(self.model, inp)

            faces = []
            for bbox, feat in zip(bboxes, feats):
                if isinstance(bbox, np.ndarray):
                    bbox = bbox.tolist()

                if isinstance(feat, np.ndarray):
                    feat = feat.tolist()

                for i in range(len(feat)):
                    if isinstance(feat[i], np.ndarray):
                        feat[i] = feat[i].tolist()

                faces.append({"bbox": bbox, "feat": feat})

            resp.append(faces)

        return resp

    def handle(self, data, context):
        return self.inference(data)


_service = Handler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
