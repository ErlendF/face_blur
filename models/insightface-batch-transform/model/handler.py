"""
ModelHandler defines an example model handler for load and inference requests for MXNet CPU models
"""
import glob
import json
import logging
import os
import io
from collections import namedtuple
from insightface.app import FaceAnalysis
from PIL import Image

import numpy as np


class Handler(object):
    def __init__(self):
        self.initialized = False

    def initialize(self, context):
        self.initialized = True
        self.fa = FaceAnalysis(root=os.getenv("MODEL_PATH", "/opt/ml/model"))
        self.fa.prepare(ctx_id=0, det_size=(640, 640))

    def inference(self, data):
        resp = []
        for input in data:
            body = input.get("body")
            img = Image.open(io.BytesIO(body)).convert("RGB")
            faces = self.fa.get(np.array(img)[:, :, ::-1])
            parsed = []
            for face in faces:
                obj = {}
                for k, v in face.items():
                    if isinstance(v, np.ndarray):
                        obj[k] = v.tolist()
                    elif isinstance(v, np.integer):
                        obj[k] = int(v)
                    elif isinstance(v, np.floating):
                        obj[k] = float(v)
                    else:
                        obj[k] = v
                parsed.append(obj)
            resp.append(parsed)

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
