from __future__ import absolute_import, division, print_function
import cv2
import sys
import numpy as np
import mxnet as mx
import os

from scipy import misc
import random
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
from skimage import transform as trans
import matplotlib.pyplot as plt
from mxnet.contrib.onnx.onnx2mx.import_model import import_model


def get_model(ctx, model):
    image_size = (112, 112)
    # Import ONNX model
    sym, arg_params, aux_params = import_model(model)
    # Define and binds parameters to the network
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[("data", (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


def preprocess(img, bbox=None, landmark=None, **kwargs):
    M = None
    image_size = []
    str_image_size = kwargs.get("image_size", "")
    # Assert input shape
    if len(str_image_size) > 0:
        image_size = [int(x) for x in str_image_size.split(",")]
        if len(image_size) == 1:
            image_size = [image_size[0], image_size[0]]
        assert len(image_size) == 2
        assert image_size[0] == 112
        assert image_size[0] == 112 or image_size[1] == 96

    # Do alignment using landmark points
    if landmark is not None:
        assert len(image_size) == 2
        src = np.array(
            [
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        if image_size[1] == 112:
            src[:, 0] += 8.0
        dst = landmark.astype(np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
        assert len(image_size) == 2
        warped = cv2.warpAffine(
            img, M, (image_size[1], image_size[0]), borderValue=0.0)
        return warped

    # If no landmark points available, do alignment using bounding box. If no bounding box available use center crop
    if M is None:
        if bbox is None:
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get("margin", 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1]: bb[3], bb[0]: bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret


def get_input(detector, face_img):
    # Pass input images through face detector
    ret = detector.detect_face(face_img, det_type=0)
    if ret is None:
        return None
    bboxes, all_points = ret
    if bboxes.shape[0] == 0:
        return None

    aligned = []
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i, 0:4]
        points = all_points[i, :].reshape((2, 5)).T
        # Call preprocess() to generate aligned images
        nimg = preprocess(face_img, bbox, points, image_size="112,112")
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned.append(np.transpose(nimg, (2, 0, 1)))
    return aligned, bboxes


# TODO: use data batching
def get_feature(model, aligned):
    embeddings = []
    for i, a in enumerate(aligned):
        input_blob = np.expand_dims(a, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        model.forward(db, is_train=False)
        embedding = model.get_outputs()[0].asnumpy()
        embedding = sklearn.preprocessing.normalize(
            embedding).flatten()
        embeddings.append(embedding)
    return embeddings


def init_db(model, detector, db_path):
    representations = []
    for r, d, f in os.walk(db_path):
        for file in f:
            if (
                (".jpg" in file.lower())
                or (".png" in file.lower())
                or (".jpeg" in file.lower())
            ):
                exact_path = r + "/" + file
                print(exact_path)
                img = cv2.imread(exact_path)
                input, bboxes = get_input(detector, img)
                representations += get_feature(model, input)

    return representations


def get_dist(db, rep):
    min = sys.float_info.max
    for face in db:
        dist = np.sum(np.square(face - rep))
        if dist < min:
            min = dist

    return min
