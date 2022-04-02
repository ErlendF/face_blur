from .file import read_frame
from deepface import DeepFace
from deepface.detectors import FaceDetector
import cv2
import numpy as np

import tensorflow as tf
tf_version = tf.__version__
tf_major_version = int(tf_version.split(".")[0])
tf_minor_version = int(tf_version.split(".")[1])

if tf_major_version == 1:
    import keras
    from keras.preprocessing.image import load_img, save_img, img_to_array
    from keras.applications.imagenet_utils import preprocess_input
    from keras.preprocessing import image
elif tf_major_version == 2:
    from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
    from tensorflow.keras.applications.imagenet_utils import preprocess_input
    from tensorflow.keras.preprocessing import image

detector_backend = "retinaface"
model_name = "VGG-Face"
face_detector = FaceDetector.build_model(detector_backend)
model = DeepFace.build_model(model_name)


# Based on https://github.com/serengil/deepface/blob/b13cca851f6415372e7baf988ba6d2098af1297e/deepface/commons/functions.py#L172
# Altered to get regions and process all faces in the image
def preprocess_faces(img, target_size=(224, 224), grayscale=False, enforce_detection=True, detector_backend='opencv', return_region=False, align=True):
    objs = FaceDetector.detect_faces(
        face_detector, detector_backend, img, align)

    pixels = []
    regions = []

    for face, region in objs:
        regions.append(region)
        # post-processing
        if grayscale == True:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        if face.shape[0] > 0 and face.shape[1] > 0:
            factor_0 = target_size[0] / face.shape[0]
            factor_1 = target_size[1] / face.shape[1]
            factor = min(factor_0, factor_1)

            dsize = (int(face.shape[1] * factor), int(face.shape[0] * factor))
            face = cv2.resize(face, dsize)

            # Then pad the other side to the target size by adding black pixels
            diff_0 = target_size[0] - face.shape[0]
            diff_1 = target_size[1] - face.shape[1]
            if grayscale == False:
                # Put the base image in the middle of the padded image
                face = np.pad(face, ((diff_0 // 2, diff_0 - diff_0 // 2),
                              (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)),   'constant')
            else:
                face = np.pad(face, ((diff_0 // 2, diff_0 - diff_0 // 2),
                              (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

        # ------------------------------------------

        # double check: if target image is not still the same size with target.
        if face.shape[0:2] != target_size:
            face = cv2.resize(face, target_size)

        # ---------------------------------------------------

        # normalizing the image pixels

        face_pixels = image.img_to_array(face)  # what this line doing? must?
        face_pixels = np.expand_dims(face_pixels, axis=0)
        face_pixels /= 255  # normalize input in [0, 1]
        pixels.append(face_pixels)

    return pixels, regions


def deep_face_process(img_names, img_nrs):
    faces = {}
    for img_name, img_nr in zip(img_names, img_nrs):
        img = read_frame(img_name)
        pixels, regions = preprocess_faces(
            img, detector_backend=detector_backend)

        embeddings = []

        for p in pixels:
            p = DeepFace.functions.normalize_input(img=p, normalization="base")
            embeddings.append(model.predict(p)[0].tolist())  # TODO: batches

        img_faces = []

        for bbox, feat in zip(regions, embeddings):
            img_faces.append({"bbox": bbox, "feat": feat})

        faces[img_nrs] = img_faces

    return faces
