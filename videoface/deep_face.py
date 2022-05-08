from .file import read_frame
from deepface import DeepFace
from deepface.detectors import FaceDetector
from cv2 import resize, cvtColor, COLOR_BGR2GRAY
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing import image

detector_backend = "retinaface"
model_name = "Facenet512"
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
            face = cvtColor(face, COLOR_BGR2GRAY)

        if face.shape[0] > 0 and face.shape[1] > 0:
            factor_0 = target_size[0] / face.shape[0]
            factor_1 = target_size[1] / face.shape[1]
            factor = min(factor_0, factor_1)

            dsize = (int(face.shape[1] * factor), int(face.shape[0] * factor))
            face = resize(face, dsize)

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
            face = resize(face, target_size)

        # ---------------------------------------------------

        # normalizing the image pixels

        face_pixels = image.img_to_array(face)  # what this line doing? must?
        face_pixels = np.expand_dims(face_pixels, axis=0)
        face_pixels /= 255  # normalize input in [0, 1]
        pixels.append(face_pixels)

    return pixels, regions


def deep_face_process(img_names, img_nrs):
    faces = {}
    pixels_con = []
    pixels_len = {}

    for img_name, img_nr in zip(img_names, img_nrs):
        img = read_frame(img_name)
        pixels, regions = preprocess_faces(
            img,
            detector_backend=detector_backend,
            target_size=(160, 160)
        )

        pixels_len[img_nr] = len(pixels)
        if len(pixels) != 0:
            pixels = np.array(pixels)  # .squeeze(axis=0)

            # FaceNet2018 Normalization https://github.com/serengil/deepface/blob/fb68d4a8f816a9cbea488f4dc24c16b78ac3d9b2/deepface/commons/functions.py#L126
            pixels /= 127.5
            pixels -= 1

            if len(pixels_con) == 0:
                pixels_con = pixels
            else:
                pixels_con = np.concatenate((pixels_con, pixels))

        img_faces = []

        for bbox in regions:
            # Converting from [x, y, width, height] to [x1, y1, x2, y2]
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]

            img_faces.append({"bbox": bbox + [img_nr]})

        faces[img_nr] = img_faces

    pred = model.predict(pixels_con.squeeze(axis=1))
    count = 0
    for k, v in pixels_len.items():
        for i in range(v):
            faces[k][i]["feat"] = pred[count].tolist()
            count += 1

    return faces
