from cv2 import resize, imread
from glob import glob
from os.path import join
import numpy as np
from transnetv2 import TransNetV2

from .dynamic import NextList

cut_detection_model = TransNetV2()


def get_scene_changes(img_dir, file_ext="png", scene_change_threshold=0.5):
    sc = NextList()
    sc_frames = []
    for filepath in sorted(glob(join(img_dir, "*" + file_ext))):
        sc_frames.append(resize(imread(filepath), dsize=(48, 27)))

    sc_frames = np.array(sc_frames)
    single_frame_predictions, _ = cut_detection_model.predict_frames(
        sc_frames[:, :, :, ::-1])
    for i, pred in enumerate(single_frame_predictions):
        if pred >= scene_change_threshold:
            sc[i] = True

    return sc
