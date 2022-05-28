from cv2 import resize, imread
from glob import glob
from os.path import join
import numpy as np
from transnetv2 import TransNetV2

from .next_list import NextList

model = TransNetV2()


def get_shot_transitions(img_dir, file_ext="png", shot_transition_threshold=0.5, frames=None):
    st = NextList()
    st_frames = []
    if frames is None:
        for filepath in sorted(glob(join(img_dir, "*" + file_ext))):
            st_frames.append(resize(imread(filepath), dsize=(48, 27)))
    else:
        for i in range(len(frames)):
            st_frames.append(resize(frames[i], dsize=(48, 27)))

    if len(st_frames) == 0:
        return None

    st_frames = np.array(st_frames)
    predictions, _ = model.predict_frames(
        st_frames[:, :, :, ::-1])
    for i, pred in enumerate(predictions):
        if pred >= shot_transition_threshold:
            st[i] = True

    return st
