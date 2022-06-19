from cv2 import resize, imread
from glob import glob
from os.path import join
import numpy as np
from transnetv2 import TransNetV2

from .file import get_file_name
from .next_list import NextList

model = TransNetV2()


def get_shot_transitions(img_dir, file_ext="png", shot_transition_threshold=0.5, frames=None, first_frame=None, last_frame=None):
    st = NextList()
    st_frames = []

    if frames is None:
        if first_frame is None:
            first_frame = 0

        if last_frame is None:
            # Finding the last frame
            last_frame = sorted(glob(join(img_dir, "*." + file_ext)))[-1]

            # +1 for / and . in filename, -1 to make the frame numbers 0-indexed
            last_frame = int(
                last_frame[len(img_dir)+1+len("img"):len(last_frame)-(len(file_ext)+1)]) - 1

        for img_nr in range(first_frame, last_frame+1):
            st_frames.append(
                resize(imread(get_file_name(img_nr, img_dir)), dsize=(48, 27)))
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
