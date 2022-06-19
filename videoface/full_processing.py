from glob import glob
from os.path import join

from .next_list import NextList
from .dist import compare
from .deep_face import deep_face_process
from .file import get_file_name


def full_process(img_dir, file_ext="png", processing_func=deep_face_process, batch_size=32, frames=None, first_frame=None, last_frame=None):
    imgs = []
    img_nrs = []
    frames_by_nr = NextList()
    matchings = {}

    if first_frame is None:
        first_frame = 0

    if last_frame is None:
        # Finding the last frame
        last_frame = sorted(glob(join(img_dir, "*." + file_ext)))[-1]

        # +1 for / and . in filename, -1 to make the frame numbers 0-indexed
        last_frame = int(
            last_frame[len(img_dir)+1+len("img"):len(last_frame)-(len(file_ext)+1)]) - 1

    # Iterating through every image and processing them
    for img_nr in range(first_frame, last_frame+1):
        imgs.append(get_file_name(img_nr, img_dir))
        img_nrs.append(img_nr)

        if len(imgs) >= batch_size:  # Batch size reached, processing
            proc_frames = processing_func(imgs, img_nrs, frames)
            for k, v in proc_frames.items():
                frames_by_nr[k] = v

            imgs = []
            img_nrs = []

    # Processing unhandled images
    if len(imgs) != 0:
        proc_frames = processing_func(imgs, img_nrs, frames)
        for k, v in proc_frames.items():
            frames_by_nr[k] = v

        imgs = []
        img_nrs = []

    # Comparing every frame to find matchings
    for i in range(first_frame, last_frame+1):
        matchings[(i-1, i)], _ = compare(frames_by_nr[i-1], frames_by_nr[i])

    return frames_by_nr, matchings
