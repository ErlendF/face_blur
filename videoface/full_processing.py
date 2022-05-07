from glob import glob
from os.path import join

from .next_list import NextList
from .dist import compare
from .face_recognition import face_recognition_process


def full_process(img_dir, file_ext="png", processing_func=face_recognition_process, batch_size=32):
    imgs = []
    img_nrs = []
    frames_by_nr = NextList()
    matchings = {}

    # Iterating through every image and processing them
    for filepath in sorted(glob(join(img_dir, "*" + file_ext))):
        imgs.append(filepath)
        img_nrs.append(int(filepath.split(
            "/")[-1].removeprefix("img").removesuffix("." + file_ext))-1)

        if len(imgs) >= batch_size:  # Batch size reached, processing
            frames = processing_func(imgs, img_nrs)
            for k, v in frames.items():
                frames_by_nr[k] = v

            imgs = []
            img_nrs = []

    # Processing unhandled images
    if len(imgs) != 0:
        frames = processing_func(imgs, img_nrs)
        for k, v in frames.items():
            frames_by_nr[k] = v

        imgs = []
        img_nrs = []

    for i in range(1, len(frames_by_nr)):
        matchings[(i-1, i)], _ = compare(frames_by_nr[i-1], frames_by_nr[i])

    return frames_by_nr, matchings
