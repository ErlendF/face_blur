from cv2 import imread, imwrite, rectangle
from os.path import join, exists
from glob import glob
from shutil import copyfile

from blur import round_blur


def get_file_name(nr, dir, file_ext="png"):
    return join(dir, "img" + str(nr+1).rjust(7, '0') + "." + file_ext)


def read_frame(filename):
    return imread(filename)


def write_faces(finished_seqs, img_dir, out_dir, file_ext="png", blur_function=round_blur, frames=None):
    faces_by_nr = {}
    for seq in finished_seqs:
        for face in seq:
            if face["bbox"][4] in faces_by_nr:
                faces_by_nr[face["bbox"][4]].append(face["bbox"])
            else:
                faces_by_nr[face["bbox"][4]] = [face["bbox"]]

    for frame_nr, bboxes in faces_by_nr.items():
        file_name = "img" + str(frame_nr+1).rjust(7, '0') + "." + file_ext
        out = join(out_dir, file_name)
        if frames is None:
            img = imread(join(img_dir, file_name))
        else:
            img = frames[frame_nr]
        img = blur_function(img, bboxes)
        imwrite(out, img)


def display_bboxes(finished_seqs, img_dir, out_dir, file_ext="png", color=(0, 0, 255), frames=None):
    faces_by_nr = {}
    for seq in finished_seqs:
        for face in seq:
            if face["bbox"][4] in faces_by_nr:
                faces_by_nr[face["bbox"][4]].append(face["bbox"])
            else:
                faces_by_nr[face["bbox"][4]] = [face["bbox"]]

    for frame_nr, bboxes in faces_by_nr.items():
        file_name = "img" + str(frame_nr+1).rjust(7, '0') + "." + file_ext
        out = join(out_dir, file_name)
        if frames is None:
            img = imread(join(img_dir, file_name))
        else:
            img = frames[frame_nr]
        for bbox in bboxes:
            rectangle(img, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])), color, 2)

        imwrite(out, img)


def copy_remaining_files(img_dir, out_dir, file_ext="png", frames=None, frames_by_nr=None, first_frame=None, last_frame=None):
    if first_frame is None:
        first_frame = 0

    if last_frame is None:
        # Finding the last frame
        last_frame = sorted(glob(join(img_dir, "*." + file_ext)))[-1]
        # +1 for / and . in filename, -1 to make the frame numbers 0-indexed

        last_frame = int(
            last_frame[len(img_dir)+1+len("img"):len(last_frame)-(len(file_ext)+1)]) - 1

    if frames is not None:
        if frames_by_nr is not None:
            for img_nr in range(first_frame, last_frame+1):
                if img_nr not in frames_by_nr:
                    out_file_path = join(out_dir, get_file_name(
                        img_nr, img_dir).removeprefix(img_dir).removeprefix("/"))
                    imwrite(out_file_path, frames[img_nr])
        else:
            for img_nr in range(first_frame, last_frame+1):
                filepath = get_file_name(img_nr, img_dir)
                out_file_path = join(
                    out_dir, filepath.removeprefix(img_dir).removeprefix("/"))
                if not exists(out_file_path):
                    imwrite(out_file_path, frames[img_nr])
    else:
        if frames_by_nr is not None:
            for img_nr in range(first_frame, last_frame+1):
                if img_nr not in frames_by_nr:
                    filepath = get_file_name(img_nr, img_dir)
                    out_file_path = join(
                        out_dir, filepath.removeprefix(img_dir).removeprefix("/"))
                    copyfile(filepath, out_file_path)
        else:
            for img_nr in range(first_frame, last_frame+1):
                filepath = get_file_name(img_nr, img_dir)
                out_file_path = join(
                    out_dir, filepath.removeprefix(img_dir).removeprefix("/"))

                if not exists(out_file_path):
                    copyfile(filepath, out_file_path)


def read_all_frames(img_dir, file_ext="png", first_frame=None, last_frame=None):
    frames = {}

    if first_frame is None:
        first_frame = 0

    if last_frame is None:
        # Finding the last frame
        last_frame = sorted(glob(join(img_dir, "*." + file_ext)))[-1]
        # +1 for / and . in filename, -1 to make the frame numbers 0-indexed

        last_frame = int(
            last_frame[len(img_dir)+1+len("img"):len(last_frame)-(len(file_ext)+1)]) - 1

    for img_nr in range(first_frame, last_frame+1):
        frames[img_nr] = imread(get_file_name(img_nr, img_dir))
    return frames
