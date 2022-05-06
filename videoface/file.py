from cv2 import imread, rectangle
from os.path import join, exists
from glob import glob
from shutil import copyfile
from matplotlib.pyplot import imsave

from blur import round_blur


def get_file_name(nr, dir):
    return join(dir, "img" + str(nr+1).rjust(7, '0') + ".png")


def read_frame(filename):
    return imread(filename)


def write_faces(finished_seqs, img_dir, out_dir):
    faces_by_nr = {}
    for seq in finished_seqs:
        for face in seq:
            if face["bbox"][4] in faces_by_nr:
                faces_by_nr[face["bbox"][4]].append(face["bbox"])
            else:
                faces_by_nr[face["bbox"][4]] = [face["bbox"]]

    for frame_nr, bboxes in faces_by_nr.items():
        file_name = "img" + str(frame_nr+1).rjust(7, '0') + ".png"
        out = join(out_dir, file_name)
        img = round_blur(imread(join(img_dir, file_name)), bboxes)
        imsave(out, img[:, :, ::-1])


def display_bboxes(finished_seqs, img_dir, out_dir):
    faces_by_nr = {}
    for seq in finished_seqs:
        for face in seq:
            if face["bbox"][4] in faces_by_nr:
                faces_by_nr[face["bbox"][4]].append(face["bbox"])
            else:
                faces_by_nr[face["bbox"][4]] = [face["bbox"]]

    for frame_nr, bboxes in faces_by_nr.items():
        file_name = "img" + str(frame_nr+1).rjust(7, '0') + ".png"
        out = join(out_dir, file_name)
        img = imread(join(img_dir, file_name))
        for bbox in bboxes:
            rectangle(img, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

        imsave(out, img[:, :, ::-1])


def copy_remaining_files(in_dir, out_dir):
    for filepath in sorted(glob(join(in_dir, "*.png"))):
        filename = filepath[len(in_dir)+1:]
        out_file_path = join(out_dir, filename)

        if not exists(out_file_path):
            copyfile(filepath, out_file_path)
