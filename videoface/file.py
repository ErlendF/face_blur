from cv2 import imread
from os.path import join, exists
from glob import glob
from shutil import copyfile
from matplotlib.pyplot import imsave

from blur import round_blur


def read_frame(nr, dir):
    return imread(join(dir, "img" + str(nr+1).rjust(7, '0') + ".png"))


def write_faces(finished_seqs, img_dir, out_dir):
    faces_by_nr = {}
    for seq in finished_seqs:
        for face in seq:
            if face[4] in faces_by_nr:
                faces_by_nr[face[4]].append(face)
            else:
                faces_by_nr[face[4]] = [face]

    for frame_nr, bboxes in faces_by_nr.items():
        file_name = "img" + str(frame_nr+1).rjust(7, '0') + ".png"
        out = join(out_dir, file_name)
        img = round_blur(imread(join(img_dir, file_name)), bboxes)
        imsave(out, img[:, :, ::-1])


def copy_remaining_files(in_dir, out_dir):
    for filepath in sorted(glob(join(in_dir, "*.png"))):
        filename = filepath[len(in_dir)+1:]
        out_file_path = join(out_dir, filename)

        if not exists(out_file_path):
            copyfile(filepath, out_file_path)
