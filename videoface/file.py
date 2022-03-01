from cv2 import imread
from os import path
from blur import round_blur
from matplotlib.pyplot import imsave


def read_frame(nr, dir):
    return imread(path.join(dir, "img" + str(nr+1).rjust(7, '0') + ".png"))


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
        out = path.join(out_dir, file_name)
        img = round_blur(imread(path.join(img_dir, file_name)), bboxes)
        imsave(out, img[:, :, ::-1])
