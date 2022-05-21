from tkinter import image_names
from face_recognition import batch_face_locations, face_encodings
from .file import read_frame


def dlib_process(img_names, img_nrs):
    if len(image_names) == 0:
        return {}

    imgs = []
    for filename in img_names:
        imgs.append(read_frame(filename))

    locs = batch_face_locations(
        imgs, number_of_times_to_upsample=0, batch_size=len(imgs))

    faces = {}
    for ls, img, inr in zip(locs, imgs, img_nrs):
        img_faces = []
        fs = face_encodings(img, known_face_locations=ls, model="large")
        for l, f in zip(ls, fs):
            img_faces.append(
                {'bbox': [l[3], l[0], l[1], l[2], inr], 'feat': f.tolist()})

        faces[inr] = img_faces

    return faces
