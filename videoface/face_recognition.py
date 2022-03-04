from face_recognition import batch_face_locations, face_encodings
from .dist import map_faces


def process(imgs, img_nrs, model_size="large"):  # TODO: make interchangeable
    locs = batch_face_locations(
        imgs, number_of_times_to_upsample=0, batch_size=len(imgs))

    faces = {}
    for ls, img, inr in zip(locs, imgs, img_nrs):
        img_faces = []
        fs = face_encodings(img, known_face_locations=ls, model=model_size)
        for l, f in zip(ls, fs):
            img_faces.append(
                {'bbox': [l[3], l[0], l[1], l[2], inr], 'feat': f.tolist()})

        faces[inr] = img_faces

    return faces


# [0] => frame number
# [1] => list of faces in frame
def compare(prev_faces, new_faces):
    if len(prev_faces[1]) != len(new_faces[1]) and new_faces[0] != prev_faces[0] + 1:
        return [], False   # Shoud get more info

    identified = map_faces(prev_faces[1], new_faces[1])
    mapped = True
    for id in identified:
        if id == -1:
            mapped = False
            break

    return identified, mapped


def remove_feats(seqs):
    for i in range(len(seqs)):
        seqs[i] = [face['bbox'] for face in seqs[i]]
    return seqs
