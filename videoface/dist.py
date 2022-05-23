import numpy as np
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment

# Bboxes from arcface: [x1, y1, x2, y2, certainty]
score_threshold = 1.0
feat_weight = 1.0
bbox_weight = 0.001


def bbox_dist(f1, f2):
    return (abs(f1['bbox'][0]-f2['bbox'][0]) + abs(f1['bbox'][1]-f2['bbox'][1]) + abs(f1['bbox'][2]-f2['bbox'][2]) + abs(f1['bbox'][3]-f2['bbox'][3]))*abs(f1['bbox'][0]-f1['bbox'][2])/100


def feat_dist(f1, f2):
    return cosine(f1['feat'], f2['feat'])


def face_dist(f1, f2):
    fd = feat_weight*feat_dist(f1, f2)
    bd = bbox_weight*bbox_dist(f1, f2)
    return fd + bd


def get_dists(prev_faces, next_faces):
    dists = []
    for nf in next_faces:
        face_dists = []

        for pf in prev_faces:
            face_dists.append(face_dist(pf, nf))
        dists.append(face_dists)

    return dists


def map_faces(prev_faces, next_faces):
    dists = get_dists(prev_faces, next_faces)

    if len(dists) == 0:
        return []

    # Mapping the new faces to the previous faces
    _, col_ind = linear_sum_assignment(dists)

    # If the mapping exceeds the threshold, setting as -1 (unmapped)
    for i, c in enumerate(col_ind):
        if dists[i][c] > score_threshold:
            col_ind[i] = -1

    return col_ind


def compare_faces(f1, f2):
    dist = face_dist(f1, f2)
    if dist < score_threshold:
        return True, dist

    return False, dist


# face[0] => frame number
# face[1] => list of faces in frame


def compare(prev_faces, new_faces):
    if len(prev_faces[1]) != len(new_faces[1]) and new_faces[0] != prev_faces[0] + 1:
        return [], False   # There is a difference in the number of faces, and the frames are not adjacent. Shoud get more info

    identified = map_faces(prev_faces[1], new_faces[1])
    mapped = True
    for id in identified:
        if id == -1:
            mapped = False
            break

    return identified, mapped
