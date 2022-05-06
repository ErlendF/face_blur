import numpy as np
from scipy.spatial.distance import cosine

# Bboxes from arcface: [x1, y1, x2, y2, certainty]
score_threshold = 2
feat_weight = 0.8
bbox_weight = 0.001


def bbox_dist(f1, f2):
    return (abs(f1['bbox'][0]-f2['bbox'][0]) + abs(f1['bbox'][1]-f2['bbox'][1]) + abs(f1['bbox'][2]-f2['bbox'][2]) + abs(f1['bbox'][3]-f2['bbox'][3]))*abs(f1['bbox'][0]-f1['bbox'][2])/100


def feat_dist(f1, f2):
    return cosine(f1['feat'], f2['feat'])


def face_dist(f1, f2):
    fd = feat_weight*feat_dist(f1, f2)
    bd = bbox_weight*bbox_dist(f1, f2)
    return fd + bd


# TODO: try not to loop over every face and sequence => n^2
def get_dists(prev_faces, next_faces):
    dists = []
    for nf in next_faces:
        face_dists = []

        for i, pf in enumerate(prev_faces):
            face_dists.append((i, face_dist(pf, nf)))
        face_dists.sort(key=lambda x: x[1])
        dists.append(face_dists)

    return dists


def map_faces(prev_faces, next_faces):
    dists = get_dists(prev_faces, next_faces)

    # List mapping the new faces to the previous faces, -1 is not mapped
    identified = [-1] * len(prev_faces)

    # Looping over the distances to map faces to previous sequences
    for i, face_dists in enumerate(dists):
        for dist in face_dists:  # TODO: minimize total cost over entire set
            if dist[1] > score_threshold:   # Only accepting scores lower than the threshold
                break

            if identified[dist[0]] == -1:   # Using the sequence with lowest distance
                identified[dist[0]] = i
                break

    return identified


def compare_faces(f1, f2):
    dist = face_dist(f1, f2)
    if dist < score_threshold:
        return True, dist

    return False, dist
