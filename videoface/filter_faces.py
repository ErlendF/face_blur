from glob import glob
from os.path import join
from sys import float_info
from scipy.spatial.distance import cosine
import numpy as np

from .deep_face import deep_face_process

# Preprocess known face to use in "filter_known_faces"
# The processed known faces are compared to the faces in the video, and should therefore use the same processing function as when parsing the video


def init_known_faces(known_people_img_dir, processing_func=deep_face_process, frames=None):
    processed_faces = {}

    if frames is None:
        img_names = sorted(glob(join(known_people_img_dir, "*")))
        img_nrs = np.arange(0, len(img_names))
        processed_faces = processing_func(img_names, img_nrs)
    else:
        img_names = np.arange(0, len(frames))
        img_nrs = np.arange(0, len(frames))
        processed_faces = processing_func(img_names, img_nrs, frames=frames)

    if len(processed_faces) == 0:
        return None

    known_faces = []
    for img in processed_faces.values():
        for f in img:
            # The positions of the known people from the reference photos is of no importance, removing it
            known_faces.append(f["feat"])

    return known_faces


def filter_known_faces(facial_sequences, known_faces, remove_known=False, samples=5, threshold=0.75):
    comparisons = [0] * len(facial_sequences)

    # Comparing each known face to a sample of each facial sequence
    for i, seq in enumerate(facial_sequences):
        for j, f in enumerate(known_faces):
            if len(seq) < samples:
                for s in seq:
                    sim = 1-cosine(f, s["feat"])
                    if sim > comparisons[i]:
                        comparisons[i] = sim
            else:
                intv = len(seq) // samples
                if intv < 1:
                    intv = 1

                for k in range(samples):
                    sim = 1-cosine(f, seq[k*intv]["feat"])
                    if sim > comparisons[i]:
                        comparisons[i] = sim

    if remove_known:
        return [f for f, s in zip(facial_sequences, comparisons) if s < threshold]
    else:
        return [f for f, s in zip(facial_sequences, comparisons) if s >= threshold]


# Filter a selected face based on location and frame number. Use the sequence to identify other sequences with the same face. x = y = 0 is the top left corner of the frame.
def filter_selected_face(sequences, frame_number, x, y, remove_known=True, samples=5, threshold=0.75):
    closest_seq = -1
    closest_dist = float_info.max

    # Identifying the sequence closest to the selection
    for i, seq in enumerate(sequences):
        if seq[0]["bbox"][4] > frame_number and seq[-1]["bbox"][4] < frame_number:
            continue

        for s in seq:
            if s["bbox"][4] != frame_number:
                continue

            dist = abs(x-s["bbox"][0]) + abs(x-s["bbox"][2]) + \
                abs(y-s["bbox"][1]) + abs(y-s["bbox"][3])
            if dist < closest_dist:
                closest_dist = dist
                closest_seq = i

    if closest_seq == -1:
        if remove_known:
            return sequences
        return []

    # Getting samples from the closest identified sequence
    seq_samples = []
    if len(sequences[closest_seq]) < samples:
        seq_samples = sequences[closest_seq]
    else:
        intv = len(sequences[closest_seq]) // samples
        if intv < 1:
            intv = 1

        for k in range(samples):
            seq_samples.append(sequences[closest_seq][intv*k])

    # Comparing the samples to other sequences to identify sequences with the same face
    identical = [closest_seq]
    for i, seq in enumerate(sequences):
        if i == closest_seq:
            continue

        intv = len(seq) // samples
        if intv < 1:
            intv = 1

        found = False
        for j in range(samples):
            if j >= len(seq):
                break

            for s in seq_samples:
                sim = 1-cosine(s["feat"], seq[j*intv]["feat"])
                if sim >= threshold:
                    identical.append(i)  # The faces are likely the same
                    found = True
                    break

            if found:
                break

    if remove_known:
        return [s for i, s in enumerate(sequences) if i not in identical]

    return [s for i, s in enumerate(sequences) if i in identical]


# Used to filter likely false positives
def filter_short_sequences(seqs, min_length=10):
    return [s for s in seqs if len(s) >= min_length]
