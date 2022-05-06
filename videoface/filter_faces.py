from glob import glob
from os.path import join
from sys import float_info
from scipy.spatial.distance import cosine

from .face_recognition import face_recognition_process

# Should use the same processing function as when parsing the video


def init_known_faces(known_people_img_dir, processing_func=face_recognition_process):
    processed_faces = []

    for filepath in sorted(glob(join(known_people_img_dir, "*"))):
        processed_faces.append(processing_func([filepath], [0]))

    if len(processed_faces) == 0:
        return None

    known_faces = []
    for m in processed_faces:
        for img_f in m.values():
            for f in img_f:
                # The positions of the known people from the reference photos is of no importance, removing it
                known_faces.append(f["feat"])

    return known_faces


def filter_known_faces(facial_sequences, known_faces, remove_known=False, samples=5, threshold=0.75):
    comparrisons = [0] * len(facial_sequences)

    # Comparing each known face to a sample of each facial sequence
    for i, seq in enumerate(facial_sequences):
        for j, f in enumerate(known_faces):
            if len(seq) < samples:
                for s in seq:
                    sim = 1-cosine(f, s["feat"])
                    if sim > comparrisons[i]:
                        comparrisons[i] = sim
            else:
                intv = len(seq) // samples
                if intv < 1:
                    intv = 1

                for k in range(samples):
                    sim = 1-cosine(f, seq[k*intv]["feat"])
                    if sim > comparrisons[i]:
                        comparrisons[i] = sim

    if remove_known:
        return [f for f, s in zip(facial_sequences, comparrisons) if s < threshold]
    else:
        return [f for f, s in zip(facial_sequences, comparrisons) if s >= threshold]


# Filter a selected face based on location and frame number. Use the sequence to identify other sequences with the same face
def filter_selected_face(sequences, frame_number, x, y, remove_known=True, samples=5, threshold=0.75):
    closest_seq = -1
    closest_dist = float_info.max
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

    seq_samples = []
    if len(sequences[closest_seq]) < samples:
        seq_samples = sequences[closest_seq]
    else:
        intv = len(sequences[closest_seq]) // samples
        if intv < 1:
            intv = 1

        for k in range(samples):
            seq_samples.append(sequences[closest_seq][intv*k])

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
                    identical.append(i)
                    found = True
                    break

            if found:
                break

    if remove_known:
        return [s for i, s in enumerate(sequences) if i not in identical]

    return [s for i, s in enumerate(sequences) if i in identical]