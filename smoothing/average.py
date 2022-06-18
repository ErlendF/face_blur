import numpy as np


def avg_smoothing(seqs, smoothing_width=9):
    df = smoothing_width//2

    for i in range(len(seqs)):
        # Getting a copy of the sequence only containing bounding boxes to make it a numpy array
        seq_copy = np.array([s["bbox"].copy() for s in seqs[i]])

        # Smoothing each point of the bounding box individually
        for j in range(len(seqs[i])):
            first = max(0, j-df)
            last = min(len(seqs[i]), j+df)
            seqs[i][j][0] = np.mean(seq_copy[first:last, 0])
            seqs[i][j][1] = np.mean(seq_copy[first:last, 1])
            seqs[i][j][2] = np.mean(seq_copy[first:last, 2])
            seqs[i][j][3] = np.mean(seq_copy[first:last, 3])

    return seqs
