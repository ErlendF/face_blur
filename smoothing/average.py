import numpy as np
from copy import deepcopy


def avg_smoothing(seqs, smoothing_length=9):
    df = smoothing_length//2
    seq_copy = deepcopy(seqs)

    for i in range(len(seqs)):
        ln = len(seqs[i])
        seq_copy[i] = np.array(seq_copy[i])
        for j in range(ln):
            first = max(0, j-df)
            last = min(ln, j+df)
            seqs[i][j][0] = np.mean(seq_copy[i][first:last, 0])
            seqs[i][j][1] = np.mean(seq_copy[i][first:last, 1])
            seqs[i][j][2] = np.mean(seq_copy[i][first:last, 2])
            seqs[i][j][3] = np.mean(seq_copy[i][first:last, 3])

    return seqs
