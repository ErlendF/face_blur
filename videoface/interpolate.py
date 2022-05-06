import numpy as np
from scipy.interpolate import pchip_interpolate


def interpolate(seqs, interpolator=pchip_interpolate):
    for i in range(len(seqs)):
        if len(seqs[i]) == 1:
            continue

        x1_observed = [f["bbox"][0] for f in seqs[i]]
        y1_observed = [f["bbox"][1] for f in seqs[i]]
        x2_observed = [f["bbox"][2] for f in seqs[i]]
        y2_observed = [f["bbox"][3] for f in seqs[i]]
        frame_nrs = [f["bbox"][4] for f in seqs[i]]

        # Using the previous valid feature for every frame missing it
        feats = []
        next = 0
        for j in range(seqs[i][-1]["bbox"][4]+1-seqs[i][0]["bbox"][4]):
            feats.append(seqs[i][next]["feat"])

            if seqs[i][next]["bbox"][4] == j+seqs[i][0]["bbox"][4]:
                next += 1

        # +1 to make inclusive
        x_pred = np.arange(seqs[i][0]["bbox"][4], seqs[i][-1]["bbox"][4]+1)

        # Interpolating each value separately
        x1_pred = interpolator(frame_nrs, x1_observed, x_pred)
        y1_pred = interpolator(frame_nrs, y1_observed, x_pred)
        x2_pred = interpolator(frame_nrs, x2_observed, x_pred)
        y2_pred = interpolator(frame_nrs, y2_observed, x_pred)

        # Replacing sequence with fully interpolated sequence
        seqs[i] = [{"bbox": [x1, y1, x2, y2, nr], "feat": feat.copy()} for x1, y1, x2, y2,
                   nr, feat in zip(x1_pred, y1_pred, x2_pred, y2_pred, x_pred, feats)]
    return seqs
