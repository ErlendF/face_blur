import numpy as np
from scipy.interpolate import pchip_interpolate


def interpolate(seqs, interpolator=pchip_interpolate):
    for i in range(len(seqs)):
        x1_observed = [f[0] for f in seqs[i]]
        y1_observed = [f[1] for f in seqs[i]]
        x2_observed = [f[2] for f in seqs[i]]
        y2_observed = [f[3] for f in seqs[i]]
        frame_nrs = [f[4] for f in seqs[i]]

        # +1 to make inclusive
        x_pred = np.arange(seqs[i][0][4], seqs[i][-1][4]+1)

        # Interpolating each value separately
        x1_pred = interpolator(frame_nrs, x1_observed, x_pred)
        y1_pred = interpolator(frame_nrs, y1_observed, x_pred)
        x2_pred = interpolator(frame_nrs, x2_observed, x_pred)
        y2_pred = interpolator(frame_nrs, y2_observed, x_pred)

        # Replacing sequence with fully interpolated sequence
        seqs[i] = [[x1, y1, x2, y2, nr] for x1, y1, x2, y2,
                   nr in zip(x1_pred, y1_pred, x2_pred, y2_pred, x_pred)]
    return seqs
