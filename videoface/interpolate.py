def interpolate(seqs):
    to_add = []

    for i in range(len(seqs)):
        prev_frame = -1
        for j, s in enumerate(seqs[i]):
            if prev_frame == -1:    # No previous frames to interpolate
                prev_frame = s[4]
                continue

            if s[4] == prev_frame+1:    # No frames between to interpolate
                prev_frame = s[4]
                continue

            frame_diff = s[4] - prev_frame
            new_faces = []
            for idx, k in enumerate(range(prev_frame, s[4]-1)):
                new = [0, 0, 0, 0, k+1]
                for l in range(4):
                    new[l] = seqs[i][j - 1][l] + \
                        ((idx + 1) *
                         ((seqs[i][j][l] - seqs[i][j - 1][l]) / frame_diff))
                new_faces.append(new)

            to_add.append((i, j, new_faces))
            prev_frame = s[4]

    for i, j, v in reversed(to_add):
        seqs[i] = seqs[i][:j] + v + seqs[i][j:]

    return seqs
