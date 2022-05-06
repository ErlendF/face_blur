from .dist import compare_faces


def make_sequences(frames_by_nr, matchings, shot_transitions=None, frame_diff_threshold=40):
    prev = -1
    finished_seqs = []
    seq_mapping = {}

    # Making initial facial sequences
    for frame_nr, faces in frames_by_nr:
        # For the first frame, there is no existing sequences to map faces to
        if prev == -1:
            prev = frame_nr
            for i in range(len(faces)):
                finished_seqs.append([faces[i]])

            for i in range(len(finished_seqs)):
                seq_mapping[i] = i
            continue

        matching = matchings[(prev, frame_nr)]
        new_seq_mapping = {}
        used_list = [False]*len(faces)

        # If there is no shot transition, checking matchings between previous and current frame
        if shot_transitions is None or not shot_transitions.between(prev, frame_nr):
            for i, m in enumerate(matching):
                if m == -1:
                    continue

                used_list[m] = True
                finished_seqs[seq_mapping[i]].append(faces[m])
                new_seq_mapping[m] = seq_mapping[i]

        # Making new sequences for every face not mapped to an existing sequence
        for i, used in enumerate(used_list):
            if used:
                continue

            finished_seqs.append([faces[i]])
            new_seq_mapping[i] = len(finished_seqs)-1

        seq_mapping = new_seq_mapping
        prev = frame_nr

    finished_seqs.sort(key=lambda s: s[0]["bbox"][4])
    map_appended_seqs = [-1] * len(finished_seqs)

    # Finding sequences that are likely the same face and combining them
    for i in range(len(finished_seqs)):
        for j in range(len(finished_seqs)):
            # The list is sorted by starting frame
            if i >= j or map_appended_seqs[j] != -1:
                continue

            # Checking for strictly increasing frame numbers
            if finished_seqs[i][-1]["bbox"][4] >= finished_seqs[j][0]["bbox"][4] or (map_appended_seqs[i] != -1 and finished_seqs[map_appended_seqs[i]][-1]["bbox"][4] >= finished_seqs[j][0]["bbox"][4]):
                continue

            frame_diff = finished_seqs[j][0]["bbox"][4] - \
                finished_seqs[i][-1]["bbox"][4]
            if frame_diff > frame_diff_threshold:
                continue

            if shot_transitions is not None and shot_transitions.between(finished_seqs[i][-1]["bbox"][4], finished_seqs[j][0]["bbox"][4]):
                continue

            # Comparing faces to check for possibly same face
            likely_same, dist = compare_faces(
                finished_seqs[i][-1], finished_seqs[j][0])
            if not likely_same:
                continue

            # The faces are likely the same
            # Looking for better matches
            better_found = False
            for k in range(len(finished_seqs)):
                if k <= i or k <= j:
                    continue

                likely_same, new_dist = compare_faces(
                    finished_seqs[i][-1], finished_seqs[k][0])
                if likely_same and new_dist < dist and finished_seqs[k][0]["bbox"][4] <= finished_seqs[j][-1]["bbox"][4] and finished_seqs[k][0]["bbox"][4] - finished_seqs[i][-1]["bbox"][4] < frame_diff_threshold:
                    if shot_transitions is not None and shot_transitions.between(finished_seqs[i][-1]["bbox"][4], finished_seqs[k][0]["bbox"][4]):
                        continue

                    better_found = True  # Found better sequence that doesn't fit with the other proposal
                    break

            if better_found:
                # A better match was found, skipping
                continue

            if map_appended_seqs[i] != -1:
                map_appended_seqs[j] = map_appended_seqs[i]
            else:
                map_appended_seqs[j] = i

            finished_seqs[map_appended_seqs[j]] += finished_seqs[j]

    finished_seqs = [seq for i, seq in enumerate(
        finished_seqs) if map_appended_seqs[i] == -1]

    return finished_seqs
