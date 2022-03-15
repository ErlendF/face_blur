from bisect import insort_right, bisect_left
from os.path import join
from glob import glob

from .file import read_frame
from .dist import compare_faces
from .face_recognition import process, compare


class NextList:
    """
    A sorted list with the ability to get the next element in the list efficiently.

    ...

    Attributes
    ----------


    Returns:
        _type_: _description_

    Yields:
        _type_: _description_
    """

    def __init__(self):
        self.list = []

    def __setitem__(self, key, value, lower=0):
        insort_right(self.list, (key, value), lo=lower, key=lambda f: f[0])

    def __getitem__(self, key, lower=0):
        i = bisect_left(self.list, key, lo=lower, key=lambda f: f[0])
        return self.list[i]

    def __iter__(self):
        for v in self.list:
            yield v

    def next_key(self, key):
        i = bisect_left(self.list, key, key=lambda f: f[0])
        if i == len(self.list)-1:
            return None
        return self.list[i+1][0]


def dynamically_process(img_dir, file_ext="png", batch_size=32, min_interval=6, max_interval=25, proc_count_treshold=6):
    interval = (min_interval + max_interval)//2
    process_consequtively = 0

    current = 0
    prev = -1
    complete = -1
    add_next = 0

    imgs = []
    img_nrs = []
    matchings = {}
    frames_by_nr = NextList()

    # Finding the last frame
    last_frame = sorted(glob(join(img_dir, "*." + file_ext)))[-1]

    # +1 for / and . in filename, -1 to make the frame numbers 0-indexed
    last_frame = int(
        last_frame[len(img_dir)+1+len("img"):len(last_frame)-(len(file_ext)+1)]) - 1

    # Looping until the entire video is processed
    while True:
        if complete == last_frame:  # Completed all frames
            break

        if add_next > last_frame:
            # Reached the last frame, setting add_next to -1 to mark it as done
            add_next = -1
            imgs.append(read_frame(last_frame, img_dir))
            img_nrs.append(last_frame)
        elif add_next >= 0 and len(imgs) < batch_size:
            # Adding the next frame to the list
            imgs.append(read_frame(add_next, img_dir))
            img_nrs.append(add_next)
            add_next += interval

        if len(imgs) >= batch_size or add_next < 0:
            # Processing the queued images
            frames = process(imgs, img_nrs)
            for k, v in frames.items():
                # Storing the processed information for future use
                frames_by_nr[k] = v

            # Cleaning up processed images
            imgs = []
            img_nrs = []

            if current == 0:
                current = frames_by_nr.next_key(0)
                prev = 0

            # Looping until there is a continuity issue
            while True:
                if complete == current:
                    new = frames_by_nr.next_key(current)
                    if new is None:
                        # Processed everything in the queue, increasing the interval
                        interval = min(int(interval*1.5), max_interval)
                        break

                    prev = current
                    current = new

                matches, matched = compare(
                    frames_by_nr[prev], frames_by_nr[current])

                # If the two frames are adjacent, there is no more exploration to do even if they don't match
                if not matched and current != prev+1:
                    # Queueing all frames between the non-matched frames
                    # TODO: more finegrained exploration?
                    for frame_nr in range(prev+1, current):
                        imgs.append(read_frame(frame_nr, img_dir))
                        img_nrs.append(frame_nr)

                    current = prev+1    # Setting back the current
                    # Reducing the interval
                    interval = max(int(interval*0.7), min_interval)
                    process_consequtively = 0
                    break
                else:
                    if current != prev+1:
                        # If enough frames have been successfully processed consequtively, increasing the interval
                        if process_consequtively >= proc_count_treshold:
                            interval = min(int(interval*1.3), max_interval)
                            process_consequtively = 0
                        else:
                            process_consequtively += 1

                    # Storing the matching and updating completed
                    matchings[(prev, current)] = matches
                    complete = current
    return frames_by_nr, matchings


def make_sequences(frames_by_nr, matchings, frame_diff_threshold=40):
    prev = -1
    finished_seqs = []
    seq_mapping = {}

    for frame_nr, faces in frames_by_nr:
        if prev == -1:
            prev = frame_nr
            if len(faces) != 0:
                finished_seqs.append(faces)
            for i in range(len(finished_seqs)):
                seq_mapping[i] = i
            continue

        matching = matchings[(prev, frame_nr)]
        prev = frame_nr
        new_seq_mapping = {}

        used_list = [False]*len(faces)
        for i, m in enumerate(matching):
            if m == -1:
                continue

            used_list[m] = True
            finished_seqs[seq_mapping[i]].append(faces[m])
            new_seq_mapping[m] = seq_mapping[i]

        for i, used in enumerate(used_list):
            if used:
                continue

            finished_seqs.append([faces[i]])
            new_seq_mapping[i] = len(finished_seqs)-1

        seq_mapping = new_seq_mapping

    finished_seqs.sort(key=lambda s: s[0]["bbox"][4])
    map_appended_seqs = [-1] * len(finished_seqs)

    for i in range(len(finished_seqs)):
        for j in range(len(finished_seqs)):
            if i >= j:  # The list is sorted by starting frame
                continue

            if finished_seqs[i][-1]["bbox"][4] < finished_seqs[j][0]["bbox"][4]:
                frame_diff = finished_seqs[j][0]["bbox"][4] - \
                    finished_seqs[i][-1]["bbox"][4]
                if frame_diff > frame_diff_threshold:
                    continue

                likely_same, dist = compare_faces(
                    finished_seqs[i][-1], finished_seqs[j][0])
                if not likely_same:
                    continue

                add_after = []
                better_found = False
                for k in range(len(finished_seqs)):
                    if k <= i or k <= j:
                        continue

                    likely_same, new_dist = compare_faces(
                        finished_seqs[i][-1], finished_seqs[k][0])
                    if likely_same and new_dist < dist and finished_seqs[k][0]["bbox"][4] <= finished_seqs[j][-1]["bbox"][4] and finished_seqs[k][0]["bbox"][4] - finished_seqs[i][-1]["bbox"][4] < frame_diff_threshold:
                        better_found = True  # Found better sequence that doesn't fit with the other proposal
                        break

                if better_found:
                    continue

                if map_appended_seqs[i] != -1:
                    map_appended_seqs[j] = map_appended_seqs[i]
                else:
                    map_appended_seqs[j] = i

                finished_seqs[map_appended_seqs[j]] += finished_seqs[j]

    finished_seqs = [seq for i, seq in enumerate(
        finished_seqs) if map_appended_seqs[i] == -1]

    return finished_seqs
