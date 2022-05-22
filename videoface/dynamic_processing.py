from os.path import join
from glob import glob

from .file import get_file_name
from .dist import compare
from .next_list import NextList
from .deep_face import deep_face_process


def dynamically_process(img_dir, file_ext="png", batch_size=32, min_interval=6, max_interval=25, proc_count_treshold=6, processing_func=deep_face_process, shot_transitions=None):
    # Initially setting the search interval to the middle of the min and max
    interval = (min_interval + max_interval)//2
    process_consequtively = 0

    current = 0  # The frame currently being processed
    prev = -1   # The previous frame that was processed
    complete = -1  # The last completely processed frame
    add_next = 0    # The next frame to add to the list of frames to be processed

    imgs = []   # Images that should be processed
    img_nrs = []  # The corresponding frame numbers
    matchings = {}  # A map of matchings between two frames

    # A map substitute storing the identified bounding boxes and features for each frame
    frames_by_nr = NextList()

    # Finding the last frame
    img_dir = img_dir.removesuffix("/")
    last_frame = sorted(glob(join(img_dir, "*." + file_ext)))[-1]

    # +1 for / and . in filename, -1 to make the frame numbers 0-indexed
    last_frame = int(
        last_frame[len(img_dir)+1+len("img"):len(last_frame)-(len(file_ext)+1)]) - 1

    # Looping until the entire video is processed
    while True:
        if complete == last_frame:  # Completed all frames
            break

        if add_next >= last_frame:
            # Reached the last frame, setting add_next to -1 to mark it as done
            add_next = -1
            imgs.append(get_file_name(last_frame, img_dir))
            img_nrs.append(last_frame)
        elif add_next >= 0 and len(imgs) < batch_size:
            # Adding the next frame to the list
            imgs.append(get_file_name(add_next, img_dir))
            img_nrs.append(add_next)
            if shot_transitions is not None:
                next_sc = shot_transitions.next_key(add_next)
                if next_sc is not None and add_next + interval >= next_sc:
                    imgs.append(get_file_name(next_sc, img_dir))
                    img_nrs.append(next_sc)

                    if next_sc+1 <= last_frame:
                        imgs.append(get_file_name(next_sc+1, img_dir))
                        img_nrs.append(next_sc+1)

            add_next += interval

        if len(imgs) < batch_size and add_next >= 0:
            continue

        # Processing the queued images
        frames = processing_func(imgs, img_nrs)
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
                    imgs.append(get_file_name(frame_nr, img_dir))
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
