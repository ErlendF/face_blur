from videoface import (
    interpolate,
    make_sequences,
    dynamically_process,
    copy_remaining_files,
    get_shot_transitions,
    write_faces,
)
from datetime import datetime



def main():
    img_dir = "/opt/ml/processing/input/"
    out_dir = "/opt/ml/processing/output/"

    start = datetime.now()
    print("Start time", start.strftime("%H:%M:%S"))
    shot_transitions = get_shot_transitions(img_dir)

    start_processing = datetime.now()
    print("Start processing", start_processing.strftime("%H:%M:%S"))
    frames, matchings = dynamically_process(img_dir, shot_transitions=shot_transitions)
    time_diff = datetime.now() - start
    min, sec = divmod(time_diff.days * 24 * 60 * 60 + time_diff.seconds, 60)
    print("Finished processing after {} min and {} sec".format(min, sec))

    seqs = make_sequences(frames, matchings, shot_transitions=shot_transitions)
    seqs = interpolate(seqs)

    write_faces(seqs, img_dir, out_dir)
    copy_remaining_files(img_dir, out_dir)

    time_diff = datetime.now() - start
    min, sec = divmod(time_diff.days * 24 * 60 * 60 + time_diff.seconds, 60)
    print("Finished after {} min and {} sec".format(min, sec))


if __name__ == "__main__":
    main()
