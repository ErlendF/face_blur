from videoface import (
    interpolate,
    make_sequences,
    dynamically_process,
    copy_remaining_files,
    get_shot_transitions,
    write_faces,
)


def main():
    img_dir = "/opt/ml/processing/input/"
    out_dir = "/opt/ml/processing/output/"

    shot_transitions = get_shot_transitions(img_dir)
    frames, matchings = dynamically_process(
        img_dir, shot_transitions=shot_transitions)
    seqs = make_sequences(frames, matchings, shot_transitions=shot_transitions)
    seqs = interpolate(seqs)
    write_faces(seqs, img_dir, out_dir)
    copy_remaining_files(img_dir, out_dir)


if __name__ == "__main__":
    main()
