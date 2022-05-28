from videoface import (
    interpolate,
    make_sequences,
    dynamically_process,
    copy_remaining_files,
    get_shot_transitions,
    write_faces,
    read_all_frames
)


def main():
    img_dir = "/mnt/sdb3/erlend/multiface_cut"
    out_dir = "/mnt/sdb3/erlend/test"

    frames = read_all_frames(img_dir)
    shot_transitions = get_shot_transitions(img_dir, frames=frames)
    frames_proc, matchings = dynamically_process(
        img_dir, shot_transitions=shot_transitions, frames=frames)
    seqs = make_sequences(frames_proc, matchings,
                          shot_transitions=shot_transitions)
    seqs = interpolate(seqs)
    write_faces(seqs, img_dir, out_dir, frames=frames)
    copy_remaining_files(img_dir, out_dir)


if __name__ == "__main__":
    main()
