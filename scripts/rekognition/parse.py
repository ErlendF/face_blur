from os.path import exists
from glob import glob
from os import makedirs
from os.path import join
from cv2 import imread, imwrite, rectangle
from json import loads

from videoface import interpolate, copy_remaining_files

import variables

rekognition_output_file = "/path/to/output.json"
img_dir = "/path/to/imgs"
out_dir = "/out/path"

frame_diff_threshold = 40


def main():
    makedirs(out_dir, exist_ok=True)
    with open(rekognition_output_file, "r") as file:
        data = file.read().replace("'", "\"")   # Converting to valid JSON
    js = loads(data)    # Parsing JSON

    seqs = {}
    rec_seqs = {}

    # Getting metadata
    width = js["VideoMetadata"]["FrameWidth"]
    height = js["VideoMetadata"]["FrameHeight"]
    fps = js["VideoMetadata"]["FrameRate"]

    # Processing the list of identified faces
    for p in js["Persons"]:
        # Converting to (x1, y1, x2, y2) format
        x1 = p["Person"]["Face"]["BoundingBox"]["Left"] * width
        y1 = p["Person"]["Face"]["BoundingBox"]["Top"] * height
        x2 = (p["Person"]["Face"]["BoundingBox"]["Left"] +
              p["Person"]["Face"]["BoundingBox"]["Width"])*width
        y2 = (p["Person"]["Face"]["BoundingBox"]["Top"] +
              p["Person"]["Face"]["BoundingBox"]["Height"])*height

        # Converting from ms to frame number
        frame_nr = int((p["Timestamp"]*fps)//1000)

        # Known faces
        if len(p["FaceMatches"]) != 0:
            if p["Person"]["Index"] not in rec_seqs:
                rec_seqs[p["Person"]["Index"]] = []
            rec_seqs[p["Person"]["Index"]].append(
                {"bbox": [x1, y1, x2, y2, frame_nr], "feat": []})
            continue

        # Unknown faces
        if p["Person"]["Index"] not in seqs:
            seqs[p["Person"]["Index"]] = []
        seqs[p["Person"]["Index"]].append(
            {"bbox": [x1, y1, x2, y2, frame_nr], "feat": []})

    # Converting from maps to lists and splitting sequences with larger differences than the theshold
    seqs_list = []
    for v in seqs.values():
        diffs = []
        last_frame = v[0]["bbox"][4]

        # Checking for differences larger than the threshold
        for i, f in enumerate(v):
            if f["bbox"][4]-last_frame > frame_diff_threshold:
                diffs.append(i)

            last_frame = f["bbox"][4]

        # If there were no large differences, adding the entire sequence
        if len(diffs) == 0:
            seqs_list.append(v)
            continue

        # Adding each sequence
        prev = 0
        for d in diffs:
            seqs_list.append(v[prev:d])
            prev = d

        seqs_list.append(v[prev:])

    rec_seqs_list = []
    for v in rec_seqs.values():
        diffs = []
        last_frame = v[0]["bbox"][4]

        # Checking for differences larger than the threshold
        for i, f in enumerate(v):
            if f["bbox"][4]-last_frame > frame_diff_threshold:
                diffs.append(i)

            last_frame = f["bbox"][4]

        # If there were no large differences, adding the entire sequence
        if len(diffs) == 0:
            rec_seqs_list.append(v)
            continue

        # Adding each sequence
        prev = 0
        for d in diffs:
            rec_seqs_list.append(v[prev:d])
            prev = d

        rec_seqs_list.append(v[prev:])

    seqs_list = interpolate(seqs_list)
    rec_seqs_list = interpolate(rec_seqs_list)

    faces_by_nr = {}
    for seq in seqs_list:
        for face in seq:
            if face["bbox"][4] in faces_by_nr:
                faces_by_nr[face["bbox"][4]].append(
                    (face["bbox"], (0, 0, 255)))
            else:
                faces_by_nr[face["bbox"][4]] = [(face["bbox"], (0, 0, 255))]

    for seq in rec_seqs_list:
        for face in seq:
            if face["bbox"][4] in faces_by_nr:
                faces_by_nr[face["bbox"][4]].append(
                    (face["bbox"], (255, 0, 0)))
            else:
                faces_by_nr[face["bbox"][4]] = [(face["bbox"], (255, 0, 0))]

    for frame_nr, bboxes in faces_by_nr.items():
        file_name = "img" + str(frame_nr+1).rjust(7, '0') + ".png"
        out = join(out_dir, file_name)
        img = imread(join(img_dir, file_name))
        for (bbox, color) in bboxes:
            rectangle(img, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])), color, 2)

        imwrite(out, img)

    copy_remaining_files(img_dir, out_dir)


if __name__ == "__main__":
    main()
