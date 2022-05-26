import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis

#! Models are available for non-commercial research purposes only.


def main():
    app = FaceAnalysis(root=os.getenv("MODEL_PATH", "/opt/ml/model"))
    app.prepare(ctx_id=0, det_size=(640, 640))

    in_dir = "/opt/ml/processing/input/"
    out_dir = "/opt/ml/processing/output/"
    os.makedirs(out_dir, exist_ok=True)

    for i, file in enumerate(os.listdir(in_dir)):
        if i % 100 == 0:
            print(f"Processing {i}\n")

        filename = os.fsdecode(file)
        if not filename.endswith(".png"):
            continue

        frame_number = filename.lstrip("img").rstrip(".png")

        img = cv2.imread("{}/{}".format(in_dir, filename))
        faces = app.get(img)
        rimg = app.draw_on(img, faces)
        cv2.imwrite("{}/img{}.png".format(out_dir, frame_number), rimg)


if __name__ == "__main__":
    main()
