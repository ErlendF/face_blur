from cv2 import blur, circle, mean
from math import sqrt
from os.path import join
import numpy as np


# Originally based on: https://stackoverflow.com/a/48274875
def alphaBlend(img1, img2, mask):
    alpha = mask/255.0
    return img1*(1-alpha) + img2*alpha


def round_blur(img, bboxes):
    if bboxes is None:
        return img

    mask = np.zeros(img.shape, dtype='uint8')

    blur_width = 0
    for bb in bboxes:
        h = abs(bb[1] - bb[3])
        w = abs(bb[0] - bb[2])
        if w > blur_width:
            blur_width = w

        circle_center = (int((bb[0] + bb[2]) // 2), int((bb[1] + bb[3]) // 2))
        circle_radius = int(sqrt(w * w + h * h) // 2)
        circle(mask, circle_center, circle_radius, (255, 255, 255), -1)

    blur_width = int((blur_width // 3)+1)

    mask_img = blur(mask, (21, 21))
    img_all_blurred = blur(img, (blur_width, blur_width))
    return alphaBlend(img, img_all_blurred, mask_img)


def square_blur(img, bboxes):
    if bboxes is None:
        return img

    for bb in bboxes:
        blur_width = int(abs(bb[1] - bb[3])//3)+1
        img[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])] = blur(
            img[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])], (blur_width, blur_width))

    return img

# Originally based on: https://pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/


def pixelated_blur(img, bboxes, regions=3):
    if bboxes is None:
        return img

    for bb in bboxes:
        if bb[3] < bb[1]:
            bb[1], bb[3] = bb[3], bb[1]
        if bb[2] < bb[0]:
            bb[0], bb[2] = bb[2], bb[0]

        xSteps = np.linspace(bb[0], bb[2], regions + 1, dtype="int")
        ySteps = np.linspace(bb[1], bb[3], regions + 1, dtype="int")
        for i in range(1, len(ySteps)):
            for j in range(1, len(xSteps)):
                startX = xSteps[j - 1]
                startY = ySteps[i - 1]
                endX = xSteps[j]
                endY = ySteps[i]
                img[startY:endY, startX:endX] = mean(
                    img[startY:endY, startX:endX])[:3]

    return img
