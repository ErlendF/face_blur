import cv2
import math
import numpy as np


def alphaBlend(img1, img2, mask):   # source: https://stackoverflow.com/a/48274875
    """ alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)
    """
    if mask.ndim == 3 and mask.shape[-1] == 3:
        alpha = mask/255.0
    else:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
    blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
    return blended


def round_blur(img, bboxes):
    mask = np.zeros(img.shape, dtype='uint8')

    for bb in bboxes:
        h = abs(bb[1] - bb[3])
        w = abs(bb[0] - bb[2])

        circle_center = (int((bb[0] + bb[2]) // 2), int((bb[1] + bb[3]) // 2))
        circle_radius = int(math.sqrt(w * w + h * h) // 2)
        cv2.circle(mask, circle_center, circle_radius,
                   (255, 255, 255), -1, cv2.LINE_AA)

    mask_img = cv2.GaussianBlur(mask, (21, 21), 11)
    img_all_blurred = cv2.medianBlur(img, 99)
    return alphaBlend(img, img_all_blurred, mask_img)
