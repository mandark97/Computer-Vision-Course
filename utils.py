import cv2
import numpy as np


def img_to_bin(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, img_bin) = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Invert the image
    img_bin = 255 - img_bin

    return img_bin


def detect_lines(img_bin, kernel_length, kernel_size=(3, 3)):
    # Defining a kernel length
    kernel_length = np.array(img_bin).shape[1]//kernel_length

    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

    return horizontal_lines_img, verticle_lines_img, kernel


def combine_lines(verticle_lines_img, horizontal_lines_img, kernel, alpha=0.5, beta=0.5):
    img_final_bin = cv2.addWeighted(
        verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(
        img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img_final_bin


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def find_sorted_contours(img, method="left-to-right"):
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    return sort_contours(contours, method)


def crop_img(img, box):
    x, y, w, h = box
    return img[y:y+h, x:x+w]


def test(coords):
    sorted_coords = sorted(coords, key=lambda c: (c[1], c[0]))
    arr = np.array(coords)[:, 1]
    arr = sorted(arr)
    distances = []
    for i in range(1, len(arr)):
        distances.append(arr[i] - arr[i-1])

    sorted_indexes = np.argsort(distances)
    sorted_indexes = sorted(sorted_indexes[-15:])

    old_index = 0
    group = {}
    for i in range(len(sorted_indexes)):
        group[i] = sorted_coords[old_index:sorted_indexes[i]+1]
        old_index = sorted_indexes[i] + 1

    group[len(sorted_indexes)] = sorted_coords[old_index: -1]

    for k, v in group.items():
        group[k] = sorted(v, key=lambda x: x[0])

    return group
