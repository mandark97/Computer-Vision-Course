import os
import shutil
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import matplotlib

# filename = 'template_test_grila.jpg'
# filename = 'data/exemple_corecte/image_1.jpg'
filename = 'cropped_info.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Thresholding the image
(thresh, img_bin) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# Invert the image
img_bin = 255 - img_bin
# cv2.imwrite("Image_bin.jpg", img_bin)

# Defining a kernel length
kernel_length = np.array(img).shape[1] // 80

# A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
# A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
# A kernel of (3 X 3) ones.
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Morphological operation to detect vertical lines from an image
img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
# cv2.imwrite("verticle_lines.jpg", verticle_lines_img)
# Morphological operation to detect horizontal lines from an image
img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
# cv2.imwrite("horizontal_lines.jpg", horizontal_lines_img)

# Weighting parameters, this will decide the quantity of an image to be added to make a new image.
alpha = 0.5
beta = 1.0 - alpha
# This function helps to add two image with specific weight parameter to get a third image as summation of two image.
img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
(thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite("img_final_bin.jpg", img_final_bin)


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


# Find contours for image, which will detect all the boxes
contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Sort all the contours by top to bottom.
(contours, boundingBoxes) = sort_contours(contours, method="left-to-right")

cropped_dir_path = "cropped_test/"

idx = 0
rows = defaultdict(list)
print("number of boxes: ", len(contours))
for c in contours:
    # Returns the location and width,height for every contour
    x, y, w, h = cv2.boundingRect(c)
    if (w > 50 and h > 50):
        key = y - (y % 100)
        rows[key].append([x, y, w, h])
        # idx += 1
        # new_img = img[y:y + h, x:x + w]
        # cv2.imwrite(cropped_dir_path + str(idx) + '.png', new_img)

shutil.rmtree(cropped_dir_path)
os.mkdir(cropped_dir_path)

for row, values in rows.items():
    print("row: ", row, "values: ", len(values), np.array(values)[:,1])
    row_img = []
    # max_w = np.max(np.array(values)[:, 2])
    max_h = np.max(np.array(values)[:, 3])
    print("max h", max_h)
    for value in values:
        x, y, w, h = value
        empty_arr = np.full((max_h, w, 3), 0)
        empty_arr[0:h, 0:w] = img[y:y + h, x:x + w]
        row_img.append(empty_arr)
    row_img = np.concatenate(row_img, axis=1)
    cv2.imwrite(cropped_dir_path + str(row) + ".png", row_img)
